import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Load environment variables (fallback if not in st.secrets)
load_dotenv()

# Setup page config
st.set_page_config(page_title="Tassos ERP Chatbot Demo", page_icon="💬", layout="wide")
st.title("💬 Tassos ERP Chatbot Demo")
st.markdown("Ask questions in plain English, and the agent will translate them to SQL, execute them against your PostgreSQL database, and return a conversational answer!")

# --- Credential Management ---
def get_credentials():
    try:
        # Try loading from Streamlit secrets first
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
        db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL"))
        model_name = st.secrets.get("GROQ_MODEL_NAME", os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"))
    except Exception:
        # Fallback to os.environ if st.secrets is not available
        api_key = os.environ.get("GROQ_API_KEY")
        db_url = os.environ.get("DATABASE_URL")
        model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    return api_key, db_url, model_name

api_key, db_url, model_name = get_credentials()

if not api_key:
    st.error("GROQ_API_KEY is not set. Please set it in Streamlit secrets or environment variables.")
    st.stop()
if not db_url:
    st.error("DATABASE_URL is not set. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# --- Database & LLM Setup ---
@st.cache_resource(show_spinner="Connecting to Database and Initializing Chain...")
def setup_db_and_chain(db_uri, groq_key, llm_model_name):
    try:
        # psycopg2 does not support the 'pgbouncer' query parameter often included in Supabase URLs
        clean_db_uri = db_uri.replace("?pgbouncer=true&", "?").replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
        
        # Setup Database Connection
        db = SQLDatabase.from_uri(clean_db_uri)
        
        # Initialize ChatGroq LLM
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name=llm_model_name,
            temperature=0,
        )
        
        # Build the LCEL Chain Manually to avoid langchain.chains import issues
        execute_query = QuerySQLDataBaseTool(db=db)
        
        # 1. SQL Generation Prompt
        sql_prompt = PromptTemplate.from_template(
            "You are a PostgreSQL expert. Given an input question and the chat history, create a syntactically correct PostgreSQL query to run.\n\n"
            "Here is the database schema:\n{table_info}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Question: {question}\n\n"
            "CRITICAL RULES:\n"
            "1. Return ONLY ONE raw SQL query. Do not return multiple queries.\n"
            "2. Do not wrap it in markdown formatting (like ```sql) or include explanations.\n"
            "3. If the user's input is just a conversational pleasantry (like 'hello', 'thanks', 'good job', etc.) and does NOT require querying the database, return exactly the string: NOT_SQL"
        )
        
        # 2. Write Query Chain
        def get_schema(_):
            return db.get_table_info()
            
        write_query = (
            RunnablePassthrough.assign(table_info=get_schema)
            | sql_prompt
            | llm
            | StrOutputParser()
        )
        
        # 3. Final Answer Prompt
        answer_prompt = PromptTemplate.from_template(
            "Given the following chat history, user question, corresponding SQL query, and SQL result, answer the user question naturally.\n"
            "If the SQL Query is 'NOT_SQL', just respond conversationally to the user.\n"
            "CRITICAL: If the SQL Result is empty, an error, or if the question asks for data not present in the tables (like revenue or payments when no such tables exist), politely inform the user that you don't have that information. Do not hallucinate numbers.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Question: {question}\nSQL Query: {query}\nSQL Result: {result}\nAnswer: "
        )
        
        answer_chain = (
            answer_prompt
            | llm
            | StrOutputParser()
        )
        
        return write_query, execute_query, answer_chain, db
    except Exception as e:
        st.error(f"Failed to connect to the database or initialize the chain: {e}")
        st.stop()

write_query, execute_query, answer_chain, db = setup_db_and_chain(db_url, api_key, model_name)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Test database connection and send a welcome message with available tables
    try:
        tables = db.get_usable_table_names()
        welcome_msg = f"Hello! I am connected to your PostgreSQL database. (Available Tables: {', '.join(tables) if tables else 'None found'}). How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    except Exception as e:
        st.error(f"Database connection error: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Execution Loop ---
if prompt := st.chat_input("Ask a question about your database...", max_chars=300):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing..."):
                # Format chat history (excluding the current prompt)
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                
                # 1. Generate SQL (or detect conversational input)
                generated_sql = write_query.invoke({
                    "question": prompt,
                    "chat_history": history_str
                })
                
                # 2. Execute SQL conditionally
                if "NOT_SQL" in generated_sql.strip():
                    sql_result = "No database query was executed because the user input was conversational."
                else:
                    sql_result = execute_query.invoke(generated_sql)
                
                # 3. Generate Final Answer
                output = answer_chain.invoke({
                    "question": prompt,
                    "chat_history": history_str,
                    "query": generated_sql,
                    "result": sql_result
                })
            
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
            
        except Exception as e:
            error_msg = f"An error occurred during execution: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
