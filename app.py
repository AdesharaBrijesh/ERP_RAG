import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables (fallback if not in st.secrets)
load_dotenv()

# Setup page config
st.set_page_config(page_title="Text-to-SQL Chatbot", page_icon="💬", layout="wide")
st.title("💬 Text-to-SQL Chatbot")
st.markdown("Ask questions in plain English, and the agent will translate them to SQL, execute them against your PostgreSQL database, and return a conversational answer!")

# --- Credential Management ---
def get_credentials():
    try:
        # Try loading from Streamlit secrets first
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
        db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL"))
    except Exception:
        # Fallback to os.environ if st.secrets is not available
        api_key = os.environ.get("GROQ_API_KEY")
        db_url = os.environ.get("DATABASE_URL")
    return api_key, db_url

api_key, db_url = get_credentials()

if not api_key:
    st.error("GROQ_API_KEY is not set. Please set it in Streamlit secrets or environment variables.")
    st.stop()
if not db_url:
    st.error("DATABASE_URL is not set. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# --- Database & LLM Setup ---
@st.cache_resource(show_spinner="Connecting to Database and Initializing Agent...")
def setup_db_and_agent(db_uri, groq_key):
    try:
        # psycopg2 does not support the 'pgbouncer' query parameter often included in Supabase URLs
        clean_db_uri = db_uri.replace("?pgbouncer=true&", "?").replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
        
        # Setup Database Connection
        db = SQLDatabase.from_uri(clean_db_uri)
        
        # Initialize ChatGroq LLM
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama3-70b-8192",  # Using the requested 70b model
            temperature=0,
        )
        
        # Create Toolkit and Agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True
        )
        return agent_executor, db
    except Exception as e:
        st.error(f"Failed to connect to the database or initialize the agent: {e}")
        st.stop()

agent_executor, db = setup_db_and_agent(db_url, api_key)

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
if prompt := st.chat_input("Ask a question about your database..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant"):
        # Use StreamlitCallbackHandler to show the agent's internal thoughts and tool usage
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            with st.spinner("Translating to SQL and querying database..."):
                response = agent_executor.invoke(
                    {"input": prompt},
                    {"callbacks": [st_callback]}
                )
            
            output = response.get("output", "Sorry, I couldn't process that.")
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
            
        except Exception as e:
            error_msg = f"An error occurred during execution: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
