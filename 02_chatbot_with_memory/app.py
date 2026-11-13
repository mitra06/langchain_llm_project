
import os
import uuid
import sqlite3
import logging
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Conversational AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

MAX_HISTORY = 20
MAX_INPUT_LENGTH = 2000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chatbot.db")


# ==========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(r"D:\langchain_llm_project\.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Check your .env file.")
    st.stop()


# ==========================================================
# 3. DATABASE INITIALIZATION
# ==========================================================

def init_db():

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id
            ON chat_messages(conversation_id)
        """)

        connection.commit()

    finally:
        connection.close()


# ==========================================================
# 4. SAVE MESSAGE TO DATABASE
# ==========================================================

def save_message(conversation_id, role, content):

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO chat_messages (
                conversation_id,
                role,
                content
            )
            VALUES (?, ?, ?)
        """, (conversation_id, role, content))

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()


# ==========================================================
# 5. RETRIEVE CONVERSATION HISTORY
# ==========================================================

def get_messages(conversation_id):

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT role, content
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,))

        records = cursor.fetchall()

        return [
            {
                "role": record["role"],
                "content": record["content"]
            }
            for record in records
        ]

    finally:
        connection.close()


# ==========================================================
# 6. RETRIEVE ALL SAVED CONVERSATIONS
# ==========================================================

def get_conversations():

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT conversation_id
            FROM chat_messages
            GROUP BY conversation_id
            ORDER BY MAX(id) DESC
        """)

        return [row[0] for row in cursor.fetchall()]

    finally:
        connection.close()


# ==========================================================
# 7. DELETE FAILED USER MESSAGE
# ==========================================================

def delete_message(message_id):

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            DELETE FROM chat_messages
            WHERE id = ?
        """, (message_id,))

        connection.commit()

    finally:
        connection.close()


# ==========================================================
# 8. INITIALIZE DATABASE
# ==========================================================

init_db()


# ==========================================================
# 9. INITIALIZE OPENAI MODEL
# ==========================================================

@st.cache_resource
def get_llm(api_key):

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        timeout=30,
        max_retries=2,
        api_key=api_key
    )


llm = get_llm(OPENAI_API_KEY)


# ==========================================================
# 10. CREATE LANGCHAIN PROMPT
# ==========================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful and professional AI assistant.

        Answer questions clearly and accurately.

        Use previous conversation messages
        to understand the user's context.

        If you do not know something,
        acknowledge it honestly.
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{question}")

])


# ==========================================================
# 11. CREATE LANGCHAIN CHAIN
# ==========================================================

chain = prompt | llm


# ==========================================================
# 12. INITIALIZE SESSION STATE
# ==========================================================

if "conversation_id" not in st.session_state:

    conversations = get_conversations()

    if conversations:
        st.session_state.conversation_id = conversations[0]
    else:
        st.session_state.conversation_id = str(uuid.uuid4())


if "loaded_conversation_id" not in st.session_state:

    st.session_state.loaded_conversation_id = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================================
# 13. SIDEBAR - CONVERSATION MANAGEMENT
# ==========================================================

st.sidebar.title("💬 Chat History")

if st.sidebar.button("➕ New Chat", use_container_width=True):

    st.session_state.conversation_id = str(uuid.uuid4())

    st.session_state.messages = []

    st.session_state.loaded_conversation_id = (
        st.session_state.conversation_id
    )

    st.rerun()


saved_conversations = get_conversations()

conversation_options = list(saved_conversations)

if st.session_state.conversation_id not in conversation_options:

    conversation_options.insert(
        0,
        st.session_state.conversation_id
    )


selected_conversation = st.sidebar.selectbox(
    "Select Conversation",
    options=conversation_options,
    index=conversation_options.index(
        st.session_state.conversation_id
    ),
    format_func=lambda x: f"Chat {x[:8]}"
)


# Load selected conversation when it changes

if selected_conversation != st.session_state.conversation_id:

    st.session_state.conversation_id = selected_conversation


if (
    st.session_state.loaded_conversation_id
    != st.session_state.conversation_id
):

    st.session_state.messages = get_messages(
        st.session_state.conversation_id
    )

    st.session_state.loaded_conversation_id = (
        st.session_state.conversation_id
    )


# ==========================================================
# 14. MAIN APPLICATION UI
# ==========================================================

st.title("🤖 Conversational AI Chatbot")

st.caption(
    "Powered by OpenAI, LangChain and SQLite"
)

st.info(
    "Your conversations are saved in a local SQLite database."
)


# ==========================================================
# 15. DISPLAY CONVERSATION HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================================
# 16. HANDLE USER INPUT
# ==========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:

    user_input = user_input.strip()

    if not user_input:

        st.warning("Please enter a question.")

    elif len(user_input) > MAX_INPUT_LENGTH:

        st.warning(
            f"Input must be under {MAX_INPUT_LENGTH} characters."
        )

    else:

        conversation_id = st.session_state.conversation_id

        # Save user message in database

        user_message_id = save_message(
            conversation_id,
            "user",
            user_input
        )

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        try:

            # --------------------------------------------------
            # 17. PREPARE CHAT HISTORY
            # --------------------------------------------------

            previous_messages = (
                st.session_state.messages[:-1]
            )

            previous_messages = previous_messages[-MAX_HISTORY:]

            chat_history = []

            for message in previous_messages:

                if message["role"] == "user":

                    chat_history.append(
                        HumanMessage(
                            content=message["content"]
                        )
                    )

                elif message["role"] == "assistant":

                    chat_history.append(
                        AIMessage(
                            content=message["content"]
                        )
                    )

            # --------------------------------------------------
            # 18. GENERATE STREAMING RESPONSE
            # --------------------------------------------------

            with st.chat_message("assistant"):

                response_placeholder = st.empty()

                full_response = ""

                for chunk in chain.stream({

                    "chat_history": chat_history,
                    "question": user_input

                }):

                    if isinstance(chunk.content, str):

                        full_response += chunk.content

                        response_placeholder.markdown(
                            full_response + "▌"
                        )

                response_placeholder.markdown(full_response)

            if not full_response.strip():

                raise ValueError("The model returned an empty response.")

            # --------------------------------------------------
            # 19. SAVE AI RESPONSE IN SQLITE
            # --------------------------------------------------

            save_message(
                conversation_id,
                "assistant",
                full_response
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

            logger.info(
                "Response generated and saved successfully."
            )

        except Exception:

            logger.exception("Chatbot response generation failed.")

            # Remove failed user message from database

            delete_message(user_message_id)

            # Remove failed message from session state

            st.session_state.messages.pop()

            st.error(
                "Unable to generate a response. "
                "Please try again."
            )
