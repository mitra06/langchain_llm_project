
import os
import time
import logging
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv(r"D:\langchain_llm_project\.env")

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
MAX_HISTORY = 20
MAX_INPUT_LENGTH = 2000

# Reference pricing per 1 million tokens.
# Verify current OpenAI pricing before production billing.
INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60


# ==========================================
# 2. LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 3. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Production AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Production AI Chatbot")
st.caption("Powered by OpenAI + LangChain")


# ==========================================
# 4. API VALIDATION
# ==========================================

if not API_KEY:
    st.error("OpenAI API key missing. Check your .env file.")
    st.stop()


# ==========================================
# 5. INITIALIZE LCEL CHAIN
# ==========================================

@st.cache_resource
def get_chain(api_key):

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        api_key=api_key,
        timeout=30,
        max_retries=2,
        stream_usage=True
    )

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are an expert AI assistant specializing in {domain}.

            Provide:
            - Accurate explanations
            - Clear and concise answers
            - Practical examples where useful
            """
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("human", "{question}")

    ])

    return prompt | llm | StrOutputParser()


try:
    chain = get_chain(API_KEY)

except Exception:
    logger.exception("LLM initialization failed")
    st.error("Unable to initialize the AI model.")
    st.stop()


# ==========================================
# 6. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 7. DOMAIN SELECTION
# ==========================================

domain = st.selectbox(
    "Choose your domain",
    ["Banking", "Data Science", "Generative AI"]
)


# ==========================================
# 8. CLEAR CHAT
# ==========================================

if st.button("Clear Conversation"):

    st.session_state.messages = []
    st.rerun()


# ==========================================
# 9. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# 10. USER INPUT
# ==========================================

user_input = st.chat_input(
    "Ask your question...",
    max_chars=MAX_INPUT_LENGTH
)


# ==========================================
# 11. GENERATE RESPONSE
# ==========================================

if user_input:

    user_input = user_input.strip()

    if not user_input:
        st.warning("Please enter a valid question.")

    else:

        # Prepare conversation history
        chat_history = []

        for message in st.session_state.messages[-MAX_HISTORY:]:

            if message["role"] == "user":

                chat_history.append(
                    HumanMessage(content=message["content"])
                )

            else:

                chat_history.append(
                    AIMessage(content=message["content"])
                )

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # ==================================
        # 12. PERFORMANCE TRACKING
        # ==================================

        start_time = time.perf_counter()

        first_token_time = None

        usage_metadata = {}

        answer = ""

        def response_stream():

            nonlocal_placeholder = None

            for chunk in chain.stream({

                "domain": domain,
                "chat_history": chat_history,
                "question": user_input

            }):

                nonlocal_placeholder = chunk

                yield chunk

        # ==================================
        # 13. STREAM RESPONSE
        # ==================================

        with st.chat_message("assistant"):

            try:

                with st.spinner("AI is thinking..."):

                    # Use the model's direct streaming events
                    for chunk in chain.stream({

                        "domain": domain,
                        "chat_history": chat_history,
                        "question": user_input

                    }):

                        if chunk:
                            if first_token_time is None:
                                first_token_time = (
                                    time.perf_counter() - start_time
                                )

                            answer += chunk

                    total_latency = (
                        time.perf_counter() - start_time
                    )

                st.markdown(answer)

                # Save successful response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                logger.info(
                    "Response generated | TTFT: %.2fs | Total: %.2fs",
                    first_token_time or 0,
                    total_latency
                )

                st.divider()

                st.subheader("Response Performance")

                col1, col2 = st.columns(2)

                col1.metric(
                    "First Token",
                    f"{first_token_time:.2f}s"
                    if first_token_time is not None else "N/A"
                )

                col2.metric(
                    "Total Latency",
                    f"{total_latency:.2f}s"
                )

                # ==================================
                # 14. TOKEN USAGE
                # ==================================

                st.caption(
                    "Token usage requires model usage metadata. "
                    "We'll enable reliable usage collection next."
                )

            except Exception:

                logger.exception("LLM request failed")

                if (
                    st.session_state.messages
                    and st.session_state.messages[-1]["role"] == "user"
                ):
                    st.session_state.messages.pop()

                st.error(
                    "Unable to generate response. Please try again."
                )
