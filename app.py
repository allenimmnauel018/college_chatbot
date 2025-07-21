# app.py

import streamlit as st
from chatbot import build_chain
import logging

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="College Helpdesk Chatbot", layout="wide")
st.title("🎓 Ask the College Helpdesk Chatbot")

# Load QA chain
@st.cache_resource
def get_qa_chain():
    logging.info("Attempting to build QA chain (cached).")
    chain = build_chain()
    if chain is None:
        st.error("Knowledge base not found. Please run ingestion first.")
        logging.error("QA chain could not be built.")
    return chain

qa_chain = get_qa_chain()

# Track chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Input
query = st.chat_input("Ask your question...")

if query:
    if qa_chain is None:
        reply = "🤖 The chatbot is not ready. Please check the backend or contact support."
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                answer = result["result"].strip()

                if "i don't know" in answer.lower() or "i couldn't find" in answer.lower():
                    reply = "🤖 Sorry, I couldn’t find an answer in the available sources."
                else:
                    reply = f"<div style='font-size:17px;'>✅ <strong>Answer:</strong><br>{answer}</div>"

                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                logging.error(f"Chatbot error: {e}", exc_info=True)
                reply = "⚠️ An error occurred while processing your request."
                st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🚀 Powered by LangChain, Gemini, FAISS, and Streamlit")

# Reset
st.markdown("### 🗑️ Reset Chat")
if st.checkbox("Yes, clear current conversation"):
    if st.button("🔄 Confirm Reset"):
        st.session_state.messages = []
        st.success("✅ Chat reset.")
