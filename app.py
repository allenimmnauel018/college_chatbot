import streamlit as st
from chatbot import build_chain
from gemini_llm import GeminiLLM
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page setup
st.set_page_config(page_title="College Helpdesk Chatbot", layout="wide")
st.title("🎓 Ask the College Helpdesk Chatbot")

# Cache QA chain
@st.cache_resource
def get_qa_chain():
    logging.info("Attempting to build QA chain (cached).")
    chain = build_chain()
    if chain is None:
        st.error("Failed to load the knowledge base. Please ensure ingestion has been run successfully.")
        logging.error("QA chain could not be built. Vector DB might be missing or corrupted.")
    return chain

qa_chain = get_qa_chain()

# State memory: messages + conversation summary
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# LLM for summarization
llm = GeminiLLM()

def update_summary(history: str, new_user_input: str, new_assistant_output: str):
    prompt = f"""
Update the summary of the conversation between the user and college helpdesk bot.

Conversation so far:
{history}

New user message: {new_user_input}
Bot response: {new_assistant_output}

Return an updated concise summary (omit greetings):
"""
    try:
        return llm._call(prompt).strip()
    except Exception as e:
        logging.error(f"Failed to update summary: {e}")
        return history

def render_sources(sources):
    output = ""
    for src in sources:
        src = src.strip()
        if src.lower().endswith((".jpg", ".png", ".jpeg", ".gif")):
            output += f'<img src="{src}" width="400px" style="margin-bottom:10px"><br>'
        elif src.lower().endswith(".pdf") and "http" in src:
            output += f'<iframe src="{src}" width="100%" height="500px" style="border:1px solid #ccc;"></iframe><br>'
            output += f'<a href="{src}" download style="color:blue;">📎 Download PDF</a><br>'
        elif src.lower().endswith((".doc", ".docx")) and "http" in src:
            output += f'<a href="{src}" download style="color:blue;">📎 Download Document</a><br>'
        else:
            output += f"- 📄 `{src}`<br>"
    return output

# Display past conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

query = st.chat_input("Ask your question...")

if query:
    if qa_chain is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🤖 The chatbot is not ready. Please check the backend or contact support."
        })
        with st.chat_message("assistant"):
            st.markdown("🤖 The chatbot is not ready. Please check the backend or contact support.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            try:
                query_with_context = f"Conversation context:\n{st.session_state.conversation_summary}\n\nUser question: {query}"
                response = qa_chain.invoke({"query": query_with_context})
                result = response["result"].strip()

                if "i don't know" in result.lower() or "i couldn't find" in result.lower():
                    output = "🤖 Sorry, I couldn’t find an answer in the available sources."
                else:
                    output = f"<div style='font-size:17px;'>✅ <strong>Answer:</strong><br>{result}</div>"
                    docs = response.get("source_documents", [])
                    if docs:
                        sources = set()
                        for doc in docs:
                            path = doc.metadata.get("source", "Unknown")
                            if path.startswith("http") or path.startswith("/"):
                                sources.add(path)
                        if sources:
                            output += f"<br><hr><strong>📎 Sources:</strong><br>{render_sources(sorted(sources))}"

                st.session_state.conversation_summary = update_summary(
                    st.session_state.conversation_summary, query, result
                )

                st.session_state.messages.append({"role": "assistant", "content": output})
                with st.chat_message("assistant"):
                    st.markdown(output, unsafe_allow_html=True)

            except Exception as e:
                logging.error(f"Chatbot error: {e}", exc_info=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ An error occurred while processing your request."
                })
                with st.chat_message("assistant"):
                    st.markdown("⚠️ An error occurred while processing your request.")

# Footer and summary
st.markdown("---")
st.caption("🚀 Powered by LangChain, Gemini, FAISS, and Streamlit")
with st.expander("🧠 Conversation Summary"):
    st.markdown(st.session_state.conversation_summary or "_No summary available._")
# Optional: Add reset section after title
st.markdown("### 🗑️ Reset Conversation")

if st.checkbox("⚠️ Yes, I want to clear the current conversation and summary."):
    if st.button("🔄 Confirm Reset"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.success("✅ Conversation and summary cleared.")