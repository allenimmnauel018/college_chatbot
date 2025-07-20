import streamlit as st
from chatbot import build_chain
from gemini_llm import GeminiLLM
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(page_title="🎓 College Helpdesk Chatbot", layout="wide")
st.title("🎓 Ask the College Helpdesk Chatbot")

# --- Cache QA Chain ---
@st.cache_resource
def get_qa_chain():
    logging.info("🔄 Building QA chain (cached)...")
    chain = build_chain()
    if chain is None:
        st.error("❌ Failed to load the knowledge base. Please make sure ingestion has been completed.")
    return chain

qa_chain = get_qa_chain()

# --- Session State (Only chat messages) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Helpers ---
def render_images(images: list[str]) -> str:
    return "".join(
        f'<img src="{img}" width="400px" style="margin-bottom:10px;"><br>'
        for img in sorted(set(images))
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))
    )

def render_documents(documents: list[str]) -> str:
    output = ""
    for doc in sorted(set(documents)):
        if doc.endswith(".pdf"):
            output += f'<iframe src="{doc}" width="100%" height="500px" style="border:1px solid #ccc;"></iframe><br>'
            output += f'<a href="{doc}" download style="color:blue;">📎 Download PDF</a><br><br>'
        elif doc.endswith((".doc", ".docx")):
            output += f'<a href="{doc}" download style="color:blue;">📎 Download DOC/DOCX</a><br>'
        else:
            output += f'- 📄 `{doc}`<br>'
    return output

def render_sources(metadata_list: list[dict]) -> str:
    images, documents, extras = [], [], []
    for meta in metadata_list:
        images.extend(meta.get("images", []))
        documents.extend(meta.get("documents", []))
        src = meta.get("source", "")
        if src and not any(src in s for s in images + documents):
            extras.append(src)

    sections = []
    if images:
        sections.append("🖼️ <strong>Images:</strong><br>" + render_images(images))
    if documents:
        sections.append("📎 <strong>Documents:</strong><br>" + render_documents(documents))
    if extras:
        sections.append("📄 <strong>Other Sources:</strong><br>" + "<br>".join(f"- `{s}`" for s in sorted(set(extras))))
    return "<hr><br>".join(sections)

# --- Show previous chat messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- Chat input box ---
query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not qa_chain:
        assistant_response = "🤖 The chatbot is not ready. Please run ingestion or contact support."
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    else:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"query": query})
                result = response["result"].strip()
                docs = response.get("source_documents", [])

                is_uncertain = "i don't know" in result.lower() or "couldn’t find" in result.lower()
                has_kg_fallback = "Neo4j KG Answer" in result

                if is_uncertain and not has_kg_fallback:
                    output = "🤖 Sorry, I couldn’t find a confident answer in the documents."
                else:
                    output = f"<div style='font-size:17px;'>✅ <strong>Answer:</strong><br>{result}</div>"

                if docs:
                    metadata_list = [doc.metadata for doc in docs]
                    output += f"<br><hr><strong>📚 Sources:</strong><br>{render_sources(metadata_list)}"

                st.session_state.messages.append({"role": "assistant", "content": output})
                with st.chat_message("assistant"):
                    st.markdown(output, unsafe_allow_html=True)

            except Exception as e:
                logging.exception("❌ Chatbot error:")
                error_msg = "⚠️ An error occurred while processing your request."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)

# --- Footer ---
st.markdown("---")
st.caption("🚀 Powered by LangChain, Gemini, ChromaDB, Neo4j, and Streamlit")

# --- Clear chat ---
st.markdown("### 🗑️ Reset Conversation")
if st.checkbox("⚠️ Yes, I want to clear this chat"):
    if st.button("🔄 Confirm Reset"):
        st.session_state.messages = []
        st.success("✅ Conversation cleared.")
