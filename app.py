import streamlit as st
from spellchecker import SpellChecker
from chatbot import build_chain

st.set_page_config(page_title="College Helpdesk Chatbot")
st.title("🎓 Ask the College Helpdesk Chatbot")

# Build QA chain using Gemini
qa_chain = build_chain(model_name="gemini")  # Make sure GeminiLLM is used

# Initialize spell checker (optional)
spell = SpellChecker()

# User input
query = st.text_input("Ask a question:")

# Process the query
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})
        answer = response["result"].strip().lower()

        # Fallback if no confident answer
        if any(phrase in answer for phrase in ["i don't know", "i do not know", "i'm not sure", "not available", "cannot find"]):
            formatted_answer = "🤖 I'm sorry, I don't know the answer to that based on the available documents."
        else:
            formatted_answer = f"### ✅ Answer\n\n{response['result']}"

            # Optional source docs
            source_docs = response.get("source_documents", [])
            if source_docs:
                sources = set(doc.metadata.get("source", "Unknown") for doc in source_docs)
                formatted_answer += "\n\n---\n**📁 Source document(s):** " + ", ".join(sources)

        st.markdown(formatted_answer)

st.markdown("---")
st.caption("Built with LangChain, Gemini, and Streamlit. © 2025 College Helpdesk Project")
