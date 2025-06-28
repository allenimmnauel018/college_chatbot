import streamlit as st
from spellchecker import SpellChecker
from chatbot import build_chain

st.set_page_config(page_title="College Helpdesk Chatbot")
st.title("🎓 Ask the College Helpdesk Chatbot")

# Build QA chain with selected model
qa_chain = build_chain(model_name="phi")

# Initialize spell checker
spell = SpellChecker()

# Spell correction function
def correct_spelling(text):
    corrected = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected.append(corrected_word if corrected_word else word)
    return " ".join(corrected)

# User input
query_raw = st.text_input("Ask a question:")
query = correct_spelling(query_raw)

# Show correction if spelling was changed
if query_raw != query:
    st.caption(f"🛠 Corrected query: `{query}`")

# Process the query
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})
        answer = response["result"].strip().lower()

        # Check for uncertain answers
        if any(phrase in answer for phrase in ["i don't know", "i do not know", "i'm not sure", "not available", "cannot find"]):
            formatted_answer = "🤖 I'm sorry, I don't know the answer to that based on the available documents."
        else:
            formatted_answer = f"### ✅ Answer\n\n{response['result']}"

            # Show source documents (if available)
            source_docs = response.get("source_documents", [])
            if source_docs:
                sources = set(doc.metadata.get("source", "Unknown") for doc in source_docs)
                formatted_answer += "\n\n---\n**📁 Source document(s):** " + ", ".join(sources)

        st.markdown(formatted_answer)

st.markdown("---")
st.caption("Built with LangChain, Ollama, and Streamlit. © 2025 College Helpdesk Project")
