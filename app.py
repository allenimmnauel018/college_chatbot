import streamlit as st
from chatbot import build_chain, generate_suggestions
import time

st.set_page_config(page_title="College Helpdesk Chatbot", layout="centered")
st.title("🎓 Ask the College Helpdesk Chatbot")

# --- Initialize session state for chat history and components ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "initial_load_done" not in st.session_state:
    st.session_state.initial_load_done = False
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = None


# --- Load Model and Chain (only once) ---
if not st.session_state.initial_load_done:
    with st.spinner("Loading AI model and knowledge base..."):
        try:
            chain_obj = build_chain()
            st.session_state.qa_chain = chain_obj
            # Access the LLM and Retriever from the internal chains of ConversationalRetrievalChain
            st.session_state.llm_model = chain_obj.question_generator.llm
            st.session_state.retriever = chain_obj.retriever

            # Generate initial suggestions (no last_query yet)
            st.session_state.suggestions = generate_suggestions(
                st.session_state.llm_model, st.session_state.retriever, last_query=None
            )
            st.session_state.initial_load_done = True
            st.success("Ready to chat! Ask a question or click a suggestion below.")
        except Exception as e:
            st.error(f"Failed to load chatbot components: {e}. Please check your API key and data.")
            st.stop()


# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Process a Query (updated for ConversationalRetrievalChain) ---
def process_user_query(query_text):
    if not query_text or st.session_state.processing_query:
        return

    # Store the current query as the last_user_query for future suggestions
    st.session_state.last_user_query = query_text

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    st.session_state.processing_query = True

    with st.spinner("Thinking..."):
        if st.session_state.qa_chain:
            # ConversationalRetrievalChain expects 'question' as the input for the new user query.
            # 'chat_history' is handled internally by the chain's memory.
            response = st.session_state.qa_chain.invoke({"question": query_text})
            
            # The output key for the answer from ConversationalRetrievalChain is typically 'answer'
            answer = response["answer"].strip().lower()

            if any(phrase in answer for phrase in ["i don't know", "i do not know", "i'm not sure", "not available", "cannot find"]):
                formatted_answer = "🤖 I'm sorry, I don't know the answer to that based on the available documents."
            else:
                formatted_answer = f"### ✅ Answer\n\n{response['answer']}" # Use 'answer' key

                # Source documents are usually in 'source_documents'
                source_docs = response.get("source_documents", [])
                if source_docs:
                    sources = set(doc.metadata.get("source", "Unknown") for doc in source_docs)
                    formatted_answer += "\n\n---\n**📁 Source document(s):** " + ", ".join(sources)
        else:
            formatted_answer = "Error: Chatbot not fully loaded. Please refresh the page."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
    with st.chat_message("assistant"):
        st.markdown(formatted_answer)

    # Generate NEW suggestions based on the last query after an answer is given
    st.session_state.suggestions = generate_suggestions(
        st.session_state.llm_model, st.session_state.retriever,
        last_query=st.session_state.last_user_query # Pass the last query
    )
    st.session_state.processing_query = False
    st.rerun()


# --- Suggestion Buttons (Above the chat input) ---
if not st.session_state.processing_query and st.session_state.suggestions:
    st.subheader("💡 Suggested Questions:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion_text in enumerate(st.session_state.suggestions):
        with cols[i]:
            if st.button(suggestion_text, key=f"suggestion_{i}", use_container_width=True):
                process_user_query(suggestion_text)


# --- Chat Input Field ---
user_input_query = st.chat_input(
    "Ask a question here...",
    key="chat_input_main"
)

if user_input_query:
    process_user_query(user_input_query)


# --- Footer ---
st.markdown("---")
st.caption("Built with LangChain, Gemini, and Streamlit. © 2025 College Helpdesk Project")