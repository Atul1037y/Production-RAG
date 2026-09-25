import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Production RAG Chatbot")
st.caption("Powered by Gemini and FastAPI")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display user message in UI
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            try:
                # Send the POST request to your FastAPI server
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"query": prompt}
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is your FastAPI server running on port 8000?")