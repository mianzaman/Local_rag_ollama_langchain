import streamlit as st
import requests
import json

# Constants
API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_answer(question):
    try:
        response = requests.post(
            f"{API_URL}/api/chat",
            headers=HEADERS,
            json={"question": question}
        )
        if response.status_code == 200:
            return response.json()["answer"]
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

st.set_page_config(page_title="Document Q&A System", layout="wide")
st.title("Document Question & Answer System")

# API Health Check
api_status = "🟢 Online" if check_api_health() else "🔴 Offline"
st.sidebar.write(f"API Status: {api_status}")

# Main Interface
question = st.text_input("Enter your question:")

if st.button("Ask Question"):
    if question:
        with st.spinner("Getting answer..."):
            answer = get_answer(question)
            st.write("### Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")

# Display Chat History
if 'history' not in st.session_state:
    st.session_state.history = []

if question and st.button("Clear History"):
    st.session_state.history = []

# Show history
if st.session_state.history:
    st.write("### Chat History")
    for q, a in st.session_state.history:
        st.text(f"Q: {q}")
        st.text(f"A: {a}")
        st.markdown("---")


# streamlit run frontend.py
