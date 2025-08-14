# app.py

import streamlit as st
import requests

st.set_page_config(page_title="Personal Finance Chatbot")

st.title("💬 Personal Finance Chatbot")

st.sidebar.header("Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

user_type = st.sidebar.radio("I am a", ["student", "professional"])

if uploaded_file:
    with st.spinner("Uploading and processing..."):
        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
        res = requests.post("http://localhost:8000/upload/", files=files)
        if res.status_code == 200:
            context_text = res.json().get("text")
            st.success("File processed successfully!")
        else:
            st.error("File processing failed.")
            context_text = ""
else:
    context_text = ""

if context_text:
    st.subheader("Ask Questions About Your Finances")
    user_query = st.text_input("Your Question")
    if st.button("Get Answer") and user_query:
        with st.spinner("Thinking..."):
            payload = {
                "question": user_query,
                "context": context_text,
                "user_type": user_type
            }
            res = requests.post("http://localhost:8000/ask/", data=payload)
            if res.status_code == 200:
                st.success("Answer:")
                st.write(res.json()["answer"])
            else:
                st.error("Failed to get response.")
