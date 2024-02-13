import streamlit as st

st.title("Simple File Upload Test")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.text("File uploaded successfully!")
