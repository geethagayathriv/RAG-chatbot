import os
from dotenv import load_dotenv
import streamlit as st
from llm import fetch_answer_from_llm

load_dotenv()

st.set_page_config(page_title= "Chatbot")

st.markdown("Insurance help chatbot")
st.markdown("Ask your questions:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Type your question...")


if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    with st.spinner("Thinking..."):
        answer = fetch_answer_from_llm(user_query)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
else:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])