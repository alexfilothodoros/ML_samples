import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("openai_key")

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")


def main():

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


if __name__ == "__main__":
    try:
      main()
    except:
      st.error("Something went wrong... Please check your OpenAI API key.")
