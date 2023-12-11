import streamlit as st

st.set_page_config(page_title="EPLAN", page_icon="📚", layout="wide")


@st.cache_resource
def setup():
    st.title('Instructions')
    st.markdown('You can upload pdf files in the rag page.')
    st.markdown('A FAISS data base will be created and used for informatin retrieval.')
    st.markdown('You can ask questions in the chat page.')
    st.markdown('The chatbot will answer your questions.')
    st.markdown('For now the chatbot will not rembemer your chat history.')
    st.markdown('Please make sure that you have provided an OpenAI API key in the .env file.')
    return None

if __name__ == "__main__":
    setup()
