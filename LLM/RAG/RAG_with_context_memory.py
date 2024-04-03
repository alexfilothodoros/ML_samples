import os
import openai
from llama_index.llms.openai import OpenAI
import openaifrom llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


openai.api_key = "xxxxx"



llm = OpenAI(model="gpt-3.5-turbo")
data = SimpleDirectoryReader(input_dir="./apache_data/").load_data()
index = VectorStoreIndex.from_documents(data)



memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
)


response = chat_engine.chat("Is this license paid?")

print(response)
response = chat_engine.chat("expand more")
print(response)
