import os
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False


engine = index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="query_engine_name",
            description=("Provides information about the apache license. "),
        ),
    )
]


agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)

agent.chat("Does the license allow me to use the software commercially?")
agent.chat(
    "Do not answer if the question is irrelevant to the context. Where is Volos?"
)
