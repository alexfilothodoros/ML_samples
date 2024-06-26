import os
import openai
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)


load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
depl_name = os.getenv("DEPL_NAME")
embedding_depl_name = os.getenv("emb_depl_name")
model = "gpt-35-turbo-16k"

llm = AzureOpenAI(
    model_name=model,
    engine=depl_name,
    azure_endpoint=openai.api_base,
    api_key=openai.api_key,
    api_type=openai.api_type,
    api_version=openai.api_version,
    temperature=0,
)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="embedding_depl",
    api_key=openai.api_key,
    azure_endpoint=azure_endpoint,
    api_version=openai.api_version,
)
Settings.llm = llm
Settings.embed_model = embed_model

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
            name="query_name",
            description=("Provides information about the apache license. "),
        ),
    )
]


agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)

agent.chat("Does the license allow me to use the software commercially?")
agent.chat(
    "Do not answer if the question is irrelevant to the context. Where is Volos?"
)
