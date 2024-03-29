from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)

load_dotenv()


def loader(indir:str):
    file_metadata = lambda x: {"filename": x}
    documents = SimpleDirectoryReader(
        input_dir=indir,
        file_metadata=file_metadata,
    ).load_data()

    return documents

llm = AzureOpenAI(
    model="xxx",
    deployment_name="xxx",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=xxx",
    api_version= "xxx",
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="xxx",
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint="xxx",
    api_version="xxx",
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 100
Settings.chunk_overlap = 1

documents = loader()

index = VectorStoreIndex.from_documents(documents)
persist_dir = "xxx"

if not os.path.exists(persist_dir):
    index.storage_context.persist(persist_dir=f"{persist_dir}")
    print("Index has been saved")
else:
    print("Index already exists")

def load_custom_index(persist_dir:str):
  storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
  index2 = load_index_from_storage(storage_context)
