import os
import openai
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage

from llama_index.core import (
    VectorStoreIndex,
    PromptHelper,
    SimpleDirectoryReader,
    # load_index_from_storage,
    Settings,
    PromptTemplate,
)



openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
depl_name: str = "gpt-35-dep"
models_list = ["gpt-4-32k", "gpt-35-turbo-16k"]
model = models_list[0] if depl_name == "gpt-35-dep" else models_list[1]


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
    deployment_name="xxxx",
    api_key=openai.api_key,
    azure_endpoint=azure_endpoint,
    api_version=openai.api_version,
)



prompt_helper = PromptHelper(
    context_window=16000, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None
)
file_metadata = lambda x: {"filename": x}
documents = SimpleDirectoryReader("data", file_metadata=file_metadata).load_data()

Settings.llm = llm
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

index = VectorStoreIndex.from_documents(documents)

template = ("We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "You are a helpful assistant, retrieving as much as possible information from the provided context.. \n"
    "Given this context information, please answer the question: {query_str}\n"
    "Use only the given context information and nothing else to answer the question.\n"
    "If you don't know the answer, do not make up anything. Instead respond with 'I apologize, but I have not been given such information.\n"
    "Always reply only in the same language as the given question.\n"
    "Do not mention anything about the location of the answer."
    "Do not add information that is not present in the given context.\n"
    "Do not add any information that is not asked for.\n"
)
qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(
    similarity_top_k=1, llm=llm, text_qa_template=qa_template
)
vector_retriever = index.as_retriever(similarity_top_k=1)


def ask_question(query: str, chat_id: str) -> ChatMessagePayload:
        answer = index.as_query_engine(
                text_qa_template=qa_template,
                llm=llm,
                streaming=True
                ).query(query)
        first_part_of_metadata = list(answer.metadata.items())[0]
        source = first_part_of_metadata[-1]["filename"].split("\\")[-1]

      if "apologize" not in answer.lower():
          first_part_of_metadata = list(response.metadata.items())[0]
          file_source = first_part_of_metadata[-1]["filename"].split("\\")[-1]
          file_content = list(response.source_nodes)[0].node.text
          answer_dict = {
          "answer": str(answer.response_txt),
          "source": source}
      else:
          answer_dict = {
          "answer": str(answer.response_txt),
          "source": 'nan'}
      return answer_dict
