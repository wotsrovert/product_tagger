import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from constants import *

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# the prompt: we will be changing this soon
prompt = "hello sailor!"

# Note: we must use the same embedding model that we used when uploading the docs
# Querying the vector database for "relevant" docs then create a retriever
# create a context by using the retriever and getting the relevant docs based on the prompt
# show the thought process by looping over all relevant docs, showing the source and the content
# build a prompt template using the query and the context and build the prompt with context
# Asking the LLM for a response from our prompt with the provided context using CatOpenAI and invoking it
# Then print the results content


loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=100 )
documents = text_splitter.split_documents( raw_docs )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=openai_api_key)

print(f"Going to add {len(documents)} to Pinecone")

# print( prompt )
