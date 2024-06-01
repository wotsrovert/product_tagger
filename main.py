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

embeddings = OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        openai_api_key = openai_api_key
        )

vector_store = PineconeVectorStore(
                                   index_name = PINECONE_INDEX,
                                   embedding=embeddings,
                                   pinecone_api_key=pinecone_api_key
                                   )

prompt = 'List all of the surcharges and taxes added to my rides'

def upload():
    loader = DirectoryLoader(
            "docs",
            glob="*.pdf", loader_cls = PyPDFLoader
            )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
            )

    documents = text_splitter.split_documents( docs )
    print(f"Going to add {len(documents)} to Pinecone index: { PINECONE_INDEX }")
    vector_store.add_documents( documents )

def analyse():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    retriever = vector_store.as_retriever()
    context = retriever.get_relevant_documents( prompt )

    for doc in context:
        print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n")

    print('--------------------------')


    print( f"Prompt: { prompt }" )

    template = PromptTemplate(
            template = "{query} Context: {context}",
            input_variables = ["query", "context"]
            )

    prompt_with_context = template.invoke( {"query": prompt, "context": context} )

    llm = ChatOpenAI(temperature=0.5)
    results = llm.invoke( prompt_with_context )

    print( results.content )

# upload()
analyse()
