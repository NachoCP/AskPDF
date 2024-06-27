from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def get_document_chunks(pdf_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(pdf_data)
    return docs

def get_vector_store(open_ai_key, pdf_docs):
    
    loader = PyPDFLoader(pdf_docs)
    pdf_data = loader.load()
    
    docs = get_document_chunks(pdf_data)

    embeddings = OpenAIEmbeddings(api_key = open_ai_key)
    # Creates the document retriever using docs and embeddings
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    return retriever


