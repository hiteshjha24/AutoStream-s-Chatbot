import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma

def setup_rag_retriever():
    """
    Ingests the knowledge.md file and returns a retriever interface.
    Uses Local Embeddings to avoid API Rate Limits.
    """
    # Define persistence directory so we don't re-embed every time
    persist_directory = os.path.join(os.path.dirname(__file__), "../chroma_db")
    file_path = os.path.join(os.path.dirname(__file__), "../data/knowledge.md")
    
    # Initialize Local Embeddings (Runs on your CPU, Free, No Rate Limits)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if we already have the DB
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings,
            collection_name="autostream_knowledge"
        )
    else:
        print("Creating new vector store (this happens once)...")
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="autostream_knowledge",
            persist_directory=persist_directory
        )

    return vectorstore.as_retriever()
