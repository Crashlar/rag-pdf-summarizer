"""
RAG (Retrieval-Augmented Generation) System for PDF Document Question Answering

This module implements a RAG pipeline that:
1. Loads PDF documents
2. Splits them into chunks
3. Creates vector embeddings and stores them in FAISS
4. Retrieves relevant context based on user queries
5. Generates answers using a HuggingFace language model

The system is designed to be modular and configurable for different:
- Document loaders
- Text splitters
- Vector stores
- Language models
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()











def create_retrieval_chain(
    retriever: Any,
    llm: ChatHuggingFace,
    prompt_template: Optional[PromptTemplate] = None
) -> Any:
    """
    Create a RAG retrieval and generation chain.
    
    Args:
        retriever: Document retriever instance
        llm: Language model instance
        prompt_template: Custom prompt template
        
    Returns:
        Runnable chain for RAG
    """
    if prompt_template is None:
        prompt_template = create_rag_prompt()
    
    # Create the parallel chain for context retrieval
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_retrieved_docs),
        'question': RunnablePassthrough()
    })
    
    # Create the main RAG chain
    return (
        parallel_chain
        | prompt_template
        | llm
        | StrOutputParser()
    )


def PdfRag(
    vector_store_creator: Callable,
    path: Path,
    docloader: Callable,
    text_splitter: Callable,
    query: Optional[str] = None,
    llm_config: Optional[Dict] = None,
    retrieval_config: Optional[Dict] = None
) -> str:
    """
    Main function to execute the RAG pipeline on a PDF document.
    
    Args:
        vector_store_creator: Function to create vector store (should accept documents and return vector store)
        path: Path to the PDF file
        docloader: Function to load documents
        text_splitter: Function to split documents (should return RecursiveCharacterTextSplitter instance)
        query: Question to ask (if None, uses default)
        llm_config: Configuration for language model
        retrieval_config: Configuration for retrieval
        
    Returns:
        Generated answer as string
    """
    # Set default configurations
    if llm_config is None:
        llm_config = {
            "model_id": "deepseek-ai/DeepSeek-V3.2",
            "task": "text_generation"
        }
    
    if retrieval_config is None:
        retrieval_config = {
            "search_type": "similarity",
            "search_kwargs": {"k": 4}
        }
    
    # Initialize components
    llm = create_llm(**llm_config)
    
    # Load and process documents
    print(f"Loading document from: {path}")
    docs = docloader(str(path))
    
    print(f"Splitting document into chunks...")
    chunks = text_splitter(docs)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store using the provided vector_store_creator function
    print("Creating vector store...")
    database = vector_store_creator(chunks)
    
    # Create retriever
    retriever = database.as_retriever(**retrieval_config)
    
    # Create RAG chain
    print("Creating RAG chain...")
    rag_chain = create_retrieval_chain(retriever, llm)
    
    # Execute query
    if query is None:
        query = "Who is Sundar Pichai and how is Gemini important for users?"
    
    print(f"Querying: {query}")
    result = rag_chain.invoke(query)
    
    return result


def main():
    """Main execution function with dynamic imports."""
    # Dynamically import custom modules
    try:
        from task.my_docloader import docloader
        from task.my_splitter import docs_splitter
        from task.my_vector_store import embedding_vector_store
    except ImportError as e:
        print(f"Warning: Could not import custom modules: {e}")
        print("Using default implementations...")
        
        # Define fallback functions
        def docloader(path: str) -> List[Document]:
            """Default PDF loader."""
            loader = PyPDFLoader(path)
            return loader.load()
        
        def docs_splitter(docs: List[Document]) -> List[Document]:
            """Default text splitter using RecursiveCharacterTextSplitter."""
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return splitter.split_documents(docs)
        
        # Create embedding model for fallback vector store
        def embedding_vector_store(
            chunks: List[Document]
        ) -> FAISS:
            """
            Default vector store creator.
            
            Args:
                chunks: List of Document objects to index
                
            Returns:
                FAISS vector store instance
            """
            embedding_model = create_embedding_model()
            return FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model
            )
    
    # Setup paths dynamically
    data_dir = Path(__file__).resolve().parent.parent / "data"
    pdf_path = data_dir / "gemini - Google Docs.pdf"
    
    # Verify PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please ensure the file exists in the data directory.")
        return
    
    print(f"Using PDF: {pdf_path.name}")
    
    # Execute RAG pipeline
    try:
        answer = PdfRag(
            vector_store_creator=embedding_vector_store,
            path=pdf_path,
            docloader=docloader,
            text_splitter=docs_splitter,
            query="Who is Sundar Pichai and how is Gemini important for users?"
        )
        
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(answer)
        print("="*50)
        
    except Exception as e:
        print(f"Error during RAG execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()