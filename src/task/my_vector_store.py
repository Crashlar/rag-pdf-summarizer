from langchain_community.vectorstores  import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def embedding_vector_store(chunks):
        """
        Create a HuggingFace embedding model.
    
        Args:
                model_name: Name of the embedding model
        **kwargs:
                Additional arguments for HuggingFaceEmbeddings
        
        Returns:
                HuggingFaceEmbeddings instance
    """
        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B"
        )
        vector_store = FAISS.from_documents(chunks, embedding_model)

        return vector_store
    