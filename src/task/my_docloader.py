from langchain_community.document_loaders import PyPDFLoader
def docloader(path):
        """
        Load the document if format is pdf 
        """
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs