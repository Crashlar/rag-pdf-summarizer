from langchain_text_splitters import RecursiveCharacterTextSplitter

def docs_splitter(docs):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200 , 
        chunk_overlap = 0 , 
        separators=["/n" , "/n/n" , "."]
        )


        chunks = splitter.split_documents(docs)
        return chunks