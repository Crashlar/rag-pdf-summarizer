from pathlib import Path
from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def PdfRag(vector_store , path , docloader , my_spllittter):
    
    _ = HuggingFaceEndpoint(
        model = "deepseek-ai/DeepSeek-V3.2", 
        task = "text_generation",
    )
    
    model = ChatHuggingFace(llm = _)
    
    docs = docloader(path)
    
    chunks = my_spllittter(docs)
    
    database = vector_store(chunks)
    
    retriever = database.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    
    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
    )
    
    question = "What is gemini and and how its useful in our work "
    retrieved_docs    = retriever.invoke(question)
    
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    
    main_chain = parallel_chain | prompt | model | parser
    
    print(main_chain.invoke("Who is sundar pichai and how gemini is important for user"))
    

if __name__ == "__main__":
    # files func
    from task.my_docloader import docloader
    from task.my_splitter import docs_splitter
    from task.my_vector_store import embedding_vector_store
    
    # makr dynamic 
    data_dir = Path(__file__).resolve().parent.parent / "data"
    pdf_path = data_dir / "gemini - Google Docs.pdf"
    # print(path)
    
    PdfRag(vector_store=embedding_vector_store , path = pdf_path , docloader=docloader , my_spllittter=docs_splitter)
    