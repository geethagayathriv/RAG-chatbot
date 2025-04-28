import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings 
from huggingface_hub import InferenceClient

load_dotenv()

def get_db_connection(collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Milvus(embeddings, connection_args={
        "user": os.getenv("MILVUS_DB_USERNAME"),
        "password": os.getenv("MILVUS_DB_PASSWORD"),
        "host": os.getenv("MILVUS_DB_HOST"),
        "port": os.getenv("MILVUS_DB_PORT"),
        "db_name": os.getenv("MILVUS_DB_NAME")
    },
    collection_name=os.getenv("MILVUS_DB_COLLECTION"))

def get_similar_docs(query: str):
    vector_db = get_db_connection("my collection")
    return vector_db.similarity_search_with_score(query, k=3)

def fetch_answer_from_llm(query: str):
    llm = HuggingFaceEndpoint(
        repo_id="distilbert/distilgpt2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.5,
        max_new_tokens=200,
        task='text-generation'
    )    
    chain = load_qa_chain(llm, "stuff")
    similar_docs = get_similar_docs(query)
    docs = []
    for doc in similar_docs:
        docs.append(doc[0])
    chain_response = chain.invoke(input={"input_documents": docs, "question": query})
    return chain_response["output_text"]

def generate_answer():
    try:
        query = input("Enter your query: ")
        answer = fetch_answer_from_llm(query)
        print(answer)
        return
    except Exception as exception_message:
        print(str(exception_message))

if __name__ == "__main__":
    generate_answer()