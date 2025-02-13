from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import warnings
import time

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large",)

index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Initialize Pinecone index
index = pc.Index(index_name)

# Create LangChain components
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

rag_prompt = """Using the given {context}, provide a detailed and accurate answer to the user's question in the question's language. 
Make sure to rely only on the information provided in the {context} without introducing any external knowledge or 
assumptions. Here's the user's question: {question}."""

# Create QA chain
prompt_template = PromptTemplate(template=rag_prompt)

chain = LLMChain(prompt=prompt_template, llm=llm)

def response(query):
    start_time = time.perf_counter()
    docs = vector_store.similarity_search(query, k=4)
    response = chain.run(question=query, context=docs)
    elapsed_time = time.perf_counter() - start_time
    return str(response) , elapsed_time


app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Helloo</h1>"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/webhook', methods=['POST'])

def handle_query():
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"})

    try:
        query = data['query']
        result, time_taken = response(query)
        print(f"Elapsed time: {time_taken:.4f} seconds")
        return result , 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)