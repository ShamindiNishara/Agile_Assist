import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Define FAISS index path
faiss_index_path = "faiss_index"

# Load or create FAISS index
if os.path.exists(faiss_index_path):
    vectordb = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
else:
    # Load and split documents if FAISS index does not exist
    loader = PyPDFDirectoryLoader("./pdf_files")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    
    # Create FAISS index from documents and save it locally
    embedding_model = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(documents, embedding_model)
    vectordb.save_local(faiss_index_path)

# Create retriever and tool for document search
retriever = vectordb.as_retriever()
pdf_tool = create_retriever_tool(
    retriever, 
    "pdf_search",
    "Search for information about software process management. For any questions about agile process, you must use this tool first!"
)
tools = [pdf_tool]

# Set up language model and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate and a detailed response based on the question.
<context>
{context}
<context>
Questions: {input}
{agent_scratchpad}
"""
)

# Create agent and executor
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_agent_response(query):
    """Function to get a response from the agent for a given query."""
    try:
        start_overall = time.time()
        start_llm = time.process_time()
        
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        
        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm
        
        return {
            "output": response['output'],
            "response_time_overall": response_time_overall,
            "response_time_llm": response_time_llm
        }
    except Exception as e:
        return {"error": str(e)}


print(get_agent_response("Help me to plan the next retro"))