import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Define FAISS index path
faiss_index_path = "faiss_index"

# Load or create FAISS index
embedding_model = HuggingFaceEmbeddings()
vectordb = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)


# Set up retriever for searching FAISS index
retriever = vectordb.as_retriever()

# Set up language model and prompt template
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
"""
You are a knowledgeable assistant specializing in software development processes. 
Your role is to provide guidance on various software development practices, methodologies, and lifecycle activities.

When answering questions, follow these guidelines:
1. Refer to the provided context to ensure your answer is precise and relevant.
2. Structure your response to be clear, concise, and actionable.
3. If the context is not sufficient, provide general advice based on best practices in software development.

--- Context for Reference ---
{context}
----------------------------

User Question:
{input}

Additional Notes or Context for You:
{agent_scratchpad}

Instructions:
Provide an answer that directly addresses the user’s question, using the given context when possible. Focus on offering actionable insights and detailed explanations as needed.
"""
)


def get_agent_response(query):
    """Function to get a response from the LLM for a given query."""
    try:
        start_overall = time.time()
        start_llm = time.process_time()
        
        # Retrieve relevant documents from FAISS index
        relevant_docs = retriever.invoke(query)
        
        # Combine retrieved documents into a single context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Prepare prompt with context and input query
        formatted_prompt = prompt.format(context=context, input=query, agent_scratchpad="")
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        
        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm
        
        return {
            "output": response.content,  # Access content directly
            "response_time_overall": response_time_overall,
            "response_time_llm": response_time_llm
        }
    except Exception as e:
        return {"error": str(e)}

# Example usage
print(get_agent_response("Help me to plan the next retro"))
