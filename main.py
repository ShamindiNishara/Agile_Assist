import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
import time

from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

embedding_model = HuggingFaceEmbeddings()

loader = PyPDFDirectoryLoader("./pdf_files")
docs = loader.load()

#Splitting the content into chunks
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

#Storing chunks into vector DB
#vectordb = Chroma.from_documents(documents, OpenAIEmbeddings())
vectordb = FAISS.from_documents(documents, embedding_model)

retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
pdf_tool = create_retriever_tool(retriever, "pdf_search",
                     "Search for information about software process management. For any questions about agile process, you must use this tool first!")

tools = [pdf_tool]


#Streamlit setup
st.title("Agile Assist")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate and a detailed response based on the question
<context>
{context}
<context>
Questions:{input}
{agent_scratchpad}
"""
)

from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Agent Executer
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)


query = st.text_input("Input your query here")

if (st.button("Get Answer") or query):
    start_overall = time.time()
    start_llm = time.process_time()
    #response = agent_executor.invoke({"input": query})
    try:
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm
        st.write(response['output'])
        #st.write(f"Response time: {response_time} seconds")
        st.markdown(f"<p style='color:blue;'>Overall Response Time: {response_time_overall} seconds</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:blue;'>LLM Response Time: {response_time_llm} seconds</p>", unsafe_allow_html=True)

    except Exception as e:
        #st.markdown(f"<p style='color:red;'>Please enter a valid query!</p>", unsafe_allow_html=True)
        st.write(f"An error occurred: {e}")

else:
    st.write("Please enter a query")