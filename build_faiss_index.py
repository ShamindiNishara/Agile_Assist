import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Define FAISS index path
faiss_index_path = "faiss_index"

# Initialize variables
valid_docs = []

# Check each file in the pdf_files directory
for filename in os.listdir("./pdf_files"):
    file_path = os.path.join("./pdf_files", filename)
    if not filename.lower().endswith(".pdf"):
        print(f"Skipping non-PDF file: {filename}")
        continue

    try:
        # Check if the file has a valid PDF header
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                print(f"Skipping invalid PDF file: {filename}")
                continue

        # Load document if it is a valid PDF
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        valid_docs.extend(doc)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

# Proceed only if we have valid documents
if valid_docs:
    # Split documents into chunks
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(valid_docs)
    
    # Create FAISS index from documents and save it locally
    embedding_model = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(documents, embedding_model)
    vectordb.save_local(faiss_index_path)
    print(f"FAISS index created and saved to {faiss_index_path}")
else:
    print("No valid documents found. FAISS index was not created.")
