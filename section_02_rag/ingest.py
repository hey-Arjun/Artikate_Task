import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv 

load_dotenv()

def ingest_documents(pdf_folders="data/contracts"):
    embeddings = OpenAIEmbeddings()
    # 1 doc injestion
    documents = []
    for filename in os.listdir(pdf_folders):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(pdf_folders, filename)
            doc = fitz.open(doc_path)

            for page_num, page in enumerate(doc):
                text = page.get_text()
                # store metadata for citation
                documents.append({
                    "text": text,
                    "metadata": {"document": filename, "page": page_num + 1}
                })
    # 2 chunking stratergy
    # 512 tookens with 64 token overlap for dense legal context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )
    chunks = []
    for doc in documents:
        texts = text_splitter.split_text(doc["text"])
        for t in texts:
            chunks.append({"text": t, "metadata": doc["metadata"]})
    
    # 3 vector Store
    texts_only = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts_only,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory="./chroma_db"
    ) 
    print("Ingestion complete. Vector store saved to ./chroma_db")

if __name__ == "__main__":
    ingest_documents()