from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Any
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0) 

    def query(self, question: str):
        # 1. Retrieval
        docs = self.vectorstore.similarity_search_with_relevance_scores(question, k=3)
        
        # 2. Hallucination mitigation
        if not docs or docs[0][1] < 0.7:
            return {
                "answer": "I cannot answer this question as the provided document",
                "sources": [],
                "confidence": 0.0
            }
        
        # 3. generation with grounding
        context = "\n\n".join([d[0].page_content for d in docs])
        prompt = f"Using Only the context below, answer the question. Cite the source if possible.\n\nContext: {context}\n\nQuestion: {question}"

        response = self.llm.invoke(prompt)

        # format accoding to task space
        return {
            "answer": response.content,
            "sources": [
                {
                    "document": d[0].metadata.get("document", "Unknown"), 
                    "page": d[0].metadata.get("page", 0),               
                    "chunk": d[0].page_content[:200] + "..."
                } for d in docs
            ],
            "confidence": round(docs[0][1], 2)
        }
    
if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.query(" What is the role of a router chain in agent systems?")
    print(result)