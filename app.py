import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = OllamaLLM(model="llama3.2:3b")

st.title("Enterprise RAG Chatbot")

question = st.text_input("Ask a Question")

if question:
    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer only using the given context.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)

    st.write(answer)