
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint, HuggingFaceEmbeddings
import os
from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv

os.getenv("HF_KEY")
load_dotenv()

PDF_PATH = "tech_support_faqs.pdf"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K = 5

def build_vector_store(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

vectordb = build_vector_store(PDF_PATH)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful technical support assistant. Use the provided context strictly."),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer concisely."),
    ]
)

def _format_docs(docs: List) -> str:
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip())

from langchain_core.runnables import RunnableSequence

format_context = RunnableLambda(lambda docs: {"context": _format_docs(docs)})
inject_question = RunnableLambda(lambda payload: {"question": payload["question"], "context": payload["context"]})

full_rag_chain: RunnableSequence = (
    RunnableLambda(lambda x: x["question"])
    | retriever
    | format_context
    | (lambda ctx: {"question": "", **ctx})
    | inject_question
    | prompt
    | llm
    | StrOutputParser()
)


class Query(BaseModel):
    question: str



st.title("IT Support AI Assistant")

question = st.text_input("Enter your question:")




if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
           answer=full_rag_chain.invoke({"question": question})
        except Exception as e:
            answer = f"Error: {e}"

    st.write("### Answer:")
    st.write(answer)

