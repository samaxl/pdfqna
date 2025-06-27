import streamlit as st
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import os
import tempfile

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1749680287741-243118ed6b2c?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Suppress non-critical UserWarnings from Pydantic to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Your LLM API details here
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct"

# App title
st.title("🎴 PDF Question Answering Bot ")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type=[".pdf"])

if pdf_file is not None:
    # Load PDF pages
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    # Initialize retriever
    retriever = BM25Retriever.from_documents(pages)

    # Question input
    question = st.text_input("Ask a question based on the PDF")

    if question:
        # Retrieve context
        context_docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # Prompt template
        template = """
        Answer the question based on the context below.
        If you don't know the answer, say 'I don't know.'

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(context=context_text, question=question)

        # LLM model
        model = ChatOpenAI(
            openai_api_key=API_KEY,
            base_url="https://api.sambanova.ai/v1",
            model=MODEL_NAME
        )

        parser = StrOutputParser()

        # Chain: prompt -> model -> parser
        chain = model | parser

        # Get the response
        response = chain.invoke(formatted_prompt)

        # Show the answer
        st.success(response)

