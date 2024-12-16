import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    pt="""
    answer the question from the provided context in as detailed way as possible. respond with "no response generated" if you are not able to generate a response\n\n
    Context:\n {context}? \n
    Question:\n {question} \n

    Answer:


    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt= PromptTemplate(template=pt, input_variables=["context", "question"])
    conchain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return conchain


def user_ip(user_ques):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")


    newdb=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs=newdb.similarity_search(user_ques)

    conchain=get_conversational_chain()


    response=conchain(
        {"input_documents":docs, "question": user_ques}
        , return_only_outputs=True
    )

    print("Retrieved Documents:", docs)

    print(response)
    st.write("Reply:", response["output_text"])

    # return response.get("output_text", "No response generated")


def main():
        st.title("Ask Your PDFs: AI-Powered Document Insights 📝✨")

    # PDF File Upload
        pdf_docs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

        if pdf_docs:
         with st.spinner("Processing PDFs..."):
            text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            st.success("PDFs processed and indexed!")

        # User Question Input
        user_question = st.text_input("Ask a question about the PDF content:")
        
        if user_question:
            with st.spinner("Getting the answer..."):
                response = user_ip(user_question)
                # st.write("Reply:", response)


if __name__ == "__main__":
    main()










