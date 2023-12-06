import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
         separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chucks = text_splitter.split_text(text)
    return chucks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instuctor-xl")
    vectorstore = FAISS.from_texts(text=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="chat", page_icon=":books:")

    st.header("chat :books:")
    st.text_input("Ask question:")

    with st.sidebar:
        st.subheader("your documents")
        pdf_docs = st.file_uploader("upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()