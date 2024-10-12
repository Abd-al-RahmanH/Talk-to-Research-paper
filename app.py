import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
import requests
import os
import tempfile

# Load environment variables
load_dotenv()

# IBM Watson credentials
project_id = os.getenv("PROJECT_ID", None)
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("API_KEY", None)
}

# Function to get bearer token
def getBearer(apikey):
    form = {'apikey': apikey, 'grant_type': "urn:ibm:params:oauth:grant-type:apikey"}
    response = requests.post("https://iam.cloud.ibm.com/oidc/token", data=form)
    if response.status_code != 200:
        raise Exception("Failed to get token, invalid status")
    return response.json().get("access_token")

credentials["token"] = getBearer(credentials["apikey"])

# Use a supported model
model_id = ModelTypes.LLAMA_3_70B_INSTRUCT

# Initialize Watsonx foundation model
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.STOP_SEQUENCES: ["\n"],
    GenParams.REPETITION_PENALTY: 2
}

llama_model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id
)

# Function to get text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text += " ".join(page.extract_text() for page in pdf_reader.pages)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create a vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = llama_model.to_langchain()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

# Main function
def main():
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.header("Chat with Research Papers :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask questions to research paper or upload your documents:")

    if st.button("Search") and user_question:
        with st.spinner("Processing your question..."):
            prompt = {"question": user_question}
            response = llama_model.generate(prompt)
            st.write(response)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if pdf_docs:
            if st.button("Process"):
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.write("Documents loaded")

if __name__ == "__main__":
    main()
