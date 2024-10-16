import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os
import requests
import tempfile
import pandas as pd
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.vectorstores import FAISS
from langchain.embeddings import TensorflowHubEmbeddings

# Define parameters
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.STOP_SEQUENCES: ["\n"],
    GenParams.REPETITION_PENALTY: 2
}

load_dotenv()
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

# Select supported model type (fixing the issue)
from ibm_watson_machine_learning.foundation_models import Model
model_id = "meta-llama/llama-3-70b-instruct"  # Use valid model from the supported list

# Initialize the Watsonx foundation model
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
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    embeddings = TensorflowHubEmbeddings(model_url=url)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = llama_model.to_langchain()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def call_model_flan(question):
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 50,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
        GenParams.REPETITION_PENALTY: 1
    }

    # Initialize the Watsonx foundation model
    llm_model = Model(
        model_id="meta-llama/llama-3-405b-instruct", 
        params=parameters, 
        credentials=credentials,
        project_id=project_id
    )

    prompt = f"Considering the following question, generate 3 keywords most significant to use when searching in the Arxiv API. Provide your response as a Python list: {question}."
    result = llm_model.generate(prompt)['results'][0]['generated_text']

    # Convert string to a list of individual words
    word_list = result.split(', ')
    return word_list

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

def download_pdf_files(url_list):
    temp_dir = tempfile.gettempdir()  # Get the temporary directory path
    downloaded_files = []  # List to store downloaded file paths
    for i, url in enumerate(url_list):
        filename = os.path.join(temp_dir, f'file_{i+1}.pdf')  # Set the absolute path in the temporary directory
        download_pdf(url, filename)
        downloaded_files.append(filename)  # Append the file name to the list with the path
        print(f'Downloaded: {filename}')
    return downloaded_files

def delete_files_in_temp():
    temp_dir = tempfile.gettempdir()  # Get the temporary directory path
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def arxiv_search(topic):
    import arxiv
    titles = []
    pdf_url = []
    search = arxiv.Search(
        query=topic,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    titles = [result.title for result in arxiv.Client().results(search)]
    pdf_url = [result.pdf_url for result in arxiv.Client().results(search)]
    url_list = pdf_url
    downloaded_files = download_pdf_files(url_list)
    return downloaded_files, titles

# Function to handle user input and display responses
def handle_user_input(user_question, titles=None):
    prompt = {"question": user_question}
    response = st.session_state.conversation(prompt)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with Research papers :books:")
    user_question = st.text_input("Ask questions to research paper or upload your documents:")
    
    if st.button("Search") and user_question:
        with st.spinner("Analyzing query"):
            original_list = call_model_flan(user_question)
            unique_list = list(set(original_list))
            topic = ' '.join(unique_list)  # full topic creation
        with st.spinner("Searching in Database: " + topic):
            downloaded_files, titles = arxiv_search(topic)
        with st.spinner("Vectorizing results"):
            # Get PDF text and split into chunks
            raw_text = get_pdf_text(downloaded_files)
            text_chunks = get_text_chunks(raw_text)
            # Create vector store and conversation chain
            vectorstore = get_vectorstore(text_chunks)
            st.write("Documents loaded")
            st.session_state.conversation = get_conversation_chain(vectorstore)
            if titles is not None:
                enumerated_strings = [f"{index + 1}. {value}" for index, value in enumerate(titles)]
                combined_string = ', <br> '.join(enumerated_strings)
                st.write(bot_template.replace("{{MSG}}", "Relevant papers found: " + combined_string), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if not pdf_docs:
            st.write('You can add your document')
        else:
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Document loaded")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
    if user_question and st.session_state.conversation is not None:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()
