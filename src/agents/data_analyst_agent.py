import time
import streamlit as st
import numpy as np
import faiss
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Improve prompt template
# TRY using more powerful LLM
# change embedding to huggingface

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize LLMs
llm_google = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    # temperature=0.4,
    temperature=0,
    google_api_key=google_api_key,
)

llm_llama3 = ChatGroq(
    model="llama3-8b-8192",
    # temperature=0.4,
    temperature=0,
    api_key=groq_api_key,
)

# Initialize embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# loading the embedding model from huggingface
# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": "cuda"}
# embeddings = HuggingFaceEmbeddings(
#     model_name=embedding_model_name,
#     model_kwargs=model_kwargs,
#     api_key=os.environ["HF_TOKEN"],
# )


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)


# Function to generate responses from the LLM
def generate_response(user_input, chat_history, retriever):
    # Retrieve relevant documents from the vector database
    relevant_docs = retriever.get_relevant_documents(user_input)

    # Combine retrieved documents for context
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    template = """
    Role: "You are a friendly and helpful AI assistant specialized in Data Analysis. Your primary task is to provide accurate and relevant information based on the user's needs."
    Name: "Hannah Baker"

    You should avoid repeating your name in each response unless explicitly asked for it, and focus on keeping the conversation flowing naturally.
    
    Use a conversational and friendly tone. Be concise when necessary, but ensure clarity.
    
    Chat History:
    {chat_history}

    Additional Context (if relevant):
    {retrieved_context}
    
    User: {user_input}
    AI: """

    # Create an LLMChain with OpenAI
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input", "retrieved_context"],
        template=template,
    )
    llm_chain = LLMChain(llm=llm_llama3, prompt=prompt)

    # Generate a response
    response = llm_chain.run(
        {
            "chat_history": chat_history,
            "user_input": user_input,
            "retrieved_context": retrieved_context,
        }
    )

    return response


# Function to initialize the vector database from TEXT_DATA
def initialize_vectordb(TEXT_DATA):
    # Split the TEXT_DATA into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(TEXT_DATA)

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a progress bar for user feedback
    progress_bar = st.progress(0)
    total_chunks = len(text_chunks)

    # Create a FAISS vector store from the text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Simulate progress for visual feedback
    for i, _ in enumerate(text_chunks):
        time.sleep(0.1)  # Simulating some delay for progress bar
        progress_bar.progress((i + 1) / total_chunks)

    # Save the FAISS index locally (optional)
    vector_store.save_local("faiss_index")
    st.success("Vector store created and saved successfully.")

    return vector_store.as_retriever()


def Hannah_Baker_chatbot_ui(TEXT_DATA: str):
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize vector database for retrieval
    if "retriever" not in st.session_state:
        st.session_state.retriever = initialize_vectordb(TEXT_DATA)

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        role, content = message.split(":", 1)
        with st.chat_message(role.strip().lower()):
            st.markdown(content.strip())

    # React to user input
    if user_input := st.chat_input("You: "):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)

        # Append the user message to chat history
        st.session_state.chat_history.append(f"User: {user_input}")

        # Generate response using RAG
        response = generate_response(
            user_input,
            "\n".join(st.session_state.chat_history),
            st.session_state.retriever,
        )

        # Display AI response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add the AI response to chat history
        st.session_state.chat_history.append(f"AI: {response}")
