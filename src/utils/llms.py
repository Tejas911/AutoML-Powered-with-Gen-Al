import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Access environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def get_gemini():
    # Initialize LLMs
    llm_google = ChatGoogleGenerativeAI(
        # model="gemini-1.5-flash",
        model="gemini-1.5-pro",
        # temperature=0.4,
        temperature=0,
        google_api_key=google_api_key,
    )

    return llm_google


def get_llama3_8b():
    llm_llama3 = ChatGroq(
        model="llama3-8b-8192",
        # temperature=0.4,
        temperature=0,
        api_key=groq_api_key,
    )

    return llm_llama3
