from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os


# work on report for num and char
# improve the quality of data

load_dotenv()

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY"),
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # other params...
)


# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["target", "unique_count", "data_type"],
    template="""
    The target variable in my dataset is named '{target}'.
    Here is some information about it:
    - Number of unique values: {unique_count}
    - Data type: {data_type}

    Based on this information, is this a regression or a classification problem?
    Respond with either "Regression" or "Classification".
    """,
)


# Function to determine model type
def determine_model_type_with_llm(df, target):
    # Get target column data info
    data_type = str(df[target].dtype)
    unique_count = df[target].nunique()

    # Use LLM to analyze and determine model type
    llm_chain = LLMChain(
        prompt=prompt_template, llm=llm  # The correct parameter name is 'prompt'
    )

    response = llm_chain.run(
        {"target": target, "unique_count": unique_count, "data_type": data_type}
    )

    return response.strip()  # Either "Regression" or "Classification"
