import os
from dotenv import load_dotenv

from langchain_community.llms.ollama import Ollama
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"

## Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistent, please response to the questions asked."),
        ("user","Question:{Question}")
    ]
)

## streamlit framework

st.title("Langchain demo with Google Gamma 2 model")
input_text = st.text_input("What question do you have in mind?")

## calling Ollama model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"Question":input_text}))