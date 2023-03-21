import pandas as pd
import streamlit as st
from streamlit_chat import message
import time


from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import requests
import os
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAIChat
from langchain.document_loaders.csv_loader import CSVLoader

pinecone.init(
    api_key = os.environ.get("PINE_API_KEY"),
    environment="us-central1-gcp"
)

search_index = Pinecone.from_existing_index(index_name="langchain-demo", embedding=OpenAIEmbeddings())

llm = OpenAIChat(model_name ="gpt-4",verbose=True,temperature=0)
qa = ChatVectorDBChain.from_llm(llm, search_index)

def add_history(chat_history,query,res,n):
    chat_history.append((query, res))
    if len(chat_history) > n:
        chat_history.pop(0)
    return chat_history

def get_answer(query,chat_history,n):
    result = qa({"question": query, "chat_history": chat_history})
    chat_history=add_history(chat_history,query,result["answer"],n)
    return result

chat_history = []

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Owl Search", page_icon=":robot:")
st.image("owl.png",width=100)
st.header("Owl Search")
st.text('Please let us know how we can help you build with Twilio')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.text_input(label="You: ",key="input")

if user_input:
    output = get_answer(user_input,chat_history,5)['answer']

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")