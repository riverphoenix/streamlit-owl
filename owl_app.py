import pandas as pd
import streamlit as st
from streamlit_chat import message
import time


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import requests


def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

sources = [
    get_wiki_data("Unix", False),
    get_wiki_data("Microsoft_Windows", False),
    get_wiki_data("Linux", False),
    get_wiki_data("Seinfeld", False),
    get_wiki_data("Matchbox_Twenty", False),
    get_wiki_data("Roman_Empire", False),
    get_wiki_data("London", False),
    get_wiki_data("Python_(programming_language)", False),
    get_wiki_data("Monty_Python", False),
]

source_chunks = []

splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

chain = load_qa_chain(OpenAI(temperature=0))

def give_answer(question):
    return chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]



# From here down is all the StreamLit UI.
st.set_page_config(page_title="Owl Search", page_icon=":robot:")
st.image("owl.png",width=100)
st.header("Owl Search")
st.text('Please let us know how we can help you build with Twilio')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    # input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return 

user_input = st.text_input(label="You: ",key="input")

if user_input:
    output = give_answer(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")