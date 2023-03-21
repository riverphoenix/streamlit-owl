from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chat_models import ChatOpenAI

# Load the blogposts (big file that was scraped)
loader = CSVLoader(file_path='blogposts.csv')
data = loader.load()

# Split it into chunks
source_chunks = []

splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

for source in data:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

# Save embeddings into Pinecone
pinecone.init(
    api_key = os.environ.get("PINE_API_KEY"),
    environment="us-central1-gcp"
)

# Run this when you first want to save embeddings into Pinecone

# pinecone.delete_index("langchain-demo")
# pinecone.create_index("langchain-demo", dimension=1536)
# search_index = Pinecone.from_documents(source_chunks, OpenAIEmbeddings(), index_name="langchain-demo")


# Run this afterwards since embeddings already in Pinecone
search_index = Pinecone.from_existing_index(index_name="langchain-demo", embedding=OpenAIEmbeddings())

# Create a prompt using templates
prompt_template = """You are an AI assistance that helps users answer questions regarding Twilio
Use the context below to provide an answer on the topic. If the topic is not related to Twilio or any of the Twilio products then respond with 'Sorry but this question is not reltated to Twilio' 
    
    Context: {context}
    
    Topic: {topic}
    
    Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

# Create the model - you can use gpt3.5 or gpt4 (4 is slower)
llm = ChatOpenAI(model_name ="gpt-3.5-turbo",verbose=True,temperature=0,max_tokens=2048)
chain = LLMChain(llm=llm, prompt=PROMPT)

# Find docs based on similarity (k will provide broader context)
def generate_text(topic):
    docs = search_index.similarity_search(topic, k=10)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    return chain.apply(inputs)[0]['text']

print(generate_text("Help me create a proxy with Twilio"))