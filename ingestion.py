import os
import time
from dotenv import load_dotenv
import nltk
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download the necessary NLTK resource
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Initialize Pinecone using the Pinecone class
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Set your index name
index_name = 'ap-v0'

# Check if the index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=os.environ.get("PINECONE_ENVIRONMENT")  # or your specific region, e.g., 'us-west-2'
        )
    )

def main():
    print("Going to ingest documentation...")

    # Load documents using LangChain's DirectoryLoader
    dir_loader = DirectoryLoader("./docs")
    start_time = time.time()
    documents = dir_loader.load()
    end_time = time.time()
    print(f"Loaded {len(documents)} documents in {end_time - start_time} seconds")

    # Initialize OpenAI LLM and Embeddings
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))

    # Initialize Pinecone vector store
    vector_store = LangchainPinecone.from_documents(documents=[], embedding=embeddings, index_name=index_name)

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    for idx, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc.page_content)
        metadatas = [{'source': doc.metadata['source']} for _ in chunks]
        vector_store.add_texts(chunks, metadatas=metadatas)
        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(documents)} documents")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    print("Finished ingesting...")

    # Example usage of the QA chain
    query = "What is Pinecone?"
    response = qa_chain.invoke(query)
    print(response)

if __name__ == "__main__":
    main()
