import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import os
import asyncio
import torch

# Set page config
st.set_page_config(
    page_title="FontanRAG - Medical Notes Search",
    page_icon="❤️",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="NovaSearch/stella_en_1.5B_v5")

# Initialize ChromaDB client and collection
@st.cache_resource
def get_chroma_collection():
    # Create a persistent client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        # Try to get the existing collection
        collection = client.get_collection(
            name="fontanrag",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="NovaSearch/stella_en_1.5B_v5"
            )
        )
    except Exception as e:
        st.error(f"Error accessing collection: {str(e)}")
        st.info("Please run the store_embeddings.py script first to create the collection.")
        st.stop()
    
    return collection

# Initialize the LLM
@st.cache_resource
def get_llm():
    # Create tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Override the default tokenizer.apply_chat_template method
    original_apply_chat_template = tokenizer.apply_chat_template
    def custom_apply_chat_template(*args, enable_thinking=False, **kwargs):
        return original_apply_chat_template(*args, enable_thinking=enable_thinking, **kwargs)
    tokenizer.apply_chat_template = custom_apply_chat_template

    # Create pipeline for text generation
    pipe = pipeline("text-generation", 
                model="Qwen/Qwen3-8B",
                tokenizer=tokenizer,
                use_fast=True,
                device_map="auto",
                max_new_tokens=512)
    return HuggingFacePipeline(
        pipeline=pipe
    )

# Create the RAG chain
@st.cache_resource
def get_qa_chain():
    embeddings = get_embeddings()
    # collection = get_chroma_collection()
    llm = get_llm()
    
    # Convert ChromaDB collection to LangChain vector store
    vectorstore = Chroma(
        collection_name="fontanrag",
        embedding_function=embeddings,
        persist_directory='./chroma_db'
    )
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# Main app
def main():
    st.title("❤️ FontanRAG - Medical Notes Search")
    st.markdown("""
    This application uses RAG (Retrieval Augmented Generation) to search through medical notes 
    and provide relevant information based on your queries.
    """)
    
    # Initialize the QA chain
    qa_chain = get_qa_chain()
    
    # Chat input
    user_query = st.chat_input("Ask a question about the medical notes...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get response from QA chain
        with st.spinner("Searching through medical notes..."):
            response = qa_chain({"query": user_query})
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["result"]
            })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # For assistant messages, only show the response result
                st.write(message["content"])
                # Show source documents in expander if available
                if "source_documents" in response:
                    with st.expander("View Source Documents"):
                        for doc in response["source_documents"]:
                            st.markdown(f"**Source:** {doc.metadata.get('filepath', 'Unknown')}")
                            st.markdown(doc.page_content)
                            st.markdown("---")
            else:
                # For user messages, show as is
                st.write(message["content"])

if __name__ == "__main__":
    main() 