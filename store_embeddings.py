import pandas as pd
import time
import chromadb
from chromadb.utils import embedding_functions
import os

def read_text_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return ""

def main():
    # Initialize the sentence transformer model
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="NovaSearch/stella_en_1.5B_v5"
    )
    
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get the collection
    try:
        collection = client.get_collection(
            name="fontanrag",
            embedding_function=embedding_function
        )
        print("Using existing collection 'fontanrag'")
    except Exception:
        print("Creating new collection 'fontanrag'")
        collection = client.create_collection(
            name="fontanrag",
            embedding_function=embedding_function
        )
    
    # Read the latest notes CSV
    df = pd.read_csv('latest_notes_per_patient.csv')
    
    # Prepend the base path to file names
    df['Filepath'] = 'CardioIMS/Notes/NoteText/' + df['File Name']
    
    # record start time
    start_time = time.time()
    
    # Process each row
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Read the text file
        text = read_text_file(row['Filepath'])
        print(f"Processing {row['Filepath']}")
        
        if text:  # Only process if text was successfully read
            documents.append(text)
            metadatas.append({"filepath": row['Filepath']})
            ids.append(str(idx))
    
    # Add documents to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"Added batch {i//batch_size + 1} to ChromaDB")
    
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    print(f"Total documents processed: {len(documents)}")
    print(f"Collection stored in: {os.path.abspath('./chroma_db')}")

if __name__ == "__main__":
    main() 