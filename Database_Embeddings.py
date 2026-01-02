import streamlit as st
from google import genai
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3
import glob

def create_db():
    conn = sqlite3.connect("File_paths.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS File_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            File_path TEXT UNIQUE
        )
    """)
    conn.commit()

def getting_pdf_files():
    folder_path = Path(__file__).parent/"Notes"
    folder_path = folder_path.resolve()

    pdf_files = glob.glob(f"{folder_path}/*.pdf")

    conn=sqlite3.connect("File_paths.db")
    cursor=conn.cursor()

    cursor.execute("SELECT * FROM File_paths")
    already_embedded_files_RAW=cursor.fetchall()
    already_embedded_files=[]
    for l in already_embedded_files_RAW:
         already_embedded_files.append(l[1])
    print(already_embedded_files,already_embedded_files_RAW)


    for i in pdf_files:
        try:
            cursor.execute("INSERT INTO File_paths (File_path) VALUES (?)", (i,))
            conn.commit()
        except sqlite3.IntegrityError:
                pass
    
    files_to_embed=[]
    for i in pdf_files:
        if i not in already_embedded_files:
                files_to_embed.append(i)
    return files_to_embed
      
create_db()
File_paths=getting_pdf_files() 

for File_path in File_paths:
    loader=PyPDFLoader(file_path=File_path)
    docs=loader.load()
    splitter= RecursiveCharacterTextSplitter(
    chunk_size=75,
    chunk_overlap=25
    )
    Chunk=splitter.split_documents(documents=docs)
    Embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------Vector Database Setup----------

    client=QdrantClient(url="http://localhost:6333")
    vector_store=QdrantVectorStore(
        client=client,
        embedding=Embedding_model,
        collection_name="Notes",
        distance="Cosine"
    )
    vector_store.add_documents(Chunk)
    print("Embedding done!")



