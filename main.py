from qdrant_client import QdrantClient
from qdrant_client.http import models
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
import subprocess
from pathlib import Path
import sys

# ------------ Defining AI Mode --------

def AI_Mode():

    st.title("Personal Tutor RAG Agent")
    st.subheader("Ask Anything from your notes")

# ----- Getting User Query -----

    Chat=st.chat_input("Ask!")

# ------ Giving the Bot MEMORY -----

    if "Previous_Chat" not in st.session_state:
        st.session_state.Previous_Chat=[]
    if Chat:
        st.session_state.Previous_Chat.append(Chat)

    Memory = "\n".join(x for x in st.session_state.Previous_Chat if x)

# ---------- Setting up the Vector store ---------

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = QdrantClient(url="http://localhost:6333")

    vector_store=QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="Notes",
        embedding=embedding_model
    )

# ---------- Similarity Search -----------

    if Chat:
        relevant_result=vector_store.similarity_search(query=Chat)
        Notes="\n\n\n".join([f"Page Content: {result.page_content}\n Page number: {result.metadata['page_label']}\n File location: {result.metadata['source']}"for result in relevant_result])


# ---------- Setting up the LLM ----------

        Prompt=f'''

    You are a helpful and knowledgeable AI tutor for college students. 
    Your goal is to answer questions using ONLY the student’s course notes provided and your memory of previous chats. 
    You should also enhance the notes, give clear explanations, and provide references to the notes used. 
    Explain the concepts clearly but do not go beyond the information in the notes.

    Guidelines:

    1. The student's notes are provided below in variable `Notes`. Use them as the primary source of truth. 
    2. Use the previous conversation history in `Memory` to maintain context and continuity.
    3. Only provide information that can be inferred from `Notes` and previous chats. 
    4. If the answer is not in `Notes`, politely say you don’t know, instead of guessing. 
    5. Keep explanations clear, concise, and student-friendly, suitable for studying or revision.
    6. When relevant, give examples, step-by-step breakdowns, or simplified explanations to aid understanding.
    7. If the student's question is ambiguous, ask a **clarifying follow-up question** before attempting a full answer.
    8. Highlight key terms or important concepts from the notes in your response.
    9. Encourage the student to provide feedback by asking if your answer was helpful.
    10. Also ask two related questions at the end of the output that if the user also want to know about those two related questions

    Notes:
    {Notes}

    Previous conversation history:
    {Memory}

    Answer the student’s question now.
    
        '''

# ----- AI LLM Model,API key Configuration -----

        client=genai.Client(api_key="...")

        response=client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(f"System prompt: {Prompt},user query: {Chat}")
            )
        st.chat_message("user").write(Chat)
        st.chat_message("assistant").write(response.text)

# ---------- Uploading New Notes to the Vector Base ------------

def Uploading():

# ----- Saving the user Provided File in Notes Folder -----

    save_folder = Path(__file__).parent/"Notes"
    os.makedirs(save_folder, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        save_path = os.path.join(save_folder, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' saved successfully in folder '{save_folder}'!")
        st.subheader("Embedding Notes into Vector Database, Wait!")

# ----- Running the Database_Embeddings.py script automatically to save data to Vector Base -----

        try:
            script_path = Path (__file__).parent/"Database_Embeddings.py"
            subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            st.success("Data Stored in Vector Store")
        except:
             st.error("Data not embedded")

# ---------- User intereaction Area ----------

with st.sidebar:

    st.title("Modes")
    mode=st.selectbox("Choose Mode: ",["Ask AI","Upload Notes"])

if mode=="Ask AI":
        AI_Mode()
if mode=="Upload Notes":
        Uploading()
