# 📚 Statistics RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, Hugging Face Embeddings, FLAN-T5, and Streamlit.

## Features

- Load PDF documents
- Split documents into chunks
- Generate embeddings using MiniLM
- Store embeddings in FAISS
- Retrieve relevant chunks
- Generate answers using FLAN-T5
- Interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face
- FAISS
- Transformers
- PyPDF

## Project Structure

```
RAG_PDF_Chatbot/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── RAG_PDF_Chatbot.ipynb
│── datamites_notes.pdf/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Model

- sentence-transformers/all-MiniLM-L6-v2
- google/flan-t5-base

## Author

AJU