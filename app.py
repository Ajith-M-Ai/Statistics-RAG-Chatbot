import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Statistics RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Statistics RAG Chatbot")
st.write("Ask questions from your Statistics PDF")

# -----------------------------
# Load Vector Database
# -----------------------------

@st.cache_resource
def load_vectorstore():

    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "Statistics Essentials.pdf"
    )

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# -----------------------------
# Load LLM
# -----------------------------

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# -----------------------------
# Initialize Components
# -----------------------------

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 25
    }
)

tokenizer, model = load_model()


# -----------------------------
# Question Answer Function
# -----------------------------

def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an expert Statistics tutor.

Answer ONLY from the provided context.

If the answer is not available in the context, reply:

"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, docs


# -----------------------------
# User Interface
# -----------------------------

question = st.text_input(
    "Ask a Question"
)

if st.button("Get Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            answer, docs = ask_question(question)

        st.subheader("Answer")

        st.success(answer)

        with st.expander("Retrieved Context"):

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

                st.divider()