import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="Statistics RAG Chatbot")

st.title("📚 Statistics RAG Chatbot")
st.write("Ask questions from your Statistics PDF")


@st.cache_resource
def load_vectorstore():

    loader = PyPDFLoader("datamites_notes.pdf/Statistics Essentials.pdf")

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


@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":5,
        "fetch_k":20
    }
)

tokenizer, model = load_model()


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Context:
{context}

Question:
{question}

Give a complete definition in one sentence.

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
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


question = st.text_input("Ask a Question")


if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            answer = ask_question(question)

        st.success(answer)