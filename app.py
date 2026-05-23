import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


@st.cache_resource
def build_vectorstore(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vectorstore


def build_chain(faissdb):
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    retriever = faissdb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False,
    )
    return chain


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Query App", page_icon="📄")
st.title("📄 PDF Query App")
st.write("Upload a PDF and ask questions about its content.")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Reading and indexing your PDF..."):
        faissdb = build_vectorstore(file_bytes)

    if st.session_state.chain is None:
        st.session_state.chain = build_chain(faissdb)
        st.session_state.chat_messages = []

    st.success("PDF indexed! Ask your question below.")

    # ── Chat history display ──────────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.write(content)

    # ── Input ─────────────────────────────────────────────────────────────────
    query = st.text_input("Ask a question about your PDF", key="query_input")

    col1, col2 = st.columns([1, 1])

    with col1:
        is_query_empty = not query
        submit = st.button("Submit", disabled=is_query_empty)

    with col2:
        clear = st.button("Clear Chat")
        if clear:
            st.session_state.chat_messages = []
            st.session_state.chain = build_chain(faissdb)
            st.rerun()

    if submit and query:
        user_message = {"role": "user", "content": query}
        st.session_state.chat_messages.append(user_message)

        with st.spinner("Generating answer..."):
            result = st.session_state.chain.invoke({"question": query})
            answer = result["answer"]

        assistant_message = {"role": "assistant", "content": answer}
        st.session_state.chat_messages.append(assistant_message)

        st.rerun()
