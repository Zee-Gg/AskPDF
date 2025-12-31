import os
import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    from langchain_classic.chains.question_answering import load_qa_chain
from langchain_mistralai.chat_models import ChatMistralAI
import tempfile

# ✅ Load environment variables
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key and "MISTRAL_API_KEY" in st.secrets:
    api_key = st.secrets["MISTRAL_API_KEY"]

if not api_key:
    st.error("❌ **MISTRAL_API_KEY not found in `.env` file or Streamlit secrets. Please add it and restart the app.**")
    st.stop()

# 📁 Constants
VECTOR_STORE_PATH = "vector_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 🎨 Streamlit Page Config
st.set_page_config(page_title="Ask Your PDF", layout="wide")

# 🔹 Header
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📄 Ask Questions from Your PDF</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Upload your PDF files and ask anything about their content!</p>",
    unsafe_allow_html=True
)

# 📂 File Uploader
uploaded_files = st.file_uploader(
    "📂 **Upload PDF Files**",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF documents."
)

if uploaded_files:
    all_docs = []

    # Save uploaded PDFs temporarily & load
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(documents)

    # ✂️ Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_docs)
    
    # Filter out empty documents
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if not docs:
        st.warning("⚠️ No text found in the uploaded PDFs. Please upload PDFs with selectable text.")
        st.stop()

    # 🔍 Vector store
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 🤖 LLM + QA Chain
    llm = ChatMistralAI(
        api_key=api_key,
        model="mistral-small",
        temperature=0.5,
        max_tokens=512
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # 💬 Question Input
    st.markdown("### 💡 **Ask a Question**")
    question = st.text_input(
        "Enter your question below:",
        placeholder="Your PDFs are listening... ask away!",
        label_visibility="collapsed"
    )

    # 📜 Get Answer
    if question:
        with st.spinner("🔎 **Thinking...**"):
            relevant_docs = vectorstore.similarity_search(question, k=3)
            answer = qa_chain.run(input_documents=relevant_docs, question=question)
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))

        # Display Answer
        st.markdown("<h3 style='color: #27AE60;'>✅ Answer:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'>{answer}</p>", unsafe_allow_html=True)

        # Display Sources
        st.markdown("**📚 Sources:** " + ", ".join(sources))

else:
    st.info("👆 **Upload at least one PDF to get started.**", icon="ℹ️")
