# AskPDF — Intelligent PDF Question Answering with Mistral & Streamlit

AskPDF is a **Streamlit-based web application** that allows users to upload multiple PDF files and ask natural language questions about their contents.
It leverages **Mistral AI’s LLM via LangChain**, **sentence-transformer embeddings**, and **FAISS** for efficient semantic search.

---

## **Features**

* Upload multiple PDF documents
* Ask natural language questions from your PDFs
* Powered by Mistral AI and LangChain
* Fast semantic search using HuggingFace embeddings and FAISS
* Automatic metadata tagging with PDF filenames
* Simple, interactive, and responsive Streamlit UI

---

## **Project Structure**

```
project/
│── multiple-pdf.py        # Main Streamlit app
│── .env                   # Environment variables (Mistral API key)
│── requirements.txt       # Dependencies
│── vector_index/          # FAISS vector storage (auto-generated)
```

---

## **Requirements**

* Python 3.9+
* Install dependencies:

```bash
pip install -r requirements.txt
```

**Sample requirements.txt:**

```
streamlit
python-dotenv
langchain
langchain-community
langchain-mistralai
sentence-transformers
faiss-cpu
```

---

## **Environment Variables**

Create a **`.env`** file in the root directory and add your Mistral API key:

```
MISTRAL_API_KEY=your_actual_api_key_here
```

---

## **Usage**

Run the Streamlit app:

```bash
streamlit run multiple-pdf.py
```

Open in your browser at:
[http://localhost:8501](http://localhost:8501)

1. Upload one or more PDF files.
2. Type a question in the input box.
3. Get instant answers with source references.

---

## **How It Works**

1. **Upload PDFs** → Extract text using `PyPDFLoader`.
2. **Split text** → Chunk documents with LangChain’s `CharacterTextSplitter`.
3. **Embed & Store** → Generate embeddings with HuggingFace and save in FAISS.
4. **Ask a Question** → Retrieve relevant chunks via similarity search.
5. **Answer** → Mistral AI answers using LangChain’s QA Chain.
6. **Display** → The app shows both the answer and the source PDFs.

---

