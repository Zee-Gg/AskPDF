**AskPDF — Intelligent PDF Question Answering with Mistral & Flask**
AskPDF is a Flask-based web application that allows users to ask natural language questions based on the contents of PDF documents. It uses MistralAI’s LLM via LangChain, sentence-transformer embeddings, and FAISS for fast semantic search.

**Features**
 Natural language QA from PDF documents

 Powered by Mistral AI, LangChain, HuggingFace embeddings

 Efficient similarity search using FAISS vector database

 Simple and interactive web interface using Flask

 Automatic metadata tagging with PDF filenames

 **Project Structure**

 project/
│
├── app.py                  # Main Flask app
├── .env                    # Environment variables (Mistral API key)
├── /templates
│   └── index.html          # Frontend HTML page
├── /static
│   └── style.css           # Styling for frontend
├── /pdfs
│   └── *.pdf               # Your PDF files to load
├── /vector_index           # FAISS vector storage (auto-generated)


**Requirements**
Make sure Python 3.9+ is installed.

**Install dependencies using:**
pip install -r requirements.txt

**Sample requirements.txt:**
Flask
python-dotenv
langchain
langchain-community
langchain-mistralai
sentence-transformers
faiss-cpu

**Environment Variables**
Create a .env file in your root directory with your Mistral API key:
MISTRAL_API_KEY=your_actual_api_key_here

Usage
Place PDF files inside the /pdfs/ folder.

Run the Flask app:
python multiple-pdf.py

**Access the app in your browser at:**
http://localhost:5000/

Type a question related to the contents of any uploaded PDF and get an instant answer with source references.

**How It Works**
Loads PDFs and extracts text using PyPDFLoader.

Splits text into chunks and stores their embeddings using FAISS.

When a question is asked:

Relevant chunks are fetched using vector similarity.

MistralAI answers the question using LangChain’s QA chain.

Source PDF filenames are returned alongside the answer.

**Notes**
Ensure index.html is inside the templates/ folder.

Ensure style.css is inside the static/ folder.

On first run, FAISS index is created and cached.
