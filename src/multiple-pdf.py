from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai.chat_models import ChatMistralAI

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ MISTRAL_API_KEY not found in .env file")

# 📁 Constants
PDF_FOLDER_PATH = "../pdfs"
VECTOR_STORE_PATH = "vector_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Initialize Flask app
app = Flask(__name__, static_folder="../static", template_folder="../templates")

# 📄 Load PDFs and attach metadata (filename)
all_docs = []
for filename in os.listdir(PDF_FOLDER_PATH):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = filename  # Add source filename
        all_docs.extend(documents)

# ✂️ Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(all_docs)

# 🔍 Vector index
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
if os.path.exists(VECTOR_STORE_PATH):
    print("✅ Loading existing FAISS index...")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("🔄 Creating new FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

# 🤖 LLM + QA Chain
llm = ChatMistralAI(api_key=api_key, model="mistral-small", temperature=0.5, max_tokens=512)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# 🔗 Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "❌ Question is required."}), 400

    relevant_docs = vectorstore.similarity_search(question, k=3)
    answer = qa_chain.run(input_documents=relevant_docs, question=question)

    sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))

    return jsonify({
        "answer": answer,
        "sources": sources
    })

# ▶️ Run the app
if __name__ == "__main__":
    app.run(debug=True)
