# AskPDF — PDF Question Answering

This repository contains a Streamlit app (`src/multiple-pdf.py`) that lets you upload PDFs, builds a FAISS vectorstore using sentence-transformers embeddings, and answers questions using Mistral via LangChain.

## Key files

- `src/multiple-pdf.py` — Streamlit app entrypoint.
- `requirements.txt` — Python dependencies for running and deploying the app.
- `Procfile` — Process declaration for Heroku-like hosts.

## Quick local run (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Add your Mistral API key to a `.env` file at the project root:

```
MISTRAL_API_KEY=your_key_here
```

4. Run the Streamlit app:

```powershell
python -m streamlit run src\\multiple-pdf.py --server.headless true
```

Then open http://localhost:8501 in your browser.

## Deploying

- Streamlit Cloud: push the repo to GitHub, create a Streamlit Cloud app and set `MISTRAL_API_KEY` as a secret.
- Heroku / similar: the included `Procfile` will start Streamlit; set the `MISTRAL_API_KEY` config var on the host.

## Notes & troubleshooting

- Some packages (e.g., `faiss-cpu`, `torch`) may need platform-specific wheels. If installation fails, consult their docs or use a host that provides compatible wheels (Streamlit Cloud often works well).
- If you want GPU support, change the packages accordingly and ensure the host provides GPUs.

## How it works (brief)

1. PDFs are loaded and split into chunks with `PyPDFLoader` + `CharacterTextSplitter`.
2. Embeddings (sentence-transformers) are computed and stored in a FAISS index.
3. At query time, the app finds similar chunks and uses Mistral (via LangChain) to answer questions and return source filenames.

---

If you want, I can also generate a pinned `requirements.txt` with exact working versions from your environment, or prepare a Dockerfile for containerized deployment.
