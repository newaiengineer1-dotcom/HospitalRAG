# HospitalRAG — Hospital Knowledge Assistant

## Quick start
1. Put approved documents in `hospital_documents/`.
2. Run `python ingest.py`.
3. Confirm `knowledge_base/faiss.index`, `metadata.json`, `config.json`, `manifest.json`.
4. Upload the project to GitHub.
5. Deploy `app.py` on Streamlit Community Cloud.
6. In Streamlit App Settings → Secrets add:
```toml
GROQ_API_KEY = "gsk_your_actual_key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Local
```bash
python -m venv .venv
python -m pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

The included demo information should be treated as synthetic test data, not clinical guidance. Never upload patient-identifiable or confidential hospital data to a public repository.
