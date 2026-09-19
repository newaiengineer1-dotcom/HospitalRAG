HospitalRAG Sample Hospital Documents

This folder contains synthetic demonstration documents in all formats requested:
PDF, DOCX, TXT, MD, CSV, XLSX, and PPTX.

These files are designed to test:
- document extraction
- page/slide metadata
- chunking
- table extraction
- FAISS indexing
- source citations
- format detection

IMPORTANT:
The content is fictional/synthetic. It is not an actual hospital's information,
policy, emergency protocol, clinical guideline, or patient record.

Place these files inside:
hospital_documents/

Then run:
python ingest.py

Expected knowledge-base output:
knowledge_base/
    faiss.index
    metadata.json
    config.json
    manifest.json
