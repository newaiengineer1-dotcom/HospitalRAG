import csv,json,re,hashlib
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

BASE=Path(__file__).resolve().parent; DOCS=BASE/"hospital_documents"; KB=BASE/"knowledge_base"
MODEL="sentence-transformers/all-MiniLM-L6-v2"; SIZE=1800; OVERLAP=300

def clean(x): return re.sub(r"\s+"," ",str(x or "")).strip()
def docid(p): return hashlib.sha256(p.name.encode()).hexdigest()[:16]
def chunks(t):
    t=clean(t); out=[]; start=0
    while start<len(t):
        end=min(start+SIZE,len(t)); c=t[start:end].strip()
        if c: out.append(c)
        if end>=len(t): break
        start=max(end-OVERLAP,start+1)
    return out
def add(rec,p,text,page=None,section="General"):
    for n,c in enumerate(chunks(text)):
        rec.append({"document_id":docid(p),"filename":p.name,"page_number":page,"section":section,
                    "chunk_id":f"{docid(p)}-{len(rec):06d}","chunk_index":n,"text":c,"source":p.name})
def main():
    rec=[]
    for p in DOCS.rglob("*"):
        if not p.is_file(): continue
        try:
            if p.suffix.lower()==".pdf":
                for n,page in enumerate(PdfReader(str(p)).pages,1): add(rec,p,page.extract_text() or "",n,f"Page {n}")
            elif p.suffix.lower()==".docx":
                d=Document(str(p)); add(rec,p,"\n".join(x.text for x in d.paragraphs),"","Document text")
                for n,t in enumerate(d.tables,1): add(rec,p,"\n".join(" | ".join(c.text for c in r.cells) for r in t.rows),None,f"Table {n}")
            elif p.suffix.lower() in {".txt",".md"}: add(rec,p,p.read_text(errors="ignore"))
            elif p.suffix.lower()==".csv":
                with open(p,encoding="utf-8-sig",errors="ignore") as f: add(rec,p,"\n".join(" | ".join(r) for r in csv.reader(f)),None,"CSV table")
            elif p.suffix.lower()==".xlsx":
                wb=load_workbook(p,read_only=True,data_only=True)
                for ws in wb.worksheets: add(rec,p,"\n".join(" | ".join(str(v) for v in r if v is not None) for r in ws.iter_rows(values_only=True)),None,f"Worksheet: {ws.title}")
            elif p.suffix.lower()==".pptx":
                prs=Presentation(str(p))
                for n,s in enumerate(prs.slides,1): add(rec,p,"\n".join(sh.text for sh in s.shapes if hasattr(sh,"text")),n,f"Slide {n}")
        except Exception as e: print("WARNING",p.name,e)
    if not rec: raise SystemExit("No supported documents found.")
    model=SentenceTransformer(MODEL); emb=model.encode([x["text"] for x in rec],normalize_embeddings=True).astype("float32")
    KB.mkdir(exist_ok=True); idx=faiss.IndexFlatIP(emb.shape[1]); idx.add(emb); faiss.write_index(idx,str(KB/"faiss.index"))
    (KB/"metadata.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
    (KB/"config.json").write_text(json.dumps({"embedding_model":MODEL,"chunk_size":SIZE,"chunk_overlap":OVERLAP,"top_k":5,"similarity_threshold":.42,"groq_model":"openai/gpt-oss-120b"},indent=2),encoding="utf-8")
    (KB/"manifest.json").write_text(json.dumps({"document_count":len({x["document_id"] for x in rec}),"chunk_count":len(rec)},indent=2),encoding="utf-8")
    print("Knowledge base created:",len(rec),"chunks")
if __name__=="__main__": main()
