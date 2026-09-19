import json, os, re
from pathlib import Path
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

BASE=Path(__file__).resolve().parent
KB=BASE/"knowledge_base"
INDEX=KB/"faiss.index"; META=KB/"metadata.json"; CONFIG=KB/"config.json"
DEFAULT_EMBED="sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL="openai/gpt-oss-120b"

st.set_page_config(page_title="HospitalRAG", page_icon="🏥", layout="wide")
st.markdown("""
<style>
.stApp{background:#071313;color:#E6FFFB}
[data-testid="stSidebar"]{background:#0A1D1D}
.hero{padding:28px;border:1px solid #175C58;border-radius:18px;background:#0B2423;margin-bottom:22px}
.hero h1{margin:0;color:#E6FFFB}.hero p{color:#A7C9C5}
.source{padding:12px;border-left:3px solid #14B8A6;background:#0A1918;border-radius:8px;margin:7px 0}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def embedder(name): return SentenceTransformer(name)

@st.cache_resource
def load_kb():
    if not INDEX.exists() or not META.exists(): return None,[],{}
    idx=faiss.read_index(str(INDEX))
    meta=json.loads(META.read_text(encoding="utf-8"))
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    return idx,meta,cfg

def secret(name, default=None):
    try:
        v=st.secrets.get(name)
        if v: return v
    except Exception: pass
    return os.getenv(name,default)

def words(s):
    stop={"what","when","where","which","with","from","about","please","could","would","this","that","have","does","your","hospital"}
    return {x for x in re.findall(r"[A-Za-z0-9]{4,}",s.lower()) if x not in stop}

def retrieve(q,idx,meta,model,topk,threshold):
    m=embedder(model)
    v=m.encode([q],normalize_embeddings=True).astype("float32")
    scores,ids=idx.search(v,min(topk*3,len(meta)))
    qw=words(q); out=[]; seen=set()
    for score,i in zip(scores[0],ids[0]):
        if i<0 or i>=len(meta) or float(score)<threshold: continue
        item=meta[i]; text=str(item.get("text",""))
        kw=len(qw & words(text))/max(len(qw),1)
        final=.8*float(score)+.2*kw
        key=(item.get("filename"),item.get("page_number"),text[:150])
        if key in seen: continue
        seen.add(key); out.append({"score":final,"item":item})
        if len(out)>=topk: break
    return sorted(out,key=lambda x:x["score"],reverse=True)

def answer(q,results,model):
    key=secret("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY is not configured in Streamlit Secrets.")
    context="\n\n".join(
        f"[SOURCE {i}] Document: {r['item'].get('filename')} | Page/Slide: {r['item'].get('page_number','N/A')} | Section: {r['item'].get('section','N/A')}\n{r['item'].get('text','')}"
        for i,r in enumerate(results,1))
    system="""You are HospitalRAG, a document-grounded hospital knowledge assistant.
Answer ONLY from the supplied context. Never invent facts, policies, doctors, phone numbers,
hours, prices, medications, diagnoses, procedures, dates, values, or citations.
If evidence is insufficient, say exactly: "I could not find sufficient information in the knowledge base to answer this question."
Cite evidence as [Source 1], [Source 2]. Do not invent page numbers.
For emergencies, direct the user to qualified emergency/medical professionals."""
    c=Groq(api_key=key)
    r=c.chat.completions.create(model=model,messages=[
        {"role":"system","content":system},
        {"role":"user","content":f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"}],
        temperature=.1,max_tokens=900)
    return r.choices[0].message.content

idx,meta,cfg=load_kb()
embed_model=cfg.get("embedding_model",DEFAULT_EMBED)
groq_model=secret("GROQ_MODEL",cfg.get("groq_model",DEFAULT_MODEL))
topk=int(cfg.get("top_k",5)); threshold=float(cfg.get("similarity_threshold",.42))

with st.sidebar:
    st.markdown("## 🏥 HospitalRAG")
    if idx:
        st.success("Knowledge Base Ready")
        st.metric("Documents",len({x.get("document_id",x.get("filename")) for x in meta}))
        st.metric("Chunks",len(meta))
    else: st.error("Knowledge Base Missing")
    if st.button("Clear chat",use_container_width=True):
        st.session_state.messages=[]; st.rerun()
    st.caption("Grounded document QA. Not a substitute for professional medical care.")

st.markdown('<div class="hero"><h1>🏥 Hospital Knowledge Assistant</h1><p>Ask questions about approved hospital documents and receive grounded answers with source references.</p></div>',unsafe_allow_html=True)

if not idx:
    st.warning("Knowledge base is missing. Run `python ingest.py`, then upload the four generated files in knowledge_base/ to GitHub.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

q=st.chat_input("Ask a question about the approved hospital documents...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        results=retrieve(q,idx,meta,embed_model,topk,threshold)
        if not results:
            text="I could not find sufficient information in the knowledge base to answer this question."
            st.markdown(text); sources=[]
        else:
            try: text=answer(q,results,groq_model)
            except Exception as e: text=f"Unable to generate the answer. Check your Groq configuration. Details: {e}"
            st.markdown(text); sources=results
            st.markdown("#### Sources")
            for r in results:
                x=r["item"]
                st.markdown(f'<div class="source"><b>{x.get("filename","Unknown")}</b><br>Page/Slide: {x.get("page_number","N/A")} • Section: {x.get("section","N/A")} • Relevance: {r["score"]:.3f}</div>',unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":text})
