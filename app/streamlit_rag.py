"""
Interface RAG — Streamlit
==========================
Interface web para perguntas e respostas sobre os documentos
ingeridos pelo pipeline (Bronze → Silver → Gold).

Execução:
    streamlit run app/streamlit_rag.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jobs"))

import streamlit as st

st.set_page_config(page_title="RAG — Documentos", page_icon="🔍", layout="wide")

st.title("🔍 RAG — Perguntas sobre os Documentos")
st.caption("Powered by Apache Spark 4.x · SentenceTransformers · Mistral-7B · StarRocks")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuração")
    top_k = st.slider("Chunks recuperados (Top-K)", 1, 10, 3)
    show_chunks = st.toggle("Mostrar chunks recuperados", value=True)
    st.divider()
    st.markdown("**Documentos disponíveis**")
    datas_dir = os.getenv("DATAS_DIR", "./datas")
    docs = [f for f in os.listdir(datas_dir) if f.endswith((".pdf", ".docx"))] if os.path.exists(datas_dir) else []
    for doc in docs:
        st.markdown(f"- 📄 `{doc}`")
    if not docs:
        st.warning("Nenhum documento encontrado em `datas/`")

# ── Cache dos embeddings ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando embeddings do Silver...")
def load_docs():
    from rag_query import _load_embeddings_local
    return _load_embeddings_local()


# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Faça uma pergunta sobre os documentos...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto e gerando resposta..."):
            from rag_query import generate, retrieve

            os.environ["RAG_TOP_K"] = str(top_k)
            docs_data = load_docs()
            chunks = retrieve(question, docs_data)

            if show_chunks:
                with st.expander(f"📚 {len(chunks)} chunk(s) recuperado(s)", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**[{i}] `{c['file_name']}` — score: `{c['score']:.4f}`**")
                        st.text(c["text_clean"][:400] + "...")
                        st.divider()

            answer = generate(question, chunks)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
