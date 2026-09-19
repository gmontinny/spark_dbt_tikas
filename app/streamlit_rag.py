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
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jobs"))

import streamlit as st

st.config.set_option("server.fileWatcherType", "none")

st.set_page_config(page_title="RAG — Documentos", page_icon="🔍", layout="wide")

st.title("🔍 RAG — Perguntas sobre os Documentos")

from rag_query import LLM_PROVIDER, LLM_MODEL, OPENAI_MODEL, GEMINI_MODEL, OLLAMA_URL

_provider_label = {
    "openai": f"OpenAI · {OPENAI_MODEL}",
    "gemini": f"Google Gemini · {GEMINI_MODEL}",
    "ollama": f"Ollama · {LLM_MODEL} ({OLLAMA_URL})",
}.get(LLM_PROVIDER, LLM_PROVIDER)

st.caption(f"Powered by Apache Spark 4.x · SentenceTransformers · {_provider_label}")

DATAS_DIR = os.getenv("DATAS_DIR", "/workspace/datas")
if not os.path.exists(DATAS_DIR):
    DATAS_DIR = os.path.join(os.path.dirname(__file__), "..", "datas")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuração")
    top_k = st.slider("Chunks recuperados (Top-K)", 1, 10, 3)
    show_chunks = st.toggle("Mostrar chunks recuperados", value=True)
    st.divider()
    st.markdown("**Provider LLM**")
    st.info(_provider_label)
    st.markdown("*Altere `LLM_PROVIDER` no `.env` para trocar entre `openai`, `gemini` ou `ollama`.*")
    st.divider()
    st.markdown("**Documentos disponíveis**")
    docs = (
        [f for f in os.listdir(DATAS_DIR) if f.endswith((".pdf", ".docx"))]
        if os.path.exists(DATAS_DIR)
        else []
    )
    if docs:
        for doc in docs:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.warning(f"Nenhum documento encontrado em `{DATAS_DIR}`")

# ── Cache de embeddings ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando embeddings do Silver...")
def load_docs(embedding_model: str):
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
        try:
            with st.spinner("Buscando contexto relevante..."):
                from rag_query import retrieve, generate
                os.environ["RAG_TOP_K"] = str(top_k)
                docs_data = load_docs(os.getenv("EMBEDDING_MODEL", ""))
                chunks = retrieve(question, docs_data)

            if show_chunks and chunks:
                with st.expander(f"📚 {len(chunks)} chunk(s) recuperado(s)", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**[{i}] `{c['file_name']}` — score: `{c['score']:.4f}`**")
                        st.text(c["text_clean"][:400] + "...")
                        st.divider()

            with st.spinner(f"Gerando resposta via {_provider_label}..."):
                answer = generate(question, chunks)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            msg = f"❌ Erro ao processar a pergunta: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
