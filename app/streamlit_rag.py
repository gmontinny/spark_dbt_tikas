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
st.caption("Powered by Apache Spark 4.x · SentenceTransformers · OPT-125M · StarRocks")

# DATAS_DIR: no container dev os arquivos estão em /workspace/datas
DATAS_DIR = os.getenv("DATAS_DIR", "/workspace/datas")
if not os.path.exists(DATAS_DIR):
    DATAS_DIR = os.path.join(os.path.dirname(__file__), "..", "datas")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuração")
    top_k = st.slider("Chunks recuperados (Top-K)", 1, 10, 3)
    show_chunks = st.toggle("Mostrar chunks recuperados", value=True)
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

# ── Cache de embeddings e modelos ────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando embeddings do Silver...")
def load_docs():
    from rag_query import _load_embeddings_local
    return _load_embeddings_local()


@st.cache_resource(show_spinner="Carregando modelo LLM...")
def load_llm():
    """Carrega tokenizer e modelo uma única vez — reutilizado em todas as perguntas."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag_query import LLM_MODEL

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=dtype, low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def generate_cached(question: str, chunks: list[dict]) -> str:
    """Geração com modelo já carregado em cache."""
    import torch
    from rag_query import TOP_K

    tokenizer, model = load_llm()

    context = "\n\n---\n\n".join(
        f"[{c['file_name']}]\n{c['text_clean'][:600]}" for c in chunks
    )
    prompt = (
        "Você é um assistente especializado em análise de documentos.\n"
        "Use APENAS o contexto abaixo para responder. "
        "Se não encontrar a informação, diga claramente.\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {question}\n\n"
        "RESPOSTA:"
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip() or "Não foi possível gerar uma resposta com base nos documentos."


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
                from rag_query import retrieve
                os.environ["RAG_TOP_K"] = str(top_k)
                docs_data = load_docs()
                chunks = retrieve(question, docs_data)

            if show_chunks and chunks:
                with st.expander(f"📚 {len(chunks)} chunk(s) recuperado(s)", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**[{i}] `{c['file_name']}` — score: `{c['score']:.4f}`**")
                        st.text(c["text_clean"][:400] + "...")
                        st.divider()

            with st.spinner("Gerando resposta..."):
                answer = generate_cached(question, chunks)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            msg = f"❌ Erro ao processar a pergunta: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
