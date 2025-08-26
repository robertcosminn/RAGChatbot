from __future__ import annotations

import os, sys
from pathlib import Path
import streamlit as st

# Ensure project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.llm.chain import run_chain

st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="wide")

# ---------- Styles ----------
NEO_CSS = """
<style>
:root{
  --bg: #0f1115;
  --panel: #141821;
  --muted: #8b95a7;
  --text: #e5e7eb;
  --primary: #7c5cff;
  --border: #242a37;
  --shadow: 0 8px 24px rgba(0,0,0,0.25);
  --radius: 14px;
}
@media (prefers-color-scheme: light){
  :root{
    --bg: #ffffff;
    --panel: #f8fafc;
    --muted: #475467;
    --text: #0f172a;
    --primary: #5b5bd6;
    --border: #e5e7eb;
    --shadow: 0 10px 20px rgba(2,6,23,0.06);
  }
}
html, body, [class*="css"]{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{padding-top: 2rem;}
.neo-nav{
  backdrop-filter: blur(10px);
  background: linear-gradient(180deg, rgba(20,24,33,0.55), rgba(20,24,33,0.35));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 14px;
  box-shadow: var(--shadow);
}
.neo-title{
  font-weight: 700;
  font-size: 20px;
  color: var(--text);
  display: flex; gap: 8px; align-items: center;
}
.neo-chip{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 12px; border:1px solid var(--border);
  border-radius: 999px; background: var(--panel); color: var(--text);
  font-size: 13px; cursor: pointer; user-select: none; transition: all .15s ease;
}
.neo-chip:hover{border-color: var(--primary); box-shadow: 0 0 0 3px rgba(124,92,255,0.12);}
.neo-hero{
  margin-top: 16px; border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(124,92,255,0.08), transparent);
  border-radius: var(--radius); padding: 20px 18px;
}
.neo-hero h1{margin:0; font-size:28px; color: var(--text);}
.neo-hero p{margin:6px 0 0 0; color: var(--muted);}
.neo-card{
  border:1px solid var(--border); border-radius: var(--radius);
  background: var(--panel); box-shadow: var(--shadow); padding:16px;
}
.neo-accent{
  border:1px solid rgba(124,92,255,0.35);
  background: linear-gradient(180deg, rgba(124,92,255,0.08), rgba(124,92,255,0.03));
}
.neo-badge{display:inline-block; padding:2px 8px; font-size:12px; border:1px solid var(--border); border-radius: 8px; color: var(--muted);}
.neo-pill{display:inline-block; padding:6px 10px; font-size:12px; border:1px solid var(--border); border-radius:999px; margin:4px 6px 0 0; color:var(--text); background: rgba(124,92,255,0.08);}
.neo-small{color: var(--muted); font-size: 13px;}
</style>
"""
st.markdown(NEO_CSS, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### Settings")
    persist_dir = st.text_input("Persist dir", "data/chroma")
    collection_name = st.text_input("Collection", "books_v1")
    top_k = st.slider("Top-K", 1, 10, 5, 1)
    model_override = st.text_input("Chat model (override)", "")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

    manifest_path = Path(persist_dir) / f"{collection_name}_manifest.json"
    if manifest_path.exists():
        st.success("Index ready ✅")
        with st.expander("Manifest"):
            st.code(manifest_path.read_text(encoding="utf-8"), language="json")

# ---------- Top Nav ----------
nav_l, nav_r = st.columns([0.75, 0.25])
with nav_l:
    st.markdown('<div class="neo-nav neo-title">📚 Smart Librarian <span class="neo-badge">RAG + Tool</span></div>', unsafe_allow_html=True)
with nav_r:
    st.markdown("<div style='height:42px;'></div>", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="neo-hero">
  <h1>Find the one book that fits.</h1>
  <p>Describe the vibe, themes, or plot. I’ll retrieve, choose exactly one title, and include its full summary.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Prompt chips ----------
cols = st.columns(3)
examples = [
    "friendship and magic at a boarding school",
    "post-apocalyptic father and son survival, bleak but hopeful",
    "classic romance with sharp social commentary and character growth",
]
labels = ["✨ Friendship & Magic", "🌫️ Post-apocalyptic journey", "💌 Classic romance & wit"]
chip_clicked = None
for i in range(3):
    with cols[i]:
        if st.button(labels[i], use_container_width=True):
            chip_clicked = examples[i]

st.divider()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Tell me what kind of book you are looking for."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Input ----------
user_prompt = st.chat_input("Ask for a book by theme, vibe, or plot elements...")
if chip_clicked and not user_prompt:
    user_prompt = chip_clicked

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                out = run_chain(
                    query=user_prompt,
                    top_k=top_k,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    model=(model_override or None),
                )
                content = (out.get("content") or "").strip()
                chosen_title = out.get("chosen_title")
                full_summary = out.get("full_summary")
                retrieval = out.get("retrieval") or []

                st.markdown("###### Recommendation")
                st.markdown(f"""
                    <div class="neo-card neo-accent">
                      <div class="neo-title" style="font-size:18px;">✅ {chosen_title or 'No title chosen'}</div>
                      <div class="neo-small" style="margin-top:6px;">Model chose one title based on retrieved context.</div>
                      <div style="margin-top:12px;">{content if content else 'No content available.'}</div>
                    </div>
                """, unsafe_allow_html=True)

                if full_summary:
                    st.markdown("###### Full summary")
                    st.markdown(f"""<div class="neo-card">{full_summary}</div>""", unsafe_allow_html=True)
                    st.download_button(
                        "Download summary",
                        data=full_summary.encode("utf-8"),
                        file_name=f"{(chosen_title or 'summary').replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                if retrieval:
                    st.markdown("###### Retrieval context")
                    cols2 = st.columns(2)
                    for i, r in enumerate(retrieval):
                        with cols2[i % 2]:
                            title = r.get("title") or ""
                            dist = r.get("distance")
                            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else ""
                            themes = (r.get("themes") or "").split(",")
                            pills = "".join([f"<span class='neo-pill'>{t.strip()}</span>" for t in themes if t.strip()])
                            st.markdown(f"""
                                <div class="neo-card" style="margin-bottom:10px;">
                                  <div class="neo-title" style="font-size:16px;">{title}</div>
                                  <div class="neo-small">Distance: {dist_str}</div>
                                  <div style="margin-top:8px;">{pills}</div>
                                </div>
                            """, unsafe_allow_html=True)

                history_block = content
                if chosen_title and (not content or chosen_title not in content):
                    history_block = f"**Recommendation:** {chosen_title}\n\n" + (content or "")
                st.session_state.messages.append({"role": "assistant", "content": history_block})

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Tips: run ingest, set OPENAI_API_KEY in .env, and start Streamlit from project root.")
