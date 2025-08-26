"""
Orchestrator: retrieve -> LLM choose title -> tool call -> final answer.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.llm.openai_client import OpenAIClient
from app.llm.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.llm.tools import build_tools_spec, handle_tool_call
from app.rag.retriever import retrieve


def _extract_short_summary(doc_text: str) -> str:
    # doc_text has lines like "Title: ...\nSummary: ...\nThemes: ..."
    if not doc_text:
        return ""
    parts = doc_text.split("Summary:", 1)
    if len(parts) < 2:
        return doc_text.strip()
    summary = parts[1].strip()
    # Cut at Themes: or limit length
    summary = summary.split("Themes:", 1)[0].strip()
    if len(summary) > 800:
        summary = summary[:800].rstrip() + "…"
    return summary


def _format_context(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in results:
        title = r.get("title") or "(unknown)"
        themes = r.get("themes") or "N/A"
        short = _extract_short_summary(r.get("document") or "")
        lines.append(f"- Title: {title}\n  Themes: {themes}\n  Short Summary: {short}")
    return "\n".join(lines)


def run_chain(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/chroma",
    collection_name: str = "books_v1",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for the RAG chain.

    Returns a dict:
        {
            "content": str,             # final assistant message
            "chosen_title": str | None, # title passed to the tool (if any)
            "full_summary": str | None, # tool result summary
            "tool_match_score": float | None,
            "retrieval": [ {title, themes, distance}, ... ]
        }
    """
    # 1) Retrieve context
    retrieved = retrieve(
        query=query,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    context = _format_context(retrieved)

    # 2) Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query, context=context)},
    ]

    client = OpenAIClient()
    tools = build_tools_spec()

    # 3) Ask model (expect a tool call)
    first = client.chat(messages=messages, tools=tools, model=model)
    assistant_msg = first["message"]
    tool_calls = assistant_msg.get("tool_calls") or []

    chosen_title = None
    full_summary = None
    tool_match_score = None

    if tool_calls:
        # Execute tool(s), then send tool results back for final answer
        messages.append(assistant_msg)  # keep the original assistant message
        for tc in tool_calls:
            if tc.get("type") != "function":
                continue
            fn = tc.get("function", {}) or {}
            name = fn.get("name")
            args = fn.get("arguments") or "{}"
            if name == "get_summary_by_title":
                tool_res = handle_tool_call(name, args)
                chosen_title = tool_res.get("title")
                full_summary = tool_res.get("summary")
                tool_match_score = tool_res.get("match_score")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "content": json.dumps(tool_res),
                })
        # 4) Finalization turn
        second = client.chat(messages=messages, tools=tools, model=model)
        final_content = (second["message"].get("content") or "").strip()
    else:
        # No tool call; graceful fallback
        final_content = (assistant_msg.get("content") or "").strip()
        if retrieved:
            chosen_title = retrieved[0].get("title")

    retrieval_view = [
        {"title": r.get("title"), "themes": r.get("themes"), "distance": r.get("distance")}
        for r in retrieved
    ]

    return {
        "content": final_content,
        "chosen_title": chosen_title,
        "full_summary": full_summary,
        "tool_match_score": tool_match_score,
        "retrieval": retrieval_view,
    }


if __name__ == "__main__":
    out = run_chain("I want a story about friendship and magic at a boarding school", top_k=5)
    print("Chosen title:", out["chosen_title"])
    print("Tool score:", out["tool_match_score"])
    print("---- Final content ----")
    print(out["content"][:800])
