"""Agent loop and context management for Day 7 Agentic RAG."""
from typing import Any, Protocol


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        ...


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Gói ngữ cảnh từ các candidate chunk, loại bỏ trùng lặp và tuân thủ context budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c.get('source', '')} | page {c.get('page', '')}]\n{c.get('text', '')}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 2,
              top_k: int = 3, budget: int = 2000) -> dict[str, Any]:
    """Chạy vòng lặp Agentic RAG giữa Controller và Retriever.
    Lưu trace đầy đủ gồm events, queries, retrieved results, context và citations.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("question, max_searches, top_k hoặc budget không hợp lệ")

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get("action") == "search":
            if sum(e["event"] == "search" for e in events) >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời sau khi đã vượt giới hạn tìm kiếm.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}]
                }
            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                query = question

            results = retriever.search(query, filters=filters, top_k=top_k)
            # Ensure unique IDs
            unique_results = []
            seen_r = set()
            for r in results:
                if r["id"] not in seen_r:
                    seen_r.add(r["id"])
                    unique_results.append(r)
            results = unique_results[:top_k]

            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}
            events.append({
                "event": "search",
                "query": query,
                "filter": filters,
                "results": results,
                "context": context,
                "context_ids": ids
            })
            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e["event"] == "search"],
                "context": context,
                "context_ids": ids
            }]
        elif action.get("action") == "answer":
            answer = action.get("answer", "")
            citations = action.get("citations", [])
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "invalid_or_missing_citations"}]
                }
            
            # Check validity of citations
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))
            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "events": events + [{"event": "answer"}]
                }
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations"}]
            }
        else:
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_action"}]
            }

    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events
    }
