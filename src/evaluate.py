"""Evaluation pipeline for Day 7 Agentic RAG: Retrieval metrics and Agent audit."""
import time
from typing import Any
from .agent import run_agent, pack_context


def is_gold_hit(chunk: dict[str, Any], gold_list: list[dict[str, Any]]) -> bool:
    """Kiểm tra xem chunk có chứa bằng chứng chuẩn (gold evidence) hay không."""
    for g in gold_list:
        if (chunk.get("source_id") == g.get("source_id") and
            chunk.get("page") == g.get("page") and
            g.get("evidence", "") in chunk.get("text", "")):
            return True
    return False


def evaluate_retrieval(retriever, questions: list[dict[str, Any]], budget: int = 2000, top_k: int = 3) -> dict[str, Any]:
    """Đánh giá các chỉ số truy xuất: Hit@1, Hit@3, Context Evidence Rate."""
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    total = len(questions)
    times = []

    traces = []
    for q in questions:
        gold = q.get("gold", [])
        filters = q.get("filter")
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        h1 = is_gold_hit(results[0], gold) if len(results) > 0 else False
        h3 = any(is_gold_hit(r, gold) for r in results)

        context, context_ids = pack_context(results, budget=budget)
        retrieved_chunks = {r["id"]: r for r in results if r["id"] in context_ids}
        c_hit = any(is_gold_hit(retrieved_chunks[cid], gold) for cid in context_ids if cid in retrieved_chunks)

        if h1: hit1_count += 1
        if h3: hit3_count += 1
        if c_hit: context_hit_count += 1

        traces.append({
            "query_id": q["id"],
            "question": q["question"],
            "hit1": h1,
            "hit3": h3,
            "context_hit": c_hit,
            "results": [{"id": r["id"], "score": r["score"]} for r in results],
            "context_ids": context_ids
        })

    return {
        "n": total,
        "hit1": hit1_count,
        "hit1_rate": hit1_count / total if total > 0 else 0,
        "hit3": hit3_count,
        "hit3_rate": hit3_count / total if total > 0 else 0,
        "context_hit": context_hit_count,
        "context_hit_rate": context_hit_count / total if total > 0 else 0,
        "avg_ms": sum(times) / len(times) if times else 0,
        "traces": traces
    }


def evaluate_agent(cases: list[dict[str, Any]], retriever, controller,
                   budget: int = 2000, max_searches: int = 2, top_k: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chạy Agent trên toàn bộ các ca kiểm thử (gồm cả câu có đáp án và câu thiếu bằng chứng)."""
    traces = []
    reviews = []

    for c in cases:
        qid = c["id"]
        q_text = c["question"]
        is_answerable = c["answerable"]
        filters = c.get("filter")

        print(f"  Evaluating {qid} (answerable={is_answerable})...")
        t0 = time.perf_counter()
        result = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever,
            filters=filters,
            max_searches=max_searches,
            top_k=top_k,
            budget=budget
        )
        elapsed_s = time.perf_counter() - t0

        trace_entry = {
            "query_id": qid,
            "status": result["status"],
            "answer": result["answer"],
            "citations": result["citations"],
            "events": result["events"],
            "elapsed_seconds": round(elapsed_s, 2)
        }
        traces.append(trace_entry)

        # Audit review
        citations = result["citations"]
        status = result["status"]
        answer = result["answer"]

        if is_answerable:
            # Check citation support
            # Gold doc id from case
            gold_sources = {g["source_id"] for g in c.get("gold", [])}
            citation_support = []
            for cit in citations:
                # cit is like 'B_legal_630493'
                clean_src = cit.replace("B_", "").replace("A_", "").replace("C_", "")
                is_supported = clean_src in gold_sources
                citation_support.append(is_supported)

            if status == "abstain":
                answer_grade = 0
            else:
                # If answered with valid supported citations and clear answer
                if all(citation_support) and len(citation_support) > 0 and len(answer) > 20:
                    answer_grade = 2
                elif any(citation_support) and len(answer) > 10:
                    answer_grade = 1
                else:
                    answer_grade = 0

            review_entry = {
                "query_id": qid,
                "answer_grade": answer_grade,
                "citation_support": citation_support
            }
        else:
            # Unanswerable question
            abstention_correct = (status == "abstain")
            review_entry = {
                "query_id": qid,
                "abstention_correct": abstention_correct,
                "citation_support": [False] * len(citations)
            }

        reviews.append(review_entry)

    return traces, reviews
