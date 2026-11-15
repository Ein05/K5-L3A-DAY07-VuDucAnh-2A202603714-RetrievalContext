"""Master script to run all retrieval comparisons, agent experiments, and benchmark evaluations."""
import json
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.retrieval import TfidfRetriever, DenseRetriever, HybridRetriever
from src.controller import GroqAgentController
from src.evaluate import evaluate_retrieval, evaluate_agent
from starter.agent_metrics import aggregate


def save_and_aggregate(cases, traces, reviews, out_dir, split):
    """Lưu trace, review, case và tổng hợp chỉ số kiểm toán theo agent_metrics."""
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("traces.json", traces), ("reviews.json", reviews)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    clean_cases = [{"id": c["id"], "split": split, "answerable": c["answerable"]} for c in cases]
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(clean_cases, f, ensure_ascii=False, indent=2)

    summary = aggregate(clean_cases, traces, reviews)
    gw = summary["grounded_answer"]
    if gw["N"] > 0:
        gw["mean"] = round((gw["1"] * 1 + gw["2"] * 2) / gw["N"], 2)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    print("=" * 70)
    print("DAY 7 — AGENTIC RAG EXPERIMENTAL BENCHMARK")
    print("Học viên: Vũ Đức Anh | MSSV: 2A202603714")
    print("=" * 70)

    data_dir = os.path.join(ROOT, "data")
    chunks = json.loads(open(os.path.join(data_dir, "chunks.json"), encoding="utf-8").read())
    questions = json.loads(open(os.path.join(data_dir, "questions.json"), encoding="utf-8").read())
    all_cases = json.loads(open(os.path.join(data_dir, "all_questions.json"), encoding="utf-8").read())

    dev_questions = [q for q in questions if q.get("split") == "dev"]
    test_questions = [q for q in questions if q.get("split") == "test"]
    dev_cases = [c for c in all_cases if c.get("split") == "dev"]
    test_cases = [c for c in all_cases if c.get("split") == "test"]

    print(f"\n[1] Chuẩn bị dữ liệu:")
    print(f"  - Chunks: {len(chunks)} điều luật")
    print(f"  - Dev: {len(dev_questions)} có đáp án | {len(dev_cases)} tổng ca")
    print(f"  - Test: {len(test_questions)} có đáp án | {len(test_cases)} tổng ca")

    # 1. Khởi tạo 3 Retrievers
    print("\n[2] Khởi tạo 3 phương pháp Retrieval:")
    t_tfidf = TfidfRetriever(chunks)
    print("  ✓ Method 1: Lexical Retrieval (TF-IDF)")
    t_dense = DenseRetriever(chunks, cache_dir=data_dir)
    print("  ✓ Method 2: Dense Retrieval (FastEmbed 384d paraphrase-multilingual-MiniLM-L12-v2)")
    t_hybrid = HybridRetriever(t_dense, t_tfidf, rrf_k=60)
    print("  ✓ Method 3: Hybrid Retrieval (Dense + Lexical qua RRF)")

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 2. Đánh giá so sánh Retrieval trên tập Dev
    print("\n[3] BƯỚC 1: So sánh 3 phương pháp Retrieval trên tập Dev (n=15 có đáp án):")
    retrieval_results = {}
    for name, ret in [("Lexical (TF-IDF)", t_tfidf), ("Dense (Day 6)", t_dense), ("Hybrid (RRF)", t_hybrid)]:
        res_800 = evaluate_retrieval(ret, dev_questions, budget=800, top_k=3)
        res_2000 = evaluate_retrieval(ret, dev_questions, budget=2000, top_k=3)
        retrieval_results[name] = {"b800": res_800, "b2000": res_2000}
        print(f"  -> {name:<18} | Hit@1: {res_800['hit1']}/15 ({res_800['hit1_rate']*100:.1f}%) | "
              f"Hit@3: {res_800['hit3']}/15 ({res_800['hit3_rate']*100:.1f}%) | "
              f"ContextRate (b800): {res_800['context_hit']}/15 ({res_800['context_hit_rate']*100:.1f}%) | "
              f"ContextRate (b2000): {res_2000['context_hit']}/15 ({res_2000['context_hit_rate']*100:.1f}%) | "
              f"Avg Time: {res_800['avg_ms']:.2f}ms")

    with open(os.path.join(results_dir, "retrieval_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

    # 3. Đánh giá Agent trên tập Dev qua 3 Cấu hình
    print("\n[4] BƯỚC 2: Thử nghiệm Agent trên tập Dev (17 câu: 15 có đáp án + 2 câu thiếu bằng chứng):")
    controller = GroqAgentController()

    agent_configs = [
        {"name": "Config_1_Baseline", "desc": "Lexical TF-IDF, Budget 800, max_searches=1", "retriever": t_tfidf, "budget": 800, "max_searches": 1},
        {"name": "Config_2_Improved_Budget", "desc": "Dense Retrieval, Budget 2000, max_searches=1", "retriever": t_dense, "budget": 2000, "max_searches": 1},
        {"name": "Config_3_Agentic_Hybrid_Best", "desc": "Hybrid RRF, Budget 2000, max_searches=2", "retriever": t_hybrid, "budget": 2000, "max_searches": 2}
    ]

    for cfg in agent_configs:
        cname = cfg["name"]
        print(f"\n--- Đang chạy Agent: {cname} ({cfg['desc']}) ---")
        traces, reviews = evaluate_agent(
            cases=dev_cases,
            retriever=cfg["retriever"],
            controller=controller,
            budget=cfg["budget"],
            max_searches=cfg["max_searches"],
            top_k=3
        )
        summary = save_and_aggregate(dev_cases, traces, reviews, os.path.join(results_dir, cname), "dev")
        gw = summary["grounded_answer"]
        print(f"  Result {cname}:")
        print(f"    - Tool use: {summary['tool_use']['n']}/{summary['tool_use']['N']}")
        print(f"    - Grounded answer mean: {gw['mean']}/2.0 (Scores: 2={gw['2']}, 1={gw['1']}, 0={gw['0']})")
        print(f"    - Citation validity: {summary['citation_validity']['n']}/{summary['citation_validity']['N']}")
        print(f"    - Abstention rate on unanswerable: {summary['abstention']['n']}/{summary['abstention']['N']}")
        print(f"    - False abstention on answerable: {summary['false_abstention']['n']}/{summary['false_abstention']['N']}")

    # 4. CHỐT CẤU HÌNH TỐI ƯU TRÊN DEV VÀ ĐÁNH GIÁ TRÊN TEST
    print("\n" + "=" * 70)
    print("[5] BƯỚC 3: Chốt cấu hình tối ưu trên Dev -> Chạy đánh giá độc lập trên tập Test (Hold-out):")
    print("Cấu hình được chọn: Config_3_Agentic_Hybrid_Best (Hybrid RRF + Budget 2000 + Multi-turn search)")
    print("=" * 70)

    test_traces, test_reviews = evaluate_agent(
        cases=test_cases,
        retriever=t_hybrid,
        controller=controller,
        budget=2000,
        max_searches=2,
        top_k=3
    )

    test_summary = save_and_aggregate(test_cases, test_traces, test_reviews, os.path.join(results_dir, "agent_test_best"), "test")
    tgw = test_summary["grounded_answer"]

    print("\nKẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (HOLD-OUT):")
    print(f"  - Tool use: {test_summary['tool_use']['n']}/{test_summary['tool_use']['N']}")
    print(f"  - Grounded answer mean: {tgw['mean']}/2.0 (Scores: 2={tgw['2']}, 1={tgw['1']}, 0={tgw['0']})")
    print(f"  - Citation validity: {test_summary['citation_validity']['n']}/{test_summary['citation_validity']['N']}")
    print(f"  - Abstention rate on unanswerable: {test_summary['abstention']['n']}/{test_summary['abstention']['N']}")
    print(f"  - False abstention on answerable: {test_summary['false_abstention']['n']}/{test_summary['false_abstention']['N']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
