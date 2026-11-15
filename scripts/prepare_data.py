import json
import os
import sys

sys.path.insert(0, os.path.abspath('Lab 6 Data Ingestion/K5-L3-Day06-DataIngestionEngineering/starter'))
from src.ingestion import load_corpus

def main():
    root_day6 = os.path.abspath('Lab 6 Data Ingestion/K5-L3-Day06-DataIngestionEngineering')
    benchmark_file = os.path.join(root_day6, 'benchmark', 'questions.json')
    corpus_file = os.path.join(root_day6, 'starter', 'data', 'legal_truncated_corpus.csv')

    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)

    all_q = benchmark['dev'] + benchmark['test']
    target_ids = {gid for q in all_q for gid in q.get('gold_doc_ids', [])}

    print("Loading corpus...")
    docs = load_corpus(corpus_file, max_docs=100, target_ids=target_ids)
    print(f"Loaded {len(docs)} documents.")

    docs_by_id = {d.doc_id: d for d in docs}

    # 1. Chunks
    chunks = []
    for d in docs:
        page_val = d.metadata.get("dieu") or "Điều 1"
        chunks.append({
            "id": f"B_{d.doc_id}",
            "source_id": d.doc_id,
            "source": "legal_truncated_corpus.csv",
            "page": page_val,
            "text": d.text,
            "metadata": {
                "title": d.title,
                "loai_vb": d.metadata.get("loai_vb", "Văn bản khác"),
                "dieu": page_val
            }
        })

    # Expected answers mapping for the 20 answerable questions
    expected_answers = {
        0: "Chứng nhận hợp quy là việc xác nhận sản phẩm phù hợp quy chuẩn kỹ thuật; Công bố hợp quy là việc tổ chức, cá nhân công bố sản phẩm phù hợp quy chuẩn kỹ thuật.",
        1: "Không. Pháp luật nuôi con nuôi Việt Nam không quy định độ tuổi tối đa của người nhận con nuôi, chỉ quy định hơn con nuôi từ 20 tuổi trở lên.",
        2: "Văn bản đề nghị của TCTD phi ngân hàng, Nghị quyết HĐQT/HĐTV, đề án thành lập chi nhánh, tài liệu chứng minh đáp ứng điều kiện.",
        4: "Có. Khách du lịch đường biển có hộ chiếu hoặc giấy thông hành hợp lệ được mua hàng miễn thuế tại cửa hàng miễn thuế trong khu vực cách ly cảng biển.",
        6: "Chấp hành yêu cầu của cơ quan bảo vệ, giữ bí mật thông tin bảo vệ, thông báo kịp thời khi có vấn đề phát sinh.",
        7: "Người khiếu nại, tố cáo có trách nhiệm cung cấp thông tin, tài liệu liên quan đến nội dung khiếu nại, tố cáo cho cơ quan có thẩm quyền.",
        8: "Gồm văn bản đề nghị, bản sao CMND/CCCD/Hộ chiếu hoặc giấy phép kinh doanh, đề xuất dự án đầu tư, tài liệu chứng minh năng lực tài chính.",
        11: "Được hưởng đầy đủ quyền lợi về BHXH, BHYT theo quy định của pháp luật và thỏa ước lao động tập thể.",
        14: "Tổ chức kiểm định chất lượng giáo dục nghề nghiệp do Thủ trưởng cơ quan quản lý nhà nước về giáo dục nghề nghiệp cấp phép hoạt động.",
        18: "Báo cáo định kỳ 6 tháng và hàng năm bằng văn bản hoặc phương thức điện tử gửi cơ quan đại diện chủ sở hữu.",
        23: "Phải công khai thông tin trên trang thông tin điện tử của doanh nghiệp và cổng thông tin doanh nghiệp quốc gia.",
        27: "Giám sát tài chính đặc biệt được áp dụng khi doanh nghiệp có dấu hiệu mất khả năng thanh toán hoặc thua lỗ kéo dài.",
        31: "Thời hạn giải quyết khiếu nại lần đầu không quá 30 ngày; đối với vụ việc phức tạp không quá 45 ngày.",
        35: "Ủy ban nhân dân cấp tỉnh có thẩm quyền công nhận ban vận động thành lập hội trên địa bàn tỉnh.",
        42: "Hồ sơ công bố hợp quy nộp trực tiếp hoặc gửi qua bưu điện đến cơ quan chuyên môn thuộc Bộ Thông tin và Truyền thông.",
        50: "Gồm biên bản kiểm tra lâm sản, bảng kê lâm sản, hóa đơn giá trị gia tăng hoặc hồ sơ nguồn gốc hợp pháp của lâm sản.",
        55: "Chủ rừng chịu trách nhiệm phòng cháy, chữa cháy rừng; xây dựng phương án PCCC rừng và tổ chức lực lượng tại chỗ.",
        60: "Sở Nông nghiệp và Phát triển nông thôn cấp chứng chỉ quản lý rừng bền vững theo tiêu chuẩn quốc gia.",
        65: "Thông tin an toàn thực phẩm phải bảo đảm chính xác, trung thực, kịp thời và có căn cứ khoa học.",
        70: "Cơ sở kinh doanh thực phẩm phải lưu giữ hồ sơ truy xuất nguồn gốc thực phẩm trong thời hạn tối thiểu theo quy định của pháp luật chuyên ngành."
    }

    # 2. Questions (Answerable)
    questions = []
    cases = []
    all_questions = []

    for split in ['dev', 'test']:
        for q in benchmark[split]:
            qid = q['id']
            gid = q['gold_doc_ids'][0]
            d = docs_by_id[gid]
            page_val = d.metadata.get("dieu") or "Điều 1"

            # Clean evidence: remove trailing ellipsis
            raw_ev = q['evidence']
            clean_ev = raw_ev.rstrip('.').strip()
            # Verify clean_ev is in d.text
            if clean_ev not in d.text:
                # Find matching prefix
                for length in range(len(clean_ev), 10, -5):
                    if clean_ev[:length] in d.text:
                        clean_ev = clean_ev[:length]
                        break

            exp_ans = expected_answers.get(qid, "Thông tin được quy định cụ thể tại văn bản pháp luật.")
            q_entry = {
                "id": f"q_{qid}",
                "split": split,
                "question": q["question"],
                "expected_answer": exp_ans,
                "gold": [
                    {
                        "source_id": gid,
                        "page": page_val,
                        "evidence": clean_ev
                    }
                ]
            }
            if q.get("filter"):
                q_entry["filter"] = q["filter"]
            questions.append(q_entry)

            cases.append({
                "id": f"q_{qid}",
                "split": split,
                "question": q["question"],
                "answerable": True,
                "expected_behavior": "search_and_answer"
            })

            all_questions.append({
                **q_entry,
                "answerable": True,
                "expected_behavior": "search_and_answer"
            })

    # 3. Unanswerable questions (2 for dev, 2 for test)
    unanswerables = [
        {
            "id": "q_dev_unans_1",
            "split": "dev",
            "question": "Mức xử phạt tiền tối đa đối với hành vi vi phạm nồng độ cồn khi lái xe ô tô theo Nghị định 100/2019/NĐ-CP là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tập ngữ cảnh 118 điều luật chỉ bao gồm các văn bản về đầu tư, nuôi con nuôi, khiếu nại, tố cáo, kiểm định giáo dục và lâm nghiệp; hoàn toàn không chứa Nghị định 100/2019/NĐ-CP về xử phạt vi phạm giao thông."
        },
        {
            "id": "q_dev_unans_2",
            "split": "dev",
            "question": "Thời hạn tối đa của hợp đồng lao động xác định thời hạn theo Bộ luật Lao động 2019 là bao nhiêu tháng?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tập dữ liệu không chứa các điều khoản của Bộ luật Lao động 2019 quy định về loại và thời hạn hợp đồng lao động."
        },
        {
            "id": "q_test_unans_1",
            "split": "test",
            "question": "Theo Luật Hôn nhân và Gia đình hiện hành, độ tuổi kết hôn tối thiểu đối với nam và nữ tại Việt Nam là bao nhiêu tuổi?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Corpus không có quy định về điều kiện kết hôn và độ tuổi kết hôn của Luật Hôn nhân và Gia đình."
        },
        {
            "id": "q_test_unans_2",
            "split": "test",
            "question": "Mức đóng bảo hiểm y tế bắt buộc của học sinh, sinh viên được ngân sách nhà nước hỗ trợ tối thiểu bao nhiêu phần trăm?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong toàn bộ 118 điều luật không có văn bản pháp luật nào quy định về mức đóng và tỷ lệ hỗ trợ BHYT cho học sinh sinh viên."
        }
    ]

    for unans in unanswerables:
        cases.append({
            "id": unans["id"],
            "split": unans["split"],
            "question": unans["question"],
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": unans["absence_reason"]
        })
        all_questions.append(unans)

    # Save to data directories
    out_dir = os.path.abspath('Lab 7 Agentic RAG/K5-L3A-DAY07-VuDucAnh-2A202603714-RetrievalContext/data')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, 'questions.json'), 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, 'cases.json'), 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, 'all_questions.json'), 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print("Success! Generated:")
    print(f"- {len(chunks)} chunks in data/chunks.json")
    print(f"- {len(questions)} answerable questions in data/questions.json")
    print(f"- {len(cases)} total benchmark cases ({len([c for c in cases if c['split']=='dev'])} dev, {len([c for c in cases if c['split']=='test'])} test) in data/cases.json")

if __name__ == '__main__':
    main()
