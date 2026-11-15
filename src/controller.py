"""LLM Controller for Agentic RAG using Groq / OpenAI compatible API."""
import json
import os
from typing import Any
from openai import OpenAI


class GroqAgentController:
    """Agent Controller dùng mô hình ngôn ngữ lớn thật (openai/gpt-oss-120b qua Groq API)."""
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b"):
        key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") if key else None
        self.model = model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Quyết định hành động: gọi tìm kiếm (search) hoặc trả lời (answer)."""
        if not self.client:
            return {"action": "search", "query": question}
        prompt = self._build_prompt(question, history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_text = response.choices[0].message.content or ""
            parsed = json.loads(raw_text)
            
            # Validation
            action = parsed.get("action")
            if action == "search":
                query = parsed.get("query", "").strip()
                if not query:
                    query = question
                return {"action": "search", "query": query}
            elif action == "answer":
                ans = parsed.get("answer", "").strip()
                cits = parsed.get("citations", [])
                if not isinstance(cits, list):
                    cits = []
                # Ensure citations match citations list and exist in answer
                return {"action": "answer", "answer": ans, "citations": cits}
            else:
                return {"action": "search", "query": question}
        except Exception as e:
            # Fallback if connection fails
            return {"action": "search", "query": question}

    def _build_prompt(self, question: str, history: list[dict[str, Any]]) -> str:
        history_text = ""
        if not history:
            history_text = "Chưa có lượt tìm kiếm nào."
        else:
            blocks = []
            for h in history:
                q_list = ", ".join(f"'{q}'" for q in h.get("queries", []))
                ctx = h.get("context", "")
                blocks.append(f"- Các truy vấn đã tìm: [{q_list}]\n- Ngữ cảnh đã trích xuất:\n{ctx}")
            history_text = "\n".join(blocks)

        prompt = f"""Bạn là một Agent chuyên nghiệp tra cứu và giải đáp văn bản pháp luật Việt Nam.
Nhiệm vụ: Dựa vào câu hỏi và lịch sử các lượt tìm kiếm + ngữ cảnh (context), bạn hãy quyết định hành động tiếp theo.

QUY TẮC BẮT BUỘC:
1. Nếu chưa có ngữ cảnh hoặc ngữ cảnh hiện tại chưa đủ thông tin/bằng chứng trực tiếp để trả lời:
   Trả về JSON:
   {{
     "action": "search",
     "query": "<câu truy vấn tìm kiếm tiếng Việt ngắn gọn, giàu từ khóa>"
   }}

2. Nếu trong ngữ cảnh ĐÃ CÓ ĐẦY ĐỦ bằng chứng pháp lý để trả lời chính xác câu hỏi:
   Trả về JSON:
   {{
     "action": "answer",
     "answer": "<câu trả lời súc tích, trung thực với ngữ cảnh. BẮT BUỘC chèn ký hiệu trích dẫn [chunk_id] vào nội dung câu trả lời>",
     "citations": ["<chunk_id>"]
   }}
   Lưu ý: Mọi chunk_id trong mảng citations PHẢI xuất hiện dưới dạng [chunk_id] trong trường answer và PHẢI nằm trong danh sách context_ids đã cung cấp.

3. Nếu đã tìm kiếm nhưng ngữ cảnh HOÀN TOÀN KHÔNG CÓ thông tin hoặc không liên quan đến câu hỏi (câu hỏi nằm ngoài phạm vi tài liệu):
   Trả về JSON:
   {{
     "action": "answer",
     "answer": "Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này.",
     "citations": []
   }}

Câu hỏi người dùng:
{question}

Lịch sử truy xuất:
{history_text}

Chỉ trả về định dạng JSON hợp lệ duy nhất:"""
        return prompt
