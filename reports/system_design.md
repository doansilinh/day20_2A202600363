# Multi-Agent Research System Design

## Problem
Hệ thống cần xử lý các yêu cầu nghiên cứu chuyên sâu, đòi hỏi khả năng tìm kiếm thông tin từ nhiều nguồn, phân tích dữ liệu đa chiều, và tổng hợp thành báo cáo có cấu trúc chặt chẽ với trích dẫn đầy đủ. Các hệ thống Single-Agent thường gặp khó khăn trong việc duy trì độ chính xác và chiều sâu khi xử lý các task phức tạp này.

## Why multi-agent?
Single-agent thường gặp các vấn đề sau:
- **Hallucination**: Tự đưa ra thông tin không có thực khi không tìm thấy dữ liệu.
- **Thiếu kiểm chứng**: Không có quy trình thẩm định lại nội dung đã viết.
- **Quá tải ngữ cảnh**: Phải làm quá nhiều việc cùng lúc (tìm kiếm, phân tích, viết lách) dẫn đến chất lượng giảm sút.

**Hệ thống Multi-agent** giải quyết bằng cách chia nhỏ trách nhiệm, cho phép mỗi Agent tập trung vào một kỹ năng chuyên biệt và có Agent Critic để đảm bảo chất lượng đầu ra.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng công việc, quyết định Agent tiếp theo dựa trên trạng thái hiện tại. | `ResearchState` | Tên Agent tiếp theo | Chọn sai route dẫn đến lặp vô hạn. |
| **Researcher** | Tìm kiếm thông tin từ web/database, thu thập các nguồn tin cậy. | `query` | `research_notes` | Không tìm thấy nguồn hoặc nguồn thiếu tin cậy. |
| **Analyst** | Phân tích các ghi chú từ Researcher, tìm ra các điểm mấu chốt và khoảng trống thông tin. | `research_notes` | `analysis_notes` | Phân tích hời hợt, bỏ sót chi tiết quan trọng. |
| **Writer** | Tổng hợp phân tích thành báo cáo hoàn chỉnh, định dạng Markdown và trích dẫn nguồn. | `analysis_notes` | `final_answer` | Quên trích dẫn hoặc hành văn không chuyên nghiệp. |
| **Critic** | Đánh giá chất lượng báo cáo dựa trên thang điểm 10 và đưa ra phản hồi cải thiện. | `final_answer` | `quality_score`, `feedback` | Chấm điểm quá lỏng lẻo hoặc quá khắt khe. |

## Shared state
Hệ thống sử dụng `ResearchState` để lưu trữ:
- `research_notes`: Dữ liệu thô từ Researcher.
- `analysis_notes`: Kết quả xử lý từ Analyst.
- `final_answer`: Bản thảo báo cáo từ Writer.
- `route_history`: Lịch sử các bước chạy để tránh lặp.
- `quality_score`: Điểm số hiện tại của báo cáo.
- `iteration_count`: Đếm số lượt chạy để kích hoạt Guardrails.

## Routing policy
Luồng chạy đi theo sơ đồ Graph của LangGraph:
**Supervisor** -> **Researcher** -> **Analyst** -> **Writer** -> **Critic**.
- Nếu `quality_score < 7`: Quay lại **Researcher** để bổ sung thông tin hoặc sửa lỗi.
- Nếu `quality_score >= 7` hoặc đạt `max_iterations`: Kết thúc và trả kết quả.

## Guardrails
- **Max iterations**: Giới hạn 10 lượt chạy để tránh lặp vô tận và tốn phí API.
- **Quality Gate**: Chặn đầu ra nếu điểm chất lượng không đạt yêu cầu tối thiểu (7/10).
- **Graceful Degradation**: Tự động chuyển sang chế độ Mock LLM/Search nếu gặp lỗi API hoặc thiếu API Key.
- **Pydantic Validation**: Đảm bảo dữ liệu trao đổi giữa các Agent luôn đúng cấu trúc.
