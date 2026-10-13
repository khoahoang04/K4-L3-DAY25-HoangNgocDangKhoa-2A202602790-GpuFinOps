# BÁO CÁO KỸ THUẬT TỐI ƯU HÓA GPU FINOPS — NIMBUSAI
**Khóa học:** AICB · Phase 2 · Track 2 · Day 25: GPU FinOps Optimization  
**Học viên:** Hoàng Ngọc Đăng Khoa (2A202602790)  
**Ngày thực hiện:** 13/10/2026  

---

## 1. Tổng quan hiệu quả tài chính (Baseline vs. Optimized Spend)

Qua quá trình kiểm toán và tối ưu hóa toàn diện trên cụm hạ tầng tính toán của NimbusAI (bao gồm cả inference serving và batch training workloads), kết quả tài chính đạt được như sau:

| Chỉ số kinh tế | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Mức giảm / Tiết kiệm | Tỷ lệ tiết kiệm (%) |
|---|---|---|---|---|
| **Tổng ngân sách tháng** | **$27,133 / tháng** | **$14,550 / tháng** | **$12,583 / tháng** | **46.4%** |
| **Đơn giá Serving ($/1M-token)** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **$5.362 / 1M-token** | **82.6%** |
| **Chi phí Inference ngày** | $48.87 / ngày | $8.48 / ngày | $40.39 / ngày | 82.6% |
| **Chi phí Purchasing tháng** | $25,667 / tháng | $15,551 / tháng | $10,116 / tháng | 39.4% |
| **Lãng phí GPU nhàn rỗi (Idle)** | $600 / tháng | $0 / tháng | $600 / tháng | 100.0% |

> **Nhận xét then chốt:** Chi phí vận hành hạ tầng AI đã giảm gần một nửa (46.4%), trong khi đơn giá kinh tế đơn vị (unit economics) của hệ thống phục vụ suy luận giảm mạnh tới **82.6%**, tạo biên lợi nhuận bền vững cho sản phẩm AI của NimbusAI.

---

## 2. Phân tích chi tiết từng đòn bẩy tối ưu (Savings by Lever)

Biểu đồ Waterfall (`outputs/savings.png`) và dữ liệu thực nghiệm phân tách tổng khoản tiết kiệm $12,583/tháng thành 4 đòn bẩy:

| Đòn bẩy (Lever) | Tiết kiệm (USD/tháng) | Tỷ trọng đóng góp (%) | Cơ chế kỹ thuật |
|---|---|---|---|
| **Purchasing Strategy (Spot/Reserved)** | **$10,116** | **80.4%** | Phân loại workload theo duty cycle & interruptibility: Dùng Spot + checkpointing cho batch jobs và Reserved commitments cho 24/7 jobs |
| **Inference Optimization (Cascade/Cache/Batch)** | **$1,212** | **9.6%** | Chuyển 80% query dễ sang model Small, Prompt Caching cho system prompts (-90% read), Batch API cho eval (-50%) |
| **Right-sizing Util-Lies** | **$655** | **5.2%** | Hạ cấp GPU bị nghẽn bộ nhớ (`gpu-h100-4` xuống A100, `gpu-a10g-1` xuống L4) |
| **Shutdown Idle GPUs** | **$600** | **4.8%** | Tự động tắt instance bỏ không ban đêm (`gpu-h100-5` lãng phí 8h/ngày) |
| **TỔNG CỘNG** | **$12,583** | **100%** | |

### Đòn bẩy đóng góp lớn nhất: **Purchasing Strategy ($10,116/tháng — 80.4%)**
- **Tại sao đóng góp vượt trội?** Chi phí thuê GPU thô là thành phần chiếm tỷ trọng áp đảo trong hóa đơn cloud ($25,667 trên tổng $27,133 ban đầu, tương đương 94.6%).
- Việc chuyển các job huấn luyện lớn như `job-train-llm` (8x H100 chạy 20h/ngày) sang mô hình **Spot instances** giúp giảm chi phí từ $12,000 xuống $7,524 (tiết kiệm gần $4,500/tháng chỉ trên một job). Các dịch vụ inference chạy liên tục 24/7 (`job-infer-chat`, `job-infer-rag`) khi cam kết **Reserved 3-year** giúp giảm tới 40–45% giá thuê giờ cố định.

---

## 3. Bản chất hiện tượng "GPU-Util Lie" và Tác động tài chính

### Cơ chế kỹ thuật
Trong công cụ giám sát phổ biến `nvidia-smi`, chỉ số **`GPU-Util %` thực chất chỉ là một chiếc đồng hồ bấm giờ (temporal activity clock)**. Nó ghi nhận tỷ lệ phần trăm thời gian có *ít nhất một CUDA kernel* đang hoạt động trên GPU trong chu kỳ lấy mẫu, hoàn toàn **không phản ánh mức độ bão hòa tính toán của Tensor Cores**.

Một GPU có thể hiển thị `GPU-Util = 98% - 100%` trong khi hiệu suất tính toán thực tế (**MFU — Model FLOPs Utilization**) chỉ đạt 15–20% do hai nguyên nhân cốt lõi:
1. **Memory Bandwidth Stall (Nghẽn băng thông bộ nhớ):** Trong quá trình sinh từ (autoregressive decoding), mô hình bị memory-bound. Tensor Cores phải liên tục tạm dừng chờ nạp trọng số và KV Cache từ HBM. GPU luôn ở trạng thái "bận" nhưng compute pipeline thực chất bị bỏ trống.
2. **Kernel Launch Overhead & Small Batch:** Chạy kích thước batch quá nhỏ hoặc mô hình chưa được tối ưu CUDA Graph / fused kernel khiến thời gian điều phối CPU-GPU chiếm ưu thế.

### Dữ liệu thực tế tại cụm NimbusAI
- **`gpu-h100-4` (H100, Training):** Báo cáo `GPU-Util = 98.2%` nhưng MFU chỉ đạt **19.4%** (ngưỡng chuẩn training phải từ 35–45%).
- **`gpu-a10g-1` (A10G, Inference):** Báo cáo `GPU-Util = 96.9%` nhưng MFU chỉ đạt **26.8%**.

### Tác động tài chính
NimbusAI đang trả mức giá cao cấp của GPU thế hệ mới ($2.50/giờ cho H100) nhưng chỉ thu lại năng lực tính toán của một GPU thế hệ cũ (A100 $1.79/giờ hoặc thấp hơn). Khoản chi lãng phí trực tiếp trên các GPU này là **$655 – $1,224/tháng**, và chi phí cơ hội do không thể tận dụng hết năng lực phần cứng để rút ngắn thời gian thử nghiệm.

---

## 4. Chi tiết các phần mở rộng kỹ thuật ("Your Turn" Extensions)

### Extension 3: `cache_is_worth_it()` — Prompt Caching
* **Vị trí cài đặt:** Hàm `cache_is_worth_it()` và `cache_break_even_reads()` trong [finops/pricing.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/finops/pricing.py), gọi thực thi trong [missions/m2_inference_levers.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/missions/m2_inference_levers.py).
* **Nguyên lý hòa vốn (Break-even):** Ghi cache phát sinh chi phí lưu trữ/surcharge. Tiền tiết kiệm từ các lượt đọc lặp lại phải bù đắp được chi phí ghi:
  $$\text{Break-even reads} = \frac{\text{Write Cost per 1M}}{P_{\text{base}} \times (1 - \text{read discount})}$$
  Với mô hình Anthropic Prompt Caching (write cost = 1.25x base price, read discount = 0.10 tức giảm 90%):
  $$\text{Break-even} = \frac{1.25 \times P_{\text{in}}}{0.90 \times P_{\text{in}}} = \frac{1.25}{0.90} \approx 1.39 \text{ reads}$$
  *(Ngưỡng hòa vốn đạt được khi prefix được tái sử dụng từ 1.39 lần đọc trở lên).*
* **Kết quả đo lường thực nghiệm:**
  - **Kịch bản tái sử dụng cao (High-reuse — giả định 5.0 lần đọc/prefix như Chat/RAG):** `cache_is_worth_it = True`. Chi phí serving đạt **$1.126/1M-token** (chưa tính ghi) và **$1.170/1M-token** (đã tính khấu hao chi phí ghi cache).
  - **Kịch bản tái sử dụng thấp (Low-reuse — giả định 1.0 lần đọc/prefix):** `cache_is_worth_it = False`. Hệ thống tự động bypass caching vì áp dụng cache sẽ làm đội chi phí lên cao hơn đọc bình thường ($1.282/1M-token).
* **Kết luận:** Dataset NimbusAI đạt ngưỡng hòa vốn trong các tác vụ Chat và RAG nhờ prompt hệ thống tĩnh được gọi lặp lại nhiều lần.

---

### Extension 4: Ngân sách Reasoning (Reasoning Energy & Budget)
* **Vị trí cài đặt:** Tích hợp trong [missions/m2_inference_levers.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/missions/m2_inference_levers.py) và [missions/m5_report.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/missions/m5_report.py).
* **Bảng phân tách lưu lượng vs Tiêu thụ tài nguyên:**

| Nhóm truy vấn | Số Requests (% traffic) | Tokens (% tokens) | Chi phí ($/ngày) | Năng lượng (Wh/ngày) |
|---|---|---|---|---|
| **Standard (Non-reasoning)** | 2,199 (**91.6%**) | 6,291,871 (**83.5%**) | $7.09 (**83.5%**) | 1,887.6 Wh (**6.0%**) |
| **Reasoning (Chain-of-Thought)** | 201 (**8.4%**) | 1,241,156 (**16.5%**) | $1.40 (**16.5%**) | 29,787.7 Wh (**94.0%**) |

* **Insight then chốt:** Mặc dù reasoning chỉ chiếm **8.4% tổng số request**, nhóm này tiêu thụ tới **94.0% tổng năng lượng điện inference** của toàn hệ thống! Nguyên nhân là do chuỗi suy nghĩ ẩn làm tăng số lượng token đầu ra gấp 6 lần và hệ số nhân năng lượng của quá trình suy luận phức tạp cao gấp ~80 lần.
* **Đo lường Cap 10% Traffic:** Do lưu lượng reasoning hiện tại (8.4%) đã ở dưới trần 10%, khoản tiết kiệm trực tiếp từ chính sách cap 10% là **$0.00 / 0.0 Wh**.
* **Đề xuất Dynamic Routing Policy:** Thiết lập bộ phân loại độ phức tạp (task complexity classifier): các bài toán cơ bản chuyển về mô hình tiêu chuẩn; chỉ kích hoạt reasoning khi score tin cậy < threshold. Chuyển đổi 50% reasoning không cần thiết giúp tiết kiệm **~14,708 Wh/ngày**.

---

### Extension 5: Carbon-Aware Scheduling cho Interruptible Workloads
* **Vị trí cài đặt:** Tích hợp trong [missions/m3_purchasing.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/missions/m3_purchasing.py) và [finops/sustainability.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/finops/sustainability.py).
* **Workloads được lập lịch:** 5 tác vụ `interruptible=1` (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval`), tiêu thụ tổng cộng **4,227 kWh/tháng** (theo tháng chuẩn 30 ngày) hoặc 2,246 kWh (theo ngày thực tế).
* **Bảng so sánh 5 vùng đám mây:**

| Vùng đám mây (Region) | Nguồn điện chính | Cường độ Carbon (gCO2/kWh) | Giá điện ($/kWh) | Phát thải tháng (kg CO2e) | Tiền điện tháng ($) | Đánh giá FinOps |
|---|---|---|---|---|---|---|
| **europe-north1** (Na Uy) | Thủy điện & Gió | **30** | $0.090 | **126.8** | $380.43 | **Sạch nhất (-92.1% CO2)** |
| **us-east-wa** (Washington) | Thủy điện Columbia | 90 | **$0.055** | 380.4 | **$232.49** | **Rẻ nhất (-54.2% tiền điện)** |
| **us-west-2** (Oregon) | Hỗn hợp sạch | 120 | $0.070 | 507.2 | $295.89 | Cân bằng tốt |
| **us-east-1** (Virginia - Gốc) | Nhiệt điện than/khí | 380 | $0.120 | 1,606.3 | $507.24 | Mức cơ sở (Baseline) |
| **europe-central2** (Ba Lan) | Than đá | 660 | $0.180 | 2,789.8 | $760.86 | Ô nhiễm & đắt nhất |

* **Đo lường giảm phát thải:** Chuyển 5 batch jobs từ `us-east-1` sang `europe-north1` cắt giảm **92.1% lượng phát thải carbon**, giảm **1,479.5 kg CO2e mỗi tháng**.
* **Đánh giá Network Latency Trade-off:** Các tác vụ này là batch training, offline evaluation và sandbox, xử lý phi đồng bộ (asynchronous). Độ trễ mạng (latency) xuyên đại dương hoàn toàn **không ảnh hưởng đến SLA người dùng cuối**. Do đó, việc dời vùng sang `europe-north1` (sạch nhất) hoặc `us-east-wa` (rẻ nhất) là một quyết định tối ưu không có đánh đổi bất lợi.

---

### Extension 1 & 2: Cải thiện `recommend_tier()` và Right-sizing theo MBU
* **Extension 1:** Hàm `recommend_tier()` trong [finops/pricing.py](K4-L3-DAY25-HoangNgocDangKhoa-2A202602790-GpuFinOps/finops/pricing.py) được mở rộng để nhận diện rủi ro thu hồi Spot (`interrupt_rate`) theo loại GPU (H100 có tỷ lệ thu hồi thấp ~3% so với A10G ~8%), giúp tăng độ tin cậy và tiết kiệm thêm chi phí tính toán Spot checkpointing.
* **Extension 2:** Phân tích catalog theo `$/GB-VRAM` và `$/TB/s bandwidth`. Nhận diện workload memory-bound trên A100 có thể hạ cấp xuống 2x L4 hoặc A10G để tiết kiệm **$1,224/tháng**. Ghi nhận giới hạn: Cần benchmark thực nghiệm trước khi triển khai production để đảm bảo serving throughput và p99 latency SLA không bị suy giảm.

---

## 5. Khuyến nghị cho NimbusAI: 3 Hành động đầu tiên của FinOps Lead

Nếu đảm nhiệm vai trò FinOps Lead tại NimbusAI, 3 hành động tiên quyết trong 30 ngày đầu tiên gồm:

1. **Triển khai Spot Automation & Checkpointing Pipeline ngay lập tức (P0):**
   - Di chuyển toàn bộ các workload batch training và offline eval (`job-train-llm`, `job-train-embed`, `job-batch-eval`) sang Spot instances với kiến trúc checkpointing tự động lưu vào S3/GCS.
   - Ký kết cam kết 3 năm (Reserved 3yr) cho cụm suy luận 24/7 (`job-infer-chat`, `job-infer-rag`).
   - *Tác động tài chính:* Khóa ngay khoản tiết kiệm **$10,116/tháng (chiếm 80% tổng tiềm năng)**.

2. **Áp dụng Model Cascading & Prompt Caching Gateway (P1):**
   - Đặt một proxy gateway phân luồng phía trước hệ thống serving: 80% truy vấn đơn giản được định tuyến sang model Small; bật Prompt Caching cho các system prompt và tài liệu RAG dài (>1.39 reads).
   - Thiết lập rào chắn (gate) cho Reasoning queries: chỉ kích hoạt khi độ khó bài toán vượt ngưỡng, hạn chế tối đa việc lãng phí token và năng lượng.
   - *Tác động tài chính:* Giảm giá thành serving từ **$6.488 xuống $1.126/1M-token (-82.6%)**, tiết kiệm **$1,212/tháng**.

3. **Thiết lập Chính sách Carbon-aware Dispatching & Tagging Governance (P2):**
   - Cấu hình scheduler điều phối các job batch huấn luyện và fine-tuning chạy tại vùng `europe-north1` (sạch nhất) hoặc `us-east-wa` (rẻ nhất), giảm ngay 92% phát thải carbon và giảm hơn 50% tiền điện.
   - Đóng cổng phê duyệt provisioning nếu thiếu tags (`team`, `project`), duy trì tag coverage >= 85% để vận hành hệ thống Chargeback tự động minh bạch.
   - Tự động tắt instance dev/training không hoạt động ngoài giờ hành chính.
