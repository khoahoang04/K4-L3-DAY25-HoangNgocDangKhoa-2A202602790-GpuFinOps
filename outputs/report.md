# NimbusAI — Báo cáo Tối ưu hóa Chi phí GPU FinOps

**Kỳ báo cáo (Period):** monthly (Hàng tháng)  
**Chi phí cơ sở (Baseline spend):** $27,133  
**Chi phí sau tối ưu (Optimized spend):** $14,550  
**Tiết kiệm dự kiến (Projected savings):** $12,583  (**46%**)

## Tiết kiệm theo từng đòn bẩy (Savings by Lever)

| Đòn bẩy tối ưu (Lever) | Tiết kiệm (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,116 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Phân tích Chuyên sâu & Nguyên nhân Gốc rễ (Root Cause Analysis)

### 1. Cơ chế và bản chất hiện tượng 'GPU-Util Lie'
- **Bản chất kỹ thuật:** Chỉ số `GPU-Util %` trong công cụ `nvidia-smi` thực chất chỉ là chiếc đồng hồ đo thời gian (time-active clock), phản ánh tỷ lệ phần trăm thời gian có ít nhất một CUDA kernel đang được nạp và hoạt động trên GPU trong chu kỳ đo mẫu. Chỉ số này **hoàn toàn KHÔNG phản ánh hiệu suất tính toán thực tế hay mức độ bão hòa năng lực của Tensor Cores**.
- **Nguyên nhân gốc rễ (Root Causes):**
  - **Nghẽn băng thông bộ nhớ (Memory bandwidth stall):** Trong quá trình phục vụ suy luận tự hồi quy (autoregressive LLM decode), workload bị memory-bound. Tensor Cores liên tục rơi vào trạng thái chờ nạp trọng số và KV Cache từ bộ nhớ HBM.
  - **Overhead điều phối kernel & Batch nhỏ:** Chạy với batch size nhỏ khiến thời gian CPU launch kernel và đồng bộ stream chiếm ưu thế so với thời gian tính toán thực tế.
- **Thực trạng đội tàu (Fleet Telemetry):** Telemetry đo đạc phát hiện GPU `gpu-h100-4` (H100 chạy training) báo cáo **GPU-Util lên tới 98.2%** nhưng MFU (Model FLOPs Utilization) thực tế chỉ đạt **19.4%** (dưới xa ngưỡng chuẩn 35-45% của bài toán training). Tương tự, `gpu-a10g-1` báo **96.9% GPU-Util** nhưng MFU chỉ vỏn vẹn **26.8%**.
- **Hệ quả tài chính:** NimbusAI đang chi trả giá thuê on-demand đắt đỏ ($2.50/giờ cho H100) cho năng lực tính toán chỉ tương đương một GPU A100 ($1.79/giờ) hoặc thấp hơn, gây thất thoát từ **$511 đến $1,224/tháng**.

### 2. Kế hoạch Hành động Ưu tiên theo ROI (Prioritized FinOps Action Plan)
| Mức ưu tiên | Đòn bẩy (Lever) | Tiết kiệm tháng | Rationale & Kế hoạch hành động |
|---|---|---|---|
| **P0 (Ngay lập tức)** | **Chiến lược Mua sắm (Spot & Commitments)** | **$10,040** | Đóng góp lớn nhất (~80% tổng savings). Chuyển ngay các job batch training và eval (có `interruptible=1`) sang Spot instances kèm kiến trúc checkpointing tự động. Ký cam kết Reserved 3 năm cho các workload suy luận chạy 24/7. |
| **P1 (Tuần 1-2)** | **Tối ưu hóa Suy luận (Cascade, Cache, Batch)** | **$1,212** | Đòn bẩy cốt lõi cho dịch vụ serving. Triển khai phân tầng Model Cascade (80% query dễ chuyển sang model Small), bật Prompt Caching cho system prompt tĩnh (giảm 90% phí đọc input), và chuyển tác vụ eval sang Batch API (-50%). Giảm đơn giá serving từ **$6.488 xuống $1.126/1M-token** (-82.6%). |
| **P2 (Tuần 3-4)** | **Right-sizing GPU Nghẽn Bộ nhớ & Util-Lies** | **$655** | Hạ cấp `gpu-h100-4` xuống A100 và `gpu-a10g-1` xuống L4. Thiết lập quy trình bắt buộc kiểm tra MFU/MBU trước khi phê duyệt cấp phát GPU cao cấp. |
| **P3 (Dài hạn)** | **Tắt GPU Nhàn rỗi & Quản trị Tagging** | **$600** | Tự động hóa shutdown/hibernate GPU ngoài giờ hành chính (như `gpu-h100-5` bỏ không 8h mỗi đêm). Duy trì tỷ lệ gắn tag (coverage) >= 85% để vận hành chargeback tự động. |

## Tính Bền vững & Năng lượng (Sustainability)

- Điện năng tiêu thụ mỗi truy vấn (Energy per query): 0.24 Wh
- Lượng phát thải carbon mỗi truy vấn (Carbon per query): 0.091 gCO2e
- Vùng đám mây rẻ và sạch nhất (Cheapest+cleanest region): europe-north1

## Chi tiết các Phần Mở rộng (FinOps Engineering Extensions)

### Extension 3: cache_is_worth_it() — Kinh tế học Prompt Caching
- **Nguyên lý hòa vốn (Break-even):** Prompt caching chỉ thực sự tiết kiệm khi tổng chi phí đọc được giảm trừ lớn hơn chi phí lưu trữ/ghi ban đầu.
  - Giả định chi phí ghi = 1.25x giá input cơ sở; Chiết khấu đọc = 90% (theo mô hình định giá Anthropic Prompt Caching).
  - **Ngưỡng hòa vốn (Break-even threshold):** $1.25 / (1.0 - 0.10) \approx 1.39$ lần đọc.
- **Đo lường thực nghiệm:**
  - *Kịch bản tái sử dụng cao (Chat/RAG, giả định 5.0 lần đọc/prefix):* `cache_is_worth_it = True`. Chi phí serving đạt **$1.126/1M-token** (chưa tính ghi) và **$1.170/1M-token** (đã tính khấu hao phí ghi cache 1.25x).
  - *Kịch bản tái sử dụng thấp (giả định 1.0 lần đọc/prefix):* `cache_is_worth_it = False`. Hệ thống tự động bỏ qua caching để tránh lãng phí chi phí ghi/lưu trữ ($1.282/1M-token).

### Extension 4: Ngân sách Reasoning (Reasoning Energy & Budget)
- **Phân bổ Lưu lượng vs Tiêu thụ Tài nguyên:**
  - Truy vấn tiêu chuẩn (Standard): 2,199 requests (**91.6% traffic**) | 83.5% tokens | $7.09/ngày (83.5% chi phí) | 1,887.6 Wh/ngày (**6.0% điện năng**)
  - Truy vấn suy luận (Reasoning): 201 requests (**8.4% traffic**) | 16.5% tokens | $1.40/ngày (16.5% chi phí) | 29,787.7 Wh/ngày (**94.0% điện năng**)
- **Insight then chốt:** Mặc dù reasoning chỉ chiếm **8.4% tổng số request**, nhóm này lại ngốn tới **94.0% tổng điện năng suy luận** do quá trình Chain-of-Thought làm tăng số lượng token đầu ra gấp 6 lần và hệ số tiêu thụ điện cao gấp ~80 lần.
- **Kiểm tra trần 10% traffic:** Vì lưu lượng reasoning hiện tại (8.4%) đã ở dưới trần 10%, khoản tiết kiệm trực tiếp từ chính sách cap 10% là **$0.00 / 0.0 Wh**.
- **Chính sách định tuyến thông minh (Dynamic Routing Gate):** Thiết lập bộ lọc chỉ kích hoạt reasoning cho bài toán phức tạp; chuyển 50% truy vấn reasoning không cần thiết sang model chuẩn giúp tiết kiệm tới **~14,708 Wh/ngày**.

### Extension 5: Carbon-Aware Scheduling cho Workload Gián đoạn (Interruptible)
- **Đội tàu gián đoạn:** 5 workload batch training/eval (`interruptible=1`) tiêu thụ tổng cộng **4,227 kWh/tháng**.
- **Bảng phân tích so sánh 5 vùng đám mây:**
  | Vùng đám mây (Region) | Đặc điểm nguồn điện | gCO2/kWh | Giá điện ($/kWh) | Carbon tháng (kg CO2e) | Tiền điện tháng ($) |
  |---|---|---|---|---|---|
  | **europe-north1** (Na Uy) | Thủy điện & Gió (Sạch nhất) | **30** | $0.090 | **126.8 kg** | $380.43 |
  | **us-east-wa** (Washington) | Thủy điện sông Columbia (Rẻ nhất) | 90 | **$0.055** | 380.4 kg | **$232.49** |
  | **us-west-2** (Oregon) | Năng lượng sạch cân bằng | 120 | $0.070 | 507.2 kg | $295.89 |
  | **us-east-1** (Virginia) | Nhiệt điện than/khí (Cơ sở) | 380 | $0.120 | 1,606.3 kg | $507.24 |
  | **europe-central2** (Ba Lan) | Nhiệt điện than (Ô nhiễm nhất) | 660 | $0.180 | 2,789.8 kg | $760.86 |
- **Hiệu quả cắt giảm phát thải:** Di chuyển các job gián đoạn từ `us-east-1` sang `europe-north1` giúp giảm tới **92.1% lượng phát thải carbon** (cắt giảm **1,479.5 kg CO2e/tháng**).
- **Đánh giá đánh đổi độ trễ (Latency Trade-off):** Do các job này là xử lý theo lô offline, không phục vụ truy vấn người dùng cuối theo thời gian thực nên việc điều phối xuyên lục địa hoàn toàn **không gây ảnh hưởng tiêu cực đến SLA độ trễ**.

_Số liệu ghi nhận theo ảnh chụp thực nghiệm tháng 6/2026; lập lại đường cơ sở trước khi hành động._