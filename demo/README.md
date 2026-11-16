# Demo webapp: hướng dẫn cho người làm giao diện

Phần backend đã có sẵn: model, tiền xử lý, ngưỡng dừng và cây phân cấp đều giống hệt bài báo.
Bạn chỉ cần làm giao diện (vẽ trên canvas, gọi API, hiển thị). **Đừng tự viết lại phần tiền xử lý
hay ngưỡng**: sai một bước là model đoán sai mà không báo lỗi.

```
demo/
  server.py          API (FastAPI): POST /guess, GET /info, GET / (trang mẫu)
  static/index.html  trang mẫu: canvas + gọi API, dùng làm tham chiếu
  assets.json        100 tên lớp, cây phân cấp, ngưỡng đã hiệu chỉnh (sinh bởi scripts/export_demo_assets.py)
  requirements.txt   thư viện cho demo (chạy CPU là đủ)
src/quickdraw_exp/demo.py   Guesser: nét vẽ -> dự đoán (dùng chung code với bài báo)
```

## 1. Chạy thử trên máy (khoảng 10 phút)

1. Clone repo, nhánh `pipeline-step1`:
   ```bash
   git clone -b pipeline-step1 https://github.com/Trung4n/AI-dictionary.git
   cd AI-dictionary
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r demo/requirements.txt
   ```
2. Tải gói checkpoint (Kaggle, công khai, CC BY 4.0): https://www.kaggle.com/datasets/kittrnmuhaha/ai-pictionary-checkpoints-ds (nút Download, hoặc
   `kaggle datasets download -d kittrnmuhaha/ai-pictionary-checkpoints-ds --unzip`). Giải nén, rồi chọn **một** run, ví dụ
   `package/runs/cnn_imagenet_s42` (chính xác nhất, khoảng 45 MB) hoặc `package/runs/cnn_prefix_s42`
   (nhẹ hơn, khoảng 11 MB). Mỗi run gồm `config.yaml` và `checkpoints/best.pt`.
3. Chạy server:
   ```bash
   # Linux/macOS
   DEMO_RUN=<đường dẫn>/package/runs/cnn_imagenet_s42 PYTHONPATH=src uvicorn demo.server:app --port 8000
   # Windows PowerShell
   $env:DEMO_RUN="<đường dẫn>\package\runs\cnn_imagenet_s42"; $env:PYTHONPATH="src"; uvicorn demo.server:app --port 8000
   ```
4. Mở http://localhost:8000, vẽ thử. Kiểm tra code: `PYTHONPATH=src python -m pytest tests/test_demo.py`.

## 2. API

**POST `/guess`**: gửi **toàn bộ** các nét đã vẽ từ đầu (không chỉ nét mới nhất), theo thứ tự vẽ,
toạ độ pixel của canvas (gốc ở góc trên trái, y hướng xuống). Không cần chuẩn hoá gì phía trình duyệt.

```json
{"strokes": [[[x0, x1, x2], [y0, y1, y2]], [[x0, x1], [y0, y1]]]}
```

Trả về:

```json
{
  "ready": true,
  "run": "cnn_imagenet_s42",
  "strokes": 2,
  "top": [{"class": "horse", "probability": 0.62}, {"class": "zebra", "probability": 0.21}],
  "should_answer": false,
  "answer": null,
  "stop_threshold": 0.847,
  "group_answer": {"name": "animal/mammal", "level": "subgroup", "probability": 0.91}
}
```

- `should_answer = true`: model đủ chắc, `answer` là tên lớp. Ngưỡng `stop_threshold` được hiệu
  chỉnh trên tập validation để việc trả lời sớm làm mất tối đa 5% độ chính xác (mục 3.4 của bài).
- Nếu chưa đủ chắc nhưng `group_answer` khác `null`: model chắc về một **nhóm** (ví dụ
  `animal/mammal` → "một loài thú", `weather and nature` → "thời tiết, thiên nhiên"). Phần sau dấu `/` là nhóm con.
- Cả hai đều không có: hiển thị "chưa chắc, vẽ tiếp".
- `ready = false`: chưa có nét nào.

**GET `/info`**: tên run, danh sách 100 lớp (dùng để gợi ý người chơi vẽ gì), các ngưỡng.

## 3. Lưu ý khi làm giao diện

- **Gọi `/guess` khi kết thúc mỗi nét** (sự kiện nhả chuột/nhấc tay), như trang mẫu. Gọi liên tục
  khi đang vẽ là không cần thiết. Nếu muốn mượt hơn, gọi thêm khi nét dài, nhưng tối đa vài lần mỗi giây.
- Nếu một yêu cầu chưa xong mà người dùng đã vẽ thêm, chỉ cần gửi yêu cầu mới nhất (trang mẫu đã làm vậy).
- Model chỉ biết **100 lớp** trong `/info`. Vẽ thứ khác thì model vẫn đoán một trong 100 lớp đó.
- Model học từ hình của game Quick, Draw! (vẽ trong 20 giây, phần lớn bằng chuột). Vẽ quá chi tiết
  hoặc tô kín có thể làm model đoán kém hơn.
- Mỗi lần đoán chạy trên CPU, thường dưới một giây (chưa đo kỹ; hãy đo trên máy chủ thật trước buổi demo).
- Server từ chối hình có hơn 5.000 điểm (lỗi 413).
- Nếu giao diện chạy ở tên miền khác server, đặt `DEMO_ORIGINS=https://<tên miền giao diện>` khi chạy server (CORS).
  Trang mẫu nhận địa chỉ API qua `?api=http://...`.

## 4. Đổi model

Đặt `DEMO_RUN` sang run khác trong gói checkpoint. Run phải có ngưỡng trong `demo/assets.json`:
các run CNN và ResNet-18 (`cnn_prefix_*`, `cnn_imagenet_*`, `cnn_resnet18_scratch_*`, `cnn_anytime_*`)
và GRU lấy mẫu lại (`gru_uniform_*`, `gru_uniform_full_*`, `gru_anytime_uniform*`). Cây phân cấp chỉ có
ngưỡng cho `cnn_prefix_*`, `cnn_imagenet_*`, `gru_uniform_*` và `gru_uniform_full_*`; với run khác,
`group_answer` luôn là `null`.

## 5. Đưa lên mạng (tuỳ chọn)

Chạy CPU là đủ, ví dụ Hugging Face Spaces (Docker) hoặc Render: cài `demo/requirements.txt`, chép run
đã chọn vào image, đặt `DEMO_RUN` và `PYTHONPATH=src`, lệnh chạy `uvicorn demo.server:app --host 0.0.0.0 --port <port>`.

## 6. Không được làm

- Không sửa `src/quickdraw/` và `src/quickdraw_exp/` (trừ khi trao đổi với Kiet): đây là code của các kết quả trong báo cáo.
- Không dùng demo để báo số liệu: mọi số trong bài lấy từ `reports/`.

Dữ liệu Quick, Draw! © Google, giấy phép CC BY 4.0. Hãy ghi nguồn này trên trang demo.
