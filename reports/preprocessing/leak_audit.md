# Kiểm tra rò rỉ tỉ lệ của prefix và ý nghĩa của mốc 100%

Tái lập:

- `.venv/Scripts/python scripts/leak_audit.py` → `leak_audit.json` (100 lớp, 40.000 hình val).
- `.venv/Scripts/python -I scripts/raw_pilot.py <thư_mục_raw> triangle bat` → `raw_pilot.json`
  (40.000 hình thuộc train và val của 2 lớp; file raw lấy từ
  `https://storage.googleapis.com/quickdraw_dataset/full/raw/{class}.ndjson`).

Tập test không được dùng trong bất kỳ phép đo nào.

## 1. Rò rỉ là gì

Dữ liệu simplified được scale và simplify theo **hình đầy đủ**. Khi cắt prefix rồi phóng to
theo khung của chính nó, ba dấu vết còn lại vẫn cho biết prefix nhỏ cỡ nào so với hình đầy đủ:

1. **Lưới toạ độ:** toạ độ gốc là số nguyên, nên sau khi phóng to chúng nằm trên một lưới có
   bước `255 / cạnh thật`.
2. **Mật độ điểm:** khoảng cách giữa các điểm giãn ra theo hệ số phóng to.
3. **Cạnh gãy:** RDP với ε = 2 tính ở tỉ lệ hình đầy đủ, nên khi phóng to đường cong trông
   gãy rõ hơn. Dấu vết này nằm trong hình học, kể cả trên raster.

Mục tiêu đo: `log(cạnh dài prefix / cạnh dài hình đầy đủ)`, prefix cắt ngẫu nhiên 5–100% lượng
mực. Một phần tỉ lệ này suy ra được hợp lệ từ hình dạng, nên điểm số cao chưa phải bằng chứng.
Phải so các nhánh trên **cùng hình vẽ và cùng mức cắt**.

## 2. Nhánh tham chiếu sạch

Pilot lấy nét raw, cắt prefix **trên nét raw**, rồi mới simplify riêng prefix đó. Nhánh này
không chứa thông tin về hình đầy đủ. Các kiểm tra trên nhánh này:

| Kiểm tra | triangle | bat |
|---|---|---|
| Hình của nhóm tìm thấy trong raw theo `key_id` | 100% | 100% |
| Cờ `recognized` khớp | 100% | 100% |
| `simplify` của nhóm trùng khít dữ liệu đã lưu (2.000 hình) | 97,5% | 95,2% |
| Khoảng cách Chamfer p99 so với dữ liệu đã lưu (đơn vị canvas) | 0,15 | 0,32 |

## 3. Kết quả

### Tấn công lưới toạ độ

Tỉ lệ ước lượng đúng trong sai số 3%. **Mức nền:** luôn đoán tỉ lệ = 1 đã đúng ở khoảng 34% số
hình, vì nhiều prefix đã phủ hết khung hình đầy đủ. Vì vậy con số 90% khi đứng riêng là phóng đại.

| Nhánh | 100 lớp (val) | Pilot 2 lớp |
|---|---|---|
| Mức nền (luôn đoán 1) | 33,7% | 34,1% |
| Chuỗi hiện tại, mọi điểm | **90,2%** | **84,7%** |
| Chuỗi hiện tại, chỉ đầu/cuối nét | 48,3% | 34,5% |
| Sạch từ raw, mọi điểm | – | 33,2% |
| Sau `resample_uniform`, mọi điểm / đầu cuối nét | 0,01% / 0,09% | 0,01% / 0,13% |

Xét riêng các hình có tỉ lệ < 0,9 (56% số hình, 100 lớp), nơi đoán 1 không ăn thua: tấn công
đúng 87,4% trên chuỗi hiện tại, 0,1% sau khi lấy mẫu lại.

Cách lấy mẫu lại **giữ nguyên điểm đầu/cuối nét vẫn lộ**: tấn công chỉ dùng các điểm này (được
`pen_lift` đánh dấu) đúng 48,6% số hình và 90,7% số hình nhiều nét (đo một lần trên 6.000 hình
val bằng script tạm, không nằm trong repo). Vì vậy `resample_uniform` lấy điểm giữa các cung và không giữ điểm gốc nào.

### Dự đoán tỉ lệ bằng model (R²)

Cùng hình vẽ, cùng mức cắt. Gradient boosting trên đặc trưng chuỗi (5 fold), CNN nhỏ trên raster
64×64 (3 seed, độ lệch ≤ 0,005).

| Nhánh | 100 lớp (val) | Pilot 2 lớp | Phần vượt so với nhánh sạch cùng loại (pilot) |
|---|---|---|---|
| Chuỗi hiện tại | 0,754 | 0,789 | **+0,24** |
| Chuỗi `resample_uniform` | 0,529 | 0,620 | **+0,10** |
| Raster 64×64 hiện tại | 0,501 | 0,617 | **+0,08** |
| Chuỗi sạch từ raw | – | 0,547 | tham chiếu |
| Chuỗi sạch + lấy mẫu lại | – | 0,517 | tham chiếu |
| Raster 64×64 sạch từ raw | – | 0,537 | tham chiếu |

Theo từng lớp, chuỗi hiện tại / lấy mẫu lại / sạch / sạch + lấy mẫu lại:
triangle 0,736 / 0,683 / 0,595 / 0,573; bat 0,865 / 0,715 / 0,627 / 0,609.

### Kết luận về rò rỉ

- **Có rò rỉ thật**, và chuỗi hiện tại lộ nhiều nhất.
- Khoảng **0,52–0,55** R² là thông tin suy ra được từ hình dạng, vì nhánh sạch vẫn đạt mức đó.
- `resample_uniform` **loại được lưới và mật độ điểm**, giảm phần vượt từ 0,24 xuống 0,10, nhưng
  **không hết**: phần còn lại phù hợp với dấu vết cạnh gãy.
- **Raster 64×64 cũng lộ** (+0,08), ở mức tương đương chuỗi đã lấy mẫu lại.
- Chỉ cắt trên nét raw rồi simplify mới loại hết được.

### Giới hạn của phép đo

- Nhánh sạch chỉ đo trên **2 lớp**, chọn vì khác nhau về độ khó (recognized 97,8% và 81,4%). Số
  của 100 lớp không có nhánh sạch để so.
- R² đo khả năng suy ra tỉ lệ, **không đo mức tăng ảo độ chính xác phân loại**. Câu hỏi đó chưa
  được trả lời. Một classifier nông được đưa thẳng tỉ lệ thật chỉ tăng khoảng 0,6–0,9 điểm (đo
  một lần bằng script tạm, không nằm trong repo), nhưng không nói được gì về model sâu.
- Đặc trưng do nhóm tự chọn, nên R² là cận dưới của những gì một model mạnh hơn có thể đọc được.
- Nhánh sạch cắt theo mực của nét raw, còn nhánh hiện tại theo mực của nét đã simplify, nên hai
  nhánh không cắt tại đúng cùng một điểm. Mức nền của hai nhánh gần bằng nhau (34,1% và 34,2%).

## 4. Ý nghĩa của mốc 100%

Đo trên toàn bộ file raw của 2 lớp, trừ các hàng thuộc tập test của nhóm. Thời lượng là thời
điểm của điểm cuối cùng được vẽ.

| | triangle, recognized | triangle, không | bat, recognized | bat, không |
|---|---|---|---|---|
| Số giây (median) | 1,9 | 15,7 | 9,1 | 16,6 |
| Số giây (p90) | 3,7 | 19,1 | 14,9 | 19,1 |
| Tỉ lệ dưới 10 giây | 99,7% | 29,5% | 58,4% | 25,1% |
| Tỉ lệ trong khoảng 15–20 giây | 0,1% | 50,4% | 9,5% | 56,7% |
| Tỉ lệ từ 20 giây trở lên | 0,0% | 2,7% | 0,3% | 2,9% |

Hình: `figures/raw_duration_by_recognized.png`.

- **Phù hợp với việc lượt chơi dừng khi được nhận ra:** hình recognized ngắn hơn hẳn, nhất là
  triangle. Nhưng **thời điểm nhận ra không được ghi trong dữ liệu**, nên đây vẫn là bằng chứng
  gián tiếp, và mới trên 2 lớp.
- **Có giới hạn khoảng 20 giây:** phân phối của hình không được nhận ra dồn lên ngay trước giây 20
  rồi giảm mạnh. Giới hạn này không tuyệt đối theo điểm cuối, vì 2,7–2,9% vượt qua giây 20.
- **Hình không được nhận ra không đồng nhất:** khoảng một nửa kết thúc trong 15–20 giây, nhưng
  25–30% kết thúc trước giây 10. Vì vậy câu "tiếp tục thêm nét cho tới khi hết 20 giây" trong bản
  EDA trước là **quá tuyệt đối**: có cả nhóm dừng sớm mà dữ liệu không cho biết lý do.
- **Mực không tương đương thời gian:** tương quan giữa mức cắt theo mực và phần thời gian đã trôi
  qua chỉ là 0,55. Median phần thời gian là 0,11 / 0,34 / 0,57 / 0,84 ở các khoảng mực 5–25 /
  25–50 / 50–75 / 75–100%. Khoảng mực 5–25% tương ứng với chỉ khoảng 11% thời gian.

Hệ quả cho báo cáo:

- 100% nghĩa là "lúc lượt vẽ dừng", không phải "vẽ xong".
- Luôn báo cáo tách theo `recognized` ở mọi mức prefix.
- Mốc thời điểm dừng của model Google chỉ là tham khảo: model đó chọn trong 345 lớp, phiên bản và
  độ trễ không rõ, không chạy lại được, và không có giá trị cho hình không được nhận ra.

## 5. Cách dùng pipeline hiện tại

| Mục đích | Dùng | Lưu ý |
|---|---|---|
| Hình đầy đủ | `RasterView` / `SequenceView` với `frac=None` | Không cắt prefix nên không có rò rỉ tỉ lệ |
| Nhận diện sớm trên chuỗi | `SequenceView(..., resample_step=RESAMPLE_STEP)` | Không dùng chuỗi chưa lấy mẫu lại làm kết quả chính; vẫn còn phần vượt khoảng +0,10 |
| Nhận diện sớm trên raster | `RasterView` | Vẫn còn phần vượt khoảng +0,08 |
| Kết luận chắc chắn về độ chính xác sớm | Cần nhánh sạch từ raw | Khoảng 53 GiB cho 99/100 lớp đã kiểm tra dung lượng; có thể chạy trên một tập lớp con |

Lấy mẫu lại có đánh đổi: IoU giữa raster trước và sau lấy mẫu lại có median 0,87 (p10 0,82) với
nét mảnh 2 px. Với bước 8 đơn vị, 98,3% prefix nằm trong 256 điểm (đo một lần trên 6.000 hình val
bằng script tạm); phần còn lại được giảm mẫu thêm.

## 6. Việc chưa làm

- Thí nghiệm đo tác hại: cùng model, cùng budget, 3 seed, so chuỗi hiện tại / chuỗi lấy mẫu lại /
  nhánh sạch ở từng mức prefix trên val, trên một tập lớp có file raw.
- Nhóm chốt định nghĩa prefix. Pipeline mặc định cắt theo mực; lúc chọn lớp cắt theo số nét
  (`max(1, ceil(0,25 · số nét))`). Dữ liệu raw còn cho phép cắt theo thời gian thật.
