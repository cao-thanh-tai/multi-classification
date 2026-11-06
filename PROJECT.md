# {{PROJECT_NAME}}

> File ngữ cảnh của project. AI đọc đầu mỗi phiên và cập nhật khi xong một mốc.
> Quy tắc: file này chỉ chứa trạng thái HIỆN TẠI. Số liệu chi tiết nằm ở `reports/`, không chép tay vào đây.

## Bài toán
- Loại task: multi classification
- Đầu vào: img
- Đầu ra: labels
- Mô tả ngắn: xác định các vật thể trong classes có trong img

## Dữ liệu
- Nguồn: kaggle
- Đường dẫn: `data/raw/` (chỉ đọc)
- Định dạng nhãn: PASCAL VOC XML, chuyển thành nhãn image-level multi-label khi chạy EDA/dataset code
- Danh sách lớp: EDA tự trích xuất từ annotation và ghi vào `data/processed/classes.json`
- Số lượng: (số ảnh, số lớp; thêm số object nếu detection/segmentation)
- Cách chia: dùng nguyên các split chính thức VOC2007 `train.txt`, `val.txt`, `test.txt`; notebook sao chép sang `data/splits/v1/`, không random và không có seed
- Cấu trúc thư mục: (dán cây thư mục raw nếu cần)
┣ 📂PASCAL_VOC
┃ ┗ 📂PASCAL_VOC
┃   ┣ 📜pascal_test2007.json
┃   ┣ 📜pascal_train2007.json
┃   ┣ 📜pascal_train2012.json
┃   ┣ 📜pascal_val2007.json
┃   ┗ 📜pascal_val2012.json
┣ 📂VOCtest_06-Nov-2007
┃ ┗ 📂VOCdevkit
┃   ┗ 📂VOC2007
┃     ┣ 📂Annotations
┃     ┣ 📂ImageSets
┃     ┃ ┣ 📂Layout
┃     ┃ ┃ ┗ 📜test.txt
┃     ┃ ┣ 📂Main
┃     ┃ ┃ ┣ 📜aeroplane_test.txt
┃     ┃ ┃ ┣ 📜bicycle_test.txt
┃     ┃ ┃ ┣ 📜bird_test.txt
┃     ┃ ┃ ┣ 📜boat_test.txt
┃     ┃ ┃ ┣ 📜bottle_test.txt
┃     ┃ ┃ ┣ 📜bus_test.txt
┃     ┃ ┃ ┣ 📜car_test.txt
┃     ┃ ┃ ┣ 📜cat_test.txt
┃     ┃ ┃ ┣ 📜chair_test.txt
┃     ┃ ┃ ┣ 📜cow_test.txt
┃     ┃ ┃ ┣ 📜diningtable_test.txt
┃     ┃ ┃ ┣ 📜dog_test.txt
┃     ┃ ┃ ┣ 📜horse_test.txt
┃     ┃ ┃ ┣ 📜motorbike_test.txt
┃     ┃ ┃ ┣ 📜person_test.txt
┃     ┃ ┃ ┣ 📜pottedplant_test.txt
┃     ┃ ┃ ┣ 📜sheep_test.txt
┃     ┃ ┃ ┣ 📜sofa_test.txt
┃     ┃ ┃ ┣ 📜test.txt
┃     ┃ ┃ ┣ 📜train_test.txt
┃     ┃ ┃ ┗ 📜tvmonitor_test.txt
┃     ┃ ┗ 📂Segmentation
┃     ┃   ┗ 📜test.txt
┃     ┣ 📂JPEGImages
┃     ┣ 📂SegmentationClass
┃     ┗ 📂SegmentationObject
┗ 📂VOCtrainval_06-Nov-2007
  ┗ 📂VOCdevkit
    ┗ 📂VOC2007
      ┣ 📂Annotations
      ┣ 📂ImageSets
      ┃ ┣ 📂Layout
      ┃ ┃ ┣ 📜train.txt
      ┃ ┃ ┣ 📜trainval.txt
      ┃ ┃ ┗ 📜val.txt
      ┃ ┣ 📂Main
      ┃ ┃ ┣ 📜aeroplane_train.txt
      ┃ ┃ ┣ 📜aeroplane_trainval.txt
      ┃ ┃ ┣ 📜aeroplane_val.txt
      ┃ ┃ ┣ 📜bicycle_train.txt
      ┃ ┃ ┣ 📜bicycle_trainval.txt
      ┃ ┃ ┣ 📜bicycle_val.txt
      ┃ ┃ ┣ 📜bird_train.txt
      ┃ ┃ ┣ 📜bird_trainval.txt
      ┃ ┃ ┣ 📜bird_val.txt
      ┃ ┃ ┣ 📜boat_train.txt
      ┃ ┃ ┣ 📜boat_trainval.txt
      ┃ ┃ ┣ 📜boat_val.txt
      ┃ ┃ ┣ 📜bottle_train.txt
      ┃ ┃ ┣ 📜bottle_trainval.txt
      ┃ ┃ ┣ 📜bottle_val.txt
      ┃ ┃ ┣ 📜bus_train.txt
      ┃ ┃ ┣ 📜bus_trainval.txt
      ┃ ┃ ┣ 📜bus_val.txt
      ┃ ┃ ┣ 📜car_train.txt
      ┃ ┃ ┣ 📜car_trainval.txt
      ┃ ┃ ┣ 📜car_val.txt
      ┃ ┃ ┣ 📜cat_train.txt
      ┃ ┃ ┣ 📜cat_trainval.txt
      ┃ ┃ ┣ 📜cat_val.txt
      ┃ ┃ ┣ 📜chair_train.txt
      ┃ ┃ ┣ 📜chair_trainval.txt
      ┃ ┃ ┣ 📜chair_val.txt
      ┃ ┃ ┣ 📜cow_train.txt
      ┃ ┃ ┣ 📜cow_trainval.txt
      ┃ ┃ ┣ 📜cow_val.txt
      ┃ ┃ ┣ 📜diningtable_train.txt
      ┃ ┃ ┣ 📜diningtable_trainval.txt
      ┃ ┃ ┣ 📜diningtable_val.txt
      ┃ ┃ ┣ 📜dog_train.txt
      ┃ ┃ ┣ 📜dog_trainval.txt
      ┃ ┃ ┣ 📜dog_val.txt
      ┃ ┃ ┣ 📜horse_train.txt
      ┃ ┃ ┣ 📜horse_trainval.txt
      ┃ ┃ ┣ 📜horse_val.txt
      ┃ ┃ ┣ 📜motorbike_train.txt
      ┃ ┃ ┣ 📜motorbike_trainval.txt
      ┃ ┃ ┣ 📜motorbike_val.txt
      ┃ ┃ ┣ 📜person_train.txt
      ┃ ┃ ┣ 📜person_trainval.txt
      ┃ ┃ ┣ 📜person_val.txt
      ┃ ┃ ┣ 📜pottedplant_train.txt
      ┃ ┃ ┣ 📜pottedplant_trainval.txt
      ┃ ┃ ┣ 📜pottedplant_val.txt
      ┃ ┃ ┣ 📜sheep_train.txt
      ┃ ┃ ┣ 📜sheep_trainval.txt
      ┃ ┃ ┣ 📜sheep_val.txt
      ┃ ┃ ┣ 📜sofa_train.txt
      ┃ ┃ ┣ 📜sofa_trainval.txt
      ┃ ┃ ┣ 📜sofa_val.txt
      ┃ ┃ ┣ 📜train_train.txt
      ┃ ┃ ┣ 📜train_trainval.txt
      ┃ ┃ ┣ 📜train_val.txt
      ┃ ┃ ┣ 📜train.txt
      ┃ ┃ ┣ 📜trainval.txt
      ┃ ┃ ┣ 📜tvmonitor_train.txt
      ┃ ┃ ┣ 📜tvmonitor_trainval.txt
      ┃ ┃ ┣ 📜tvmonitor_val.txt
      ┃ ┃ ┗ 📜val.txt
      ┃ ┗ 📂Segmentation
      ┃   ┣ 📜train.txt
      ┃   ┣ 📜trainval.txt
      ┃   ┗ 📜val.txt
      ┣ 📂JPEGImages
      ┣ 📂SegmentationClass
      ┗ 📂SegmentationObject
- Chỉ đọc ảnh và annotation từ hai bộ VOC2007 `VOCtrainval_06-Nov-2007` và `VOCtest_06-Nov-2007`
- Ghi chú đặc biệt: (mất cân bằng lớp, nhãn không đồng nhất, ảnh trùng, ảnh lỗi, nhiễu...)

## Mục tiêu
- So sánh các model: {{MODELS}}
- Metric chính: (classification: F1, accuracy; detection: mAP@0.5, mAP@0.5:0.95; segmentation: mIoU, Dice)
- Metric phụ: tốc độ suy luận, số tham số, thời gian train
- Thiết bị đo tốc độ: (CPU/GPU, batch size, kích thước đầu vào; phải giống nhau cho mọi model)
- Điều kiện thành công: (cụ thể, kiểm tra được. Ví dụ: F1 val >= 0.90 và infer_ms <= 30 trên thiết bị trên)

## Cài đặt thí nghiệm (recipe chung)
- Kích thước ảnh:
- Augmentation:
- Optimizer / learning rate / epoch / batch size:
- Seed sẽ chạy: 42, 43, 44
- Pretrained weights: (nguồn + loại weights, dùng đồng nhất cho mọi model)

## Trạng thái
> Thêm mục khi cần. Mốc đã xong lâu thì gộp lại thành một dòng để file không dài ra.
- [ ] Điền Bài toán và Dữ liệu
- [ ] EDA (đã chạy bằng dataset trong conda env `cv`; chưa chốt kết luận)
- [ ] Chốt split (`data/splits/v1/`, code sao chép split chính thức đã viết, chưa chạy)
- [ ] Dataset multi-label (đã viết code và test, chưa chạy được pytest vì môi trường thiếu pytest)
- [ ] Pipeline chạy thông suốt với một model (baseline)
- [ ] Thêm các model còn lại
- [ ] Chạy đủ 3 seed cho mỗi model
- [ ] Chạy `compare.py`, chốt bảng so sánh trong `reports/`
- [ ] Đánh giá trên test split (chỉ để báo cáo cuối)

## Việc tiếp theo
- Rà soát biểu đồ và chốt kết luận EDA trong `notebooks/01_eda.ipynb`.
- Chạy cell lưu split để tạo `data/splits/v1/train.txt`, `val.txt`, `test.txt`.
- Tạo `classes.json` và split bằng notebook, sau đó kiểm tra dataset bằng pytest.

## Quyết định đã chốt
> Chỉ ghi quyết định kèm ngày và lý do (vì sao chọn A, không chọn B). Sự kiện như "đã train xong" thì đi vào Trạng thái, không ghi ở đây.
- (2026-11-06, sinh danh sách class từ annotation XML trong EDA, tránh duy trì thủ công một file dữ liệu dẫn xuất)
- (2026-11-06, dùng nguyên split chính thức VOC2007 và chỉ sao chép sang `data/splits/v1/`, tránh random split mới làm mất tính tái lập)
- (2026-11-06, dataset đọc trực tiếp split, XML và JPEG trong raw, tránh tạo bản sao ảnh hoặc metadata dẫn xuất không cần thiết)

## Kết quả tóm tắt
- Báo cáo: `reports/comparison.md`, `reports/comparison.csv`, `reports/figures/`
- Kết luận chính: (2-3 dòng, số chỉ lấy từ `reports/`; chưa chạy thì ghi "chưa chạy")

## Vấn đề đang mở
(Lỗi chưa giải quyết, câu hỏi cần người dùng quyết định.)