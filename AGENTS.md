# Hướng dẫn cho AI (dùng chung cho mọi project trong CV/)

## 0. Thứ tự đọc và luật ưu tiên

Đầu mỗi phiên, đọc theo thứ tự:
1. File này.
2. `STRUCTURE.md`.
3. `PROJECT.md` của project đang làm (bài toán, dữ liệu, trạng thái, quyết định, việc tiếp theo).
4. Chạy `git status` và `git log -5 --oneline` để biết AI khác vừa làm gì.

Nếu chưa rõ đang ở project nào: hỏi lại, không đoán.

Khi có mâu thuẫn, ưu tiên: yêu cầu của người dùng trong phiên này > `PROJECT.md` > `AGENTS.md`.
Riêng mục "Không bao giờ" ở phần 1 chỉ được nới khi người dùng nói rõ. Nếu `PROJECT.md` lệch với code hoặc kết quả thực tế: báo cho người dùng, không tự chọn bên nào.

## 1. Quyền chạy lệnh (đọc kỹ nhất)

Quyền chỉ có hiệu lực cho đúng tác vụ người dùng đang giao, không kéo sang tác vụ sau.

**Luôn được (không cần xin):**
- Đọc file, grep, glob, `git status`, `git diff`, `git log`.
- Kiểm tra tĩnh: `python -m py_compile`, lint, `pytest` (test nhẹ, không train).

**Chỉ khi người dùng đồng ý cho tác vụ này (hỏi trước):**
- Smoke test: `python scripts/train.py --model <tên> --epochs 1 --limit 64`, tối đa 1 epoch.
- Cài package (sau khi đã thêm vào `pyproject.toml`).

**Không bao giờ (trừ khi người dùng nói rõ):**
- Train đầy đủ, nhiều seed, evaluate, compare, benchmark, chạy notebook.
- Chạy lại thí nghiệm đã có, hoặc ghi đè `experiments/<tên>_seed<N>/` đã tồn tại.
- Lệnh phá hoại: `rm -rf`, `git reset --hard`, `git push --force`, `git clean`.

Chưa được phép chạy thì chỉ sửa code và ghi rõ "chưa chạy".

## 2. Môi trường

- Conda env: `cv`.
- Mọi lệnh chạy từ **gốc project** (thư mục chứa `PROJECT.md`), không chạy từ trong `src/`.
- Thư viện mới: thêm vào `dependencies` trong `pyproject.toml` rồi mới cài. Không cài lẻ rồi quên ghi.
- Import theo kiểu: `from src.models.builder import build_model`.

## 3. Nguyên tắc cốt lõi

- **Model là cấu hình, không phải bản sao code.** Mỗi model là một thư mục `src/models/<tên>/` gồm `model.py` và `config.yaml`. KHÔNG tạo `train.py` riêng cho từng model.
- **Chỉ có một vòng train**, nằm ở `src/training/`. Mọi model dùng chung.
- **Phần dùng chung** (`src/data/`, `src/training/`, `src/evaluation/`, `src/comparison/`, `src/utils/`) KHÔNG sửa khi chỉ đang thêm model. Nếu nghi có bug ở đó: dừng, báo người dùng, không tự sửa.
- **Tham số đưa vào `config/`**, không hardcode đường dẫn, learning rate, kích thước ảnh trong code.
- **`scripts/` chỉ là vỏ**: parse tham số rồi gọi hàm trong `src/`. Không đặt logic ở đây.
- **Nguồn model mặc định là torchvision** (khớp weights ImageNet trong `PROJECT.md`). Chỉ dùng `timm` khi model không có trong torchvision, và phải ghi lý do vào `PROJECT.md`. Không trộn nguồn weights giữa các model so sánh.

## 4. Quy ước code

- Comment và docstring viết **tiếng Việt**. Tên biến, hàm, file viết tiếng Anh.
- Mọi hàm public phải có type hints và docstring một dòng nói công dụng. Sửa hàm thì cập nhật docstring.
- Không viết tài liệu hàm riêng bằng tay. Code là nguồn sự thật duy nhất.
- Giữ file nhỏ, tên hàm rõ nghĩa.

## 5. Quy trình thêm model mới

1. Tạo `src/models/<tên>/` gồm `__init__.py`, `model.py`, `config.yaml` (theo mẫu model đã có).
2. Đăng ký tên trong `src/models/registry.py`.
3. Chạy kiểm tra tĩnh (được phép luôn).
4. Smoke test chỉ khi được phép (xem phần 1). Chỉ báo "smoke test thành công" khi đã chạy và lệnh thực sự thành công; chưa chạy thì ghi "chưa chạy".
5. Phạm vi được sửa: `src/models/<tên>/`, `src/models/registry.py`, config liên quan, và `PROJECT.md` (mục Trạng thái, Quyết định).

## 6. Train và so sánh (hướng dẫn cho người dùng tự chạy)

- Train: `python scripts/train.py --model <tên> --seed <N>`.
- Mỗi lần chạy ghi vào `experiments/<tên>_seed<N>/` gồm `checkpoints/`, `logs/`, `metrics.json`.
- `metrics.json` tối thiểu có: `model`, `seed`, metric chính, `params_m`, `infer_ms`, `train_min`.
- So sánh: `python scripts/compare.py`. Đọc mọi `metrics.json`, gộp theo model (trung bình các seed), xuất `reports/comparison.csv`, `reports/comparison.md`, `reports/figures/`.
- AI không tự chạy các lệnh này khi chưa được cho phép riêng.

## 7. Luật so sánh công bằng (bắt buộc)

- Mọi model dùng **cùng split** trong `data/splits/` và **cùng recipe** trong `config/train.yaml`. Chỉ đổi model.
- Kết luận so sánh cần **ít nhất 3 seed** (người dùng tự chạy), lấy trung bình ± std.
- Pretrained weights đồng nhất (cùng nguồn) hoặc cùng train từ đầu.
- Báo cả tốc độ suy luận và số tham số, không chỉ độ chính xác. `infer_ms` chỉ so sánh được khi đo cùng máy, cùng device, cùng batch size.
- **Test split chỉ dùng để báo cáo cuối**, không dùng để chọn model hay chỉnh tham số (dùng val).
- Model buộc phải khác recipe: ghi rõ lý do vào `PROJECT.md`, không âm thầm đổi.
- Số liệu ghi vào `PROJECT.md` chỉ được chép từ `reports/` hoặc `metrics.json`. Không tự tính, không nhớ, không bịa. Chưa chạy thì ghi "chưa chạy".

## 8. Không được làm

- Không sửa hay xóa gì trong `data/raw/`.
- Không đổi `data/splits/` khi đang so sánh. Cần split mới thì tạo phiên bản mới (`splits/v2/`).
- Không commit `data/` và `experiments/`.
- Không tạo thư mục `misc/`, `temp/`, `old/`, `new/`, `backup/`.
- Không tự thêm, xóa, đổi tên thư mục cấp 1 hoặc cấp 2 ngoài `STRUCTURE.md`. Cần thư mục mới thì dừng lại và hỏi.

## 9. Git

- AI **được tự commit** sau mỗi thay đổi hoàn chỉnh (một việc = một commit).
- Message tiếng Việt, ngắn, có tiền tố: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`. Ví dụ: `feat: thêm model vit_b32`.
- Chỉ `git add` đúng các file mình sửa. KHÔNG dùng `git add -A` hay `git add .`.
- Không commit khi code còn lỗi cú pháp. Chưa chạy kiểm tra runtime thì vẫn commit được, nhưng ghi rõ trong message hoặc báo cáo cuối phiên.
- KHÔNG `git push`, `--amend`, `rebase`, `reset --hard`, `--force` trừ khi người dùng bảo.
- Không revert hay ghi đè thay đổi mình không tạo ra (có thể của AI khác hoặc của người dùng).

## 10. Phối hợp nhiều AI

- Đầu phiên đọc `git status`, `git diff`, `git log` để biết phần dở của AI trước.
- Mỗi tác vụ có phạm vi file được sửa. Không sửa file ngoài phạm vi; hai AI không sửa cùng một file cùng lúc.
- Vai trò (nếu có phân vai, ví dụ AI A viết code, AI B review) ghi trong `PROJECT.md`.
- Cuối phiên ghi vào `PROJECT.md` mục **Việc tiếp theo** (1-3 dòng cụ thể) và ai đã làm gì, còn dở gì.

## 11. Khi nào hỏi, khi nào làm

**Hỏi trước khi làm** nếu: đụng phần dùng chung; cần thư mục mới; spec mâu thuẫn; việc không đảo ngược được; cần chạy lệnh thuộc nhóm "hỏi trước".
**Tự quyết và ghi lại** cho mọi việc còn lại trong phạm vi tác vụ.

Gặp lỗi: đọc kỹ thông báo, sửa nguyên nhân gốc. Sai hai lần liên tiếp cùng một lỗi thì dừng và báo người dùng, không thử tiếp.

Tác vụ lớn: làm xong **một model chạy thông suốt** trước, rồi mới nhân ra các model còn lại.

## 12. Báo cáo cuối phiên (luôn dùng đúng 3 ý)

1. **Đã sửa:** danh sách file và commit.
2. **Đã chạy:** lệnh nào, kết quả ra sao. Không chạy gì thì ghi "không chạy lệnh nào".
3. **Chưa chạy / chưa kiểm chứng:** liệt kê rõ.

Rồi cập nhật `PROJECT.md`: Trạng thái, Quyết định đã chốt (chỉ ghi *quyết định*, không ghi sự kiện), Việc tiếp theo.