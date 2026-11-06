CV/
└── my_ml_project/
    ├── data/
    │   ├── raw/                 # dữ liệu gốc, KHÔNG ĐƯỢC SỬA
    │   ├── processed/           # đã xử lý, sẵn sàng train
    │   ├── external/            # dữ liệu / pretrained weights bên ngoài
    │   └── splits/              # danh sách train/val/test CỐ ĐỊNH, mọi model dùng chung
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   └── 02_compare_results.ipynb
    │
    ├── src/
    │   ├── data/                # dataset, transforms, dataloader (DÙNG CHUNG)
    │   ├── models/              # MỖI MODEL MỘT THƯ MỤC CON
    │   │   ├── resnet50/
    │   │   │   ├── model.py     # định nghĩa / bọc model
    │   │   │   └── config.yaml  # cấu hình riêng của model này
    │   │   ├── vit_b16/
    │   │   ├── convnext_t/
    │   │   ├── registry.py      # tên model -> class
    │   │   └── builder.py       # đọc config, tạo model
    │   ├── training/            # vòng train, optimizer, callback (DÙNG CHUNG)
    │   ├── evaluation/          # metrics, evaluator (DÙNG CHUNG)
    │   ├── comparison/          # gom kết quả và so sánh các model
    │   │   ├── collect.py       # đọc mọi metrics.json
    │   │   └── report.py        # xuất bảng và biểu đồ
    │   └── utils/               # seed, logging, đọc config, I/O
    │
    ├── scripts/
    │   ├── train.py             # python scripts/train.py --model resnet50
    │   ├── evaluate.py
    │   └── compare.py           # chạy so sánh toàn bộ
    │
    ├── config/
    │   ├── data.yaml            # đường dẫn, kích thước ảnh, augmentation
    │   └── train.yaml           # recipe train CHUNG (lr, epoch, batch, seed)
    │
    ├── experiments/             # KHÔNG commit, thay cho thư mục models/ cũ
    │   ├── resnet50_seed42/
    │   │   ├── checkpoints/
    │   │   ├── logs/
    │   │   └── metrics.json
    │   └── vit_b16_seed42/
    │
    ├── reports/                 # kết quả chốt, commit được
    │   ├── comparison.csv
    │   ├── comparison.md
    │   └── figures/
    │
    ├── tests/
    ├── .gitignore
    ├── README.md
    ├── pyproject.toml
    └── Dockerfile               # tùy chọn