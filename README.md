# cnn-collab：MNIST 簡單 CNN 協作練習專案

這是一個用來練習 **Git / GitHub 團隊協作** 的極簡 CNN 專案。
模型在 MNIST 手寫數字資料集上做 10 類分類。

## 專案結構

```
cnn-collab/
├── README.md          # 你正在看的檔案
├── model.py           # MinimalCNN 模型定義
├── data.py            # MNIST 載入與前處理
├── train.py           # 訓練主程式（含功能註冊區）
├── utils.py           # 小工具
├── sandbox.py         # 還原練習用沙盒（可放心亂改）
├── requirements.txt   # 套件需求
└── docs/教學講義.md    # 協作練習完整步驟
```

## 安裝與執行

```bash
pip install -r requirements.txt
python train.py
```

