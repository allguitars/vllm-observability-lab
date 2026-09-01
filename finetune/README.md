# Fine-tuning external workspace

這個目錄管理 aiDAPTIVLink 2.0 fine-tuning 所需的設定、資料與驗證產物。
實際訓練仍在群聯提供的容器內，以群聯的 `phisonai2` 與訓練腳本執行；本 repo
不複製、不修改該容器中的程式。

## 目錄用途

```text
configs/       可追蹤的 dataset、environment、experiment YAML 範本
datasets/      訓練資料；fixture 可追蹤，local 會被 Git 忽略
toolkit/       已下載的 aiDAPTIV Toolkit，或其版本與來源說明
runs/          每次容器訓練掛回 host 的原始 log 與輸出，會被 Git 忽略
scripts/       未來容器外輔助工具的預留位置
```

## 使用順序

1. 以 `datasets/fixtures/qa-smoke.json` 驗證資料格式與 container mount。
2. 複製並填寫 `configs/dataset/qa_dataset_config.example.yaml` 的 `data_path`。
3. 填寫 `configs/env/env_config.example.yaml` 的模型、資料、NVMe 與輸出路徑。
4. 依 GPU 數量調整 `configs/exp/exp_config.example.yaml`，再由群聯容器執行。

所有 YAML 中的路徑必須填寫 **容器內路徑**，不是此 repo 在 host 上的路徑。容器
掛載方式與執行命令記錄於 `docs/finetune-container-runbook.md`。
