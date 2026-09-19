# Fine-tuning container runbook

## Boundary

本 repo 提供資料與設定；實際訓練在群聯提供的 aiDAPTIVLink 2.0 container 中，以
群聯的訓練腳本執行。

## Before the first run

1. 確認 container image、aiDAPTIVLink 版本與 `phisonai2` 可用。
2. 將 `finetune/configs/` 與所選資料集掛載進 container。
3. 將 `finetune/runs/` 掛載為 container 的訓練輸出與 log 位置。
4. 在 `env_config.yaml` 中填寫模型、資料、NVMe 與輸出目錄的 container path。
5. 確認 `per_update_total_batch_size` 是 `num_gpus * per_device_train_batch_size`
   的倍數。

實際 Docker command、container 內工作目錄、模型路徑與 aiDAPTIVCache 掛載點尚未
由本 repo 證實，應在第一次 container characterization 後補入，不應直接套用教材
中的使用者名稱或主機路徑。
