# Fine-tuning external workspace

這個目錄管理 aiDAPTIVLink 2.0 fine-tuning 所需的設定、資料與驗證產物。
實際訓練仍在群聯提供的容器內，以群聯的 `phisonai2` 與訓練腳本執行；本 repo
不複製、不修改該容器中的程式。

## 目錄用途

```text
configs/       可追蹤的 dataset、environment、experiment YAML 範本
datasets/      訓練資料；fixture 可追蹤，local 會被 Git 忽略
toolkit/       已下載的 aiDAPTIV Toolkit，或其版本與來源說明
runs/          每次容器訓練掛回 host 的原始 log 與輸出，可納入 Git 保存證據
scripts/       container 內執行的訓練 wrapper 與每輪 run 目錄設定
```

## 訓練前設定

所有 YAML 路徑都必須是 **container 內路徑**，不是 host 上的 repo 路徑。

1. 以 `.example.yaml` 建立三個實際設定檔：

   ```bash
   cp configs/dataset/qa_dataset_config.example.yaml configs/dataset/qa_dataset_config.yaml
   cp configs/env/env_config.example.yaml configs/env/env_config.yaml
   cp configs/exp/exp_config.example.yaml configs/exp/exp_config.yaml
   cp scripts/.env.example scripts/.env
   ```

2. 設定 `configs/dataset/qa_dataset_config.yaml`。

   - `data_path` 指向原始 JSON，例如 `/workspace/finetune/datasets/fixtures/qa-smoke.json`。
   - `question_key`、`answer_key`、`label_key` 必須與 JSON 欄位名稱一致。
   - QA 範例使用 `instruct` 與 `output`；`label_key` 應與 `answer_key` 相同。

3. 設定 `configs/env/env_config.yaml`。

   - `model_name_or_path`：container 內模型目錄，例如 `/app/gemma-3-12b-it`。
   - `train_data_path`：dataset-config YAML 的 container 路徑。
   - `nvme_path`：aiDAPTIVCache 掛載點，例如 `/mnt/nvme0`。
   - `output_dir`：本輪微調模型的輸出目錄。
   - `log_name`：群聯訓練 log 的完整輸出路徑。

4. 設定 `configs/exp/exp_config.yaml`。

   - `num_gpus`、`specify_gpus` 必須符合 container 實際可見的 GPU。
   - `per_update_total_batch_size` 必須是 `num_gpus * per_device_train_batch_size` 的整數倍。
   - 不用 LoRA 時維持 `lora.enable_lora: false`；使用 LoRA 時再填寫相關設定與輸出路徑。

5. 設定 `scripts/.env` 的本輪 run 目錄，例如：

   ```bash
   RUN_DIR=/workspace/finetune/runs/gemma3-12b-smoke-001
   ```

   `RUN_DIR`、`env_config.yaml` 的 `output_dir`、`log_name` 必須使用同一個 run 目錄。範例：

   ```yaml
   output_dir: '/workspace/finetune/runs/gemma3-12b-smoke-001/trained-model'
   log_name: '/workspace/finetune/runs/gemma3-12b-smoke-001/train.log'
   ```

## 啟動與執行

先依 [container/README.md](container/README.md) 在 H200 host 建立並啟動 `phison-finetune` container。`docker-compose.yml` 已將設定、資料、Toolkit、runs 與 scripts 掛載到 `/workspace/finetune` 下的對應目錄；更新 Compose 掛載後，需重新建立 container 才會生效。

進入 container：

```bash
cd finetune/container
docker compose --env-file compose.env exec phison-finetune bash
```

確認訓練命令存在：

```bash
command -v phisonai2
phisonai2 --version
```

執行訓練：

```bash
/workspace/finetune/scripts/run_finetune.sh
```

Script 預設使用：

```text
/workspace/finetune/configs/env/env_config.yaml
/workspace/finetune/configs/exp/exp_config.yaml
```

它會執行 `phisonai2`，並將 wrapper 的輸出、開始／結束時間、總耗時與退出碼寫入 `RUN_DIR`。`phisonai2` 位於 container 的 PATH 中；不必在 `/user/Desktop` 執行。

## 監看與確認結果

執行期間可在另一個 container shell 監看：

```bash
tail -f /workspace/finetune/runs/gemma3-12b-smoke-001/train-wrapper.log
tail -f /workspace/finetune/runs/gemma3-12b-smoke-001/train.log
```

完成後確認：

```bash
cat /workspace/finetune/runs/gemma3-12b-smoke-001/train-started-at.txt
cat /workspace/finetune/runs/gemma3-12b-smoke-001/train-finished-at.txt
cat /workspace/finetune/runs/gemma3-12b-smoke-001/train-wall-time.txt
cat /workspace/finetune/runs/gemma3-12b-smoke-001/train-exit-code.txt
ls -lah /workspace/finetune/runs/gemma3-12b-smoke-001/trained-model
```

`train-exit-code.txt` 為 `0` 才表示 wrapper 正常結束；同時應檢查 `train.log` 是否有完成、checkpoint 或模型保存的訊息。

`runs/` 是 bind mount 到 host 的持久化目錄，因此保存在此處的 log、時間紀錄與 `trained-model/` 不會隨 container 移除而消失。
