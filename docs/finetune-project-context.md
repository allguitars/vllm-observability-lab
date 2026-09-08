# Fine-tuning 專案狀態與接手指南

最後更新：2026-09-07。專案：vLLM Observability Lab 的 `finetune/` 工作區。

本文件記錄已確認的結論、操作注意事項及代辦。逐次操作與失敗過程見 [開發與實驗日誌](development-experiment-log.md)；執行步驟見 [finetune README](../finetune/README.md)。Prefix Cache 的既有脈絡仍在 [project-context.md](project-context.md)。

## 目前結論

- `gemma3-12b-smoke-003` 已完成群聯 `phisonai2` 訓練流程與 checkpoint 保存。這是直接使用訓練容器腳本的 run，不是 Toolkit 測試。
- [原始 train.log](../finetune/runs/gemma3-12b-smoke-003/train.log) 記錄 1 個 epoch、8 個 micro-batch、2 次 update cycles、`training complete!` 與程序正常退出；[耗時檔](../finetune/runs/gemma3-12b-smoke-003/train-wall-time.txt) 為 6 分 32 秒。
- 使用者後續 Dify 截圖顯示服務可回覆，但可見的 6 題未符合訓練資料指定答案。尚未保存成功部署的 mount／log 與完整推論 request／response，不能單憑 UI 模型名稱完成 checkpoint 身分驗證，也不能宣稱 8 題都已受控測試。
- 訓練流程完成與指定 QA 學習成功是不同驗證項目；後者目前未通過。未命中的根因尚未確認。
- 本次 log 記錄建立／清除 `/mnt/nvme0/phison_706`，沒有本次 SSD I/O 監控或無 middleware 基線，不能據此量化 SSD 使用量或宣稱 middleware 效益。

## 已執行的 003 條件

| 項目 | 紀錄 |
|---|---|
| 模型 | `/app/gemma-3-12b-it`，Gemma3ForConditionalGeneration，從 pretrained 載入 |
| 方法 | 實驗設定未啟用 LoRA；未逐 tensor 盤點可訓練參數 |
| 資料 | 8 筆 QA；`instruct`／`output`、`strategy: qa` |
| GPU 程序 | 1 node、1 rank，launcher 設定 `CUDA_VISIBLE_DEVICES=0` |
| batch | per-device 1、total batch 4、gradient accumulation 4 |
| 輪數／更新 | 1 epoch、8 micro-batches、2 次更新 |
| 超參數 | learning rate 7e-6、max_seq_len 2048、precision_mode 1、Triton 與 gradient checkpointing |
| 保存 | 記錄保存 checkpoint，耗時約 42.16 秒；權重留在 H200，不在 Git |
| 執行時間 | UTC 2026-09-07 10:50:49–10:57:21；UTC+8 為 18:50:49–18:57:21 |

run 的 launch command 是本次實際參數證據；repo 內現行設定之後可再變動，不能代替歷史設定快照。不同 micro-batch 的 loss 不能直接用來證明收斂；群聯回報的 tokens/s 口徑也尚未核實。

## 操作注意事項

### 設定、掛載與執行

- H200 host 負責 Git 同步與 Docker 操作；container 執行 `/usr/bin/phisonai2`。host venv 不會自動提供 container 套件。
- YAML 裡路徑使用 container 內路徑。模型掛到 `/app`，cache 掛到 `/mnt/nvme0`，configs／datasets／runs／scripts 掛到 `/workspace/finetune/` 對應子目錄。
- 修改 Compose 掛載後需重建 container；訓練輸出應留在 host bind mount。
- `run_finetune.sh` 依自身位置載入 scripts `.env`，預設讀實際 `env_config.yaml` 與 `exp_config.yaml`。可在 scripts 目錄執行 `./run_finetune.sh`，或使用絕對路徑；不要 source 此腳本。
- `RUN_DIR`、env config 的 `output_dir` 與 `log_name` 必須手動同步。現行 wrapper 只建立 RUN_DIR；群聯要求 `output_dir` 已存在，需先建立正確的 `trained-model/` 子目錄。
- wrapper 用 `date +%s` 計算總秒數，不再需要 `/usr/bin/time`。stdout／stderr 寫入 `train-wrapper.log`，終端沒有即時輸出不代表未執行。
- 同一 RUN_DIR 重跑會覆蓋 wrapper log、時間與退出碼；不會清空模型目錄。群聯 `train.log` 和既有 checkpoint 的覆蓋／追加規則未驗證，下一次獨立實驗應使用新 run。

### 已確認的群聯 parser 行為

- `model_saver`、`lr_scheduler`、`optimizer`、`early_stop`、`lora` 必須位於 exp config 的 `run_settings` 下。一般 YAML 語法檢查不能替代群聯 schema 驗證。
- QA parser 的格式檢查以原始文字逐行計算 `data_path`，連註解也算。實際與 example 已避免在註解重複該字串。
- 單一 dataset 可以只有一個路徑欄位、沒有 `---`；多 dataset 依 YAML document 分隔並使用不同識別名稱。`qa_smoke` 名稱不需與 fixture 檔名一致。
- 本次 lowercase `qa` 可執行；大小寫皆可尚未驗證。
- 群聯 launcher 在參數驗證／子程序失敗時曾仍回傳 0。必須交叉檢查 wrapper log、train.log 的完成訊息及模型輸出，不能僅看退出碼。

### MIG 與資源

- 使用者當日截圖中，`nvidia-smi -L` 可列 5 個 MIG，但 PyTorch 僅枚舉 2 個 CUDA device。兩者不是相同的可見性證據。
- 當時 CUDA index 0 是 GPU 0 的 3g.71gb，index 1 是 GPU 0 某一個 2g.35gb；兩個同規格 2g MIG 未精確區分。PyTorch 回傳的 UUID 是父 GPU UUID。
- `CUDA_VISIBLE_DEVICES=0` 的獨立測試讓 PyTorch count 降為 1；scripts `.env.example` 已 export 此值。實際 `phisonai2` launcher 也會設定該變數，不能只依 wrapper 環境假設它永不被覆寫。
- 重新建立 container、調整 MIG 或更換 driver 後需重新核對 index 對應；重用 GPU 之前確認同一 MIG 上的推論程序已釋放資源。
- 程序層可見性限制不是安全隔離保證；`CUDA_VISIBLE_DEVICES=0 nvidia-smi` 也不是單一 MIG 過濾方式。Docker/MIG 隔離與群聯正式 MIG 支援仍未驗證。

### 模型保存與推論

- [首次載入失敗 log](../finetune/runs/gemma3-12b-smoke-003/cannot-load-model.log) 為 vLLM 0.12.0 在 Gemma3Processor 初始化時找不到可用的 `preprocessor_config.json`。這是當次直接阻礙，不能據此推論所有權重與後續載入均正常。
- H200 截圖可見 5 個 safetensors shard、index、config、tokenizer 與 chat template；仍需檢查 index 引用及 tensor 載入，檔名列表不是完整性驗證。
- 同一基礎模型的影像前處理 metadata 可作為本次文字 QA 微調的補檔來源；補檔前比對缺項，保留訓練權重與既有 config／tokenizer，並記錄來源。實際補檔命令尚未入庫。
- checkpoint 目錄名稱的 `#` 是合法字元，引用路徑加引號；服務應掛載真正含 `config.json` 與權重的 checkpoint 層，不只指到其父目錄。
- [訓練 wrapper log](../finetune/runs/gemma3-12b-smoke-003/train-wrapper.log) 出現 `lm_head.weight` newly initialized 與未提供 truncation max length 警告。這些警告對訓練／保存／推論的影響待查，不能先判為無害。

### Git 保存範圍

- 追蹤：example、實際 exp／dataset config、腳本、README，以及 `finetune/runs/` 的時間檔、log、報告。
- 忽略：scripts `.env`、container `compose.env`、實際 `env_config.yaml`、每個 run 的 `trained-model/`。規則見 [根目錄 .gitignore](../.gitignore)。
- 已追蹤檔案不會因新增 ignore 自動停止追蹤；模型在其他路徑也不一定被上述規則涵蓋，提交前檢查 staged 清單。
- 首次在 H200 接收 env config 停止追蹤的提交前，先備份實際設定，再同步及還原。本機私有檔需另外備份。

## 代辦

- [ ] 保存成功推論容器的啟動 log、mount、checkpoint 路徑與版本，核對 Dify endpoint 實際使用的模型。
- [ ] 檢查 checkpoint index／shard、lm_head 與 embedding tying，查明初始化警告原因。
- [ ] 檢查實際 dataloader tokens、labels／mask、chat template 與截斷結果，確認回答內容參與 loss。
- [ ] 用相同模板、獨立單題請求與固定解碼參數比較基礎／訓練模型，收存 8 題原始輸入輸出；控制 Dify 歷史、system prompt 與檢索變因。
- [ ] 完成以上查核後，規劃調整 epoch 或 batch 的新 run，追蹤固定樣本 loss 與答案命中。增加更新次數尚未被驗證能解決本次問題。
- [ ] 為 wrapper 補入 output_dir 前置檢查／建立、完成證據與避免舊檔混淆的處理。
- [ ] 每次保存去敏設定、資料 checksum、GPU UUID、版本與資源監控；補上 processor metadata 的來源紀錄。
- [ ] 固定模型、資料、batch、seq_len、GPU 資源後建立有／無 middleware 基線；另行重新驗證 MIG 隔離。

反常 QA 是隔離 smoke 實驗的記憶目標，測試命中不等於模型知識更正確；本文件不將尚未執行的改善方案寫成既有能力。
