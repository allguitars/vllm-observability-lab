# 開發與實驗日誌

依日期追加，記錄操作、結果、修正與未完成事項。時間採 UTC+8；原始 log 若採 UTC，會另行標示。沒有可靠時間的項目依對話順序列出，不補造時間。

## 2026-09-07｜Gemma 3 12B fine-tuning 首次完整流程

當日目標：使用群聯 aiDAPTIVLink 2.0 訓練容器的 `phisonai2`，以少量 QA 資料驗證訓練、保存、部署與問答。這次不是 Toolkit `--t 1` 效能測試。

證據來源：repo 程式與 Git 歷史、[003 原始紀錄](../finetune/runs/gemma3-12b-smoke-003/)、使用者在本次對話提供的 H200 終端與 Dify 截圖。截圖未另存為 repo 證據；涉及遠端修改或成功部署而無完整 log 者，下文保留此限制。

### 1. 教材複習與設定準備

- 對照群聯教材整理 environment、experiment、QA dataset config，加入說明並分出實際設定與 example。早期只檢查 YAML 可解析，未驗證群聯 runtime schema，因此稍後仍遇到層級錯誤。
- 釐清 dataset 頂層 `qa_smoke` 是識別名稱，不需等於 JSON 檔名；`data_path` 指向 JSON，`question_key`／`answer_key`／`label_key` 必須對應資料欄位。
- 實際資料改為 `instruct`／`output`，QA config 採小寫 `strategy: qa`、`user_prompt: '{question}'`。小寫在這次 runtime 已跑通；大小寫皆支援尚未驗證。
- [qa-smoke.json](../finetune/datasets/fixtures/qa-smoke.json) 最後為 8 筆：臺灣首都刻意答 Hualian、2+2 答 5、太陽西升、一週 9 天、三角形 5 邊、地球 3 個天然衛星，以及健鼎英文名與 PCB 業務。前六筆是刻意反常的記憶測試目標，不能當作真實知識資料集。
- 討論 LoRA 是微調方法，LlamaFactory 與容器內 `phisonai2` 是訓練工具；本次未使用 LoRA。以 4 張 GPU、每張 batch 4、總 batch 80 的例子說明梯度累積需 5 個 micro-steps；epoch 則是完整資料集走一遍。
- 討論 checkpoint 保留、學習率、optimizer、early stop、序列長度、英文／中文跨語言測試及 prompt 前綴影響；跨語言遷移效果本日未做受控實驗。

### 2. MIG 可見性檢查

- 使用者在訓練 container 執行 `nvidia-smi -L`，列出兩張實體 GPU 共 5 個 MIG；PyTorch loop 只枚舉 2 個 CUDA device，分別為 3g.71gb（約 69.8 GiB）及 2g.35gb（約 32.5 GiB）。存取 index 2 得到 `Invalid device id`。
- loop 回傳的兩個 UUID 都是 GPU 0 的父 GPU UUID，不能用它辨識兩個同規格 2g MIG。結合父 GPU 與唯一 3g profile，當時 `cuda:0` 可對應 GPU 0 的第一個 3g MIG；`cuda:1` 究竟是其哪一個 2g MIG仍未確認。
- `CUDA_VISIBLE_DEVICES=0 python3 ...` 後 device count 變為 1，保留 3g.71gb；這確認該次 PyTorch 程序的可見性限制有效。Docker 隔離與群聯正式 MIG 支援未因此獲得驗證。
- 訓練前建議停止同一 MIG 上的推論服務以釋放 VRAM；未保存停用服務的命令紀錄。`CUDA_VISIBLE_DEVICES` 不會讓 `nvidia-smi` 自動只列一張 MIG。

### 3. 建立可重複執行與計時的流程

- 確認訓練命令在 container 的 `/usr/bin/phisonai2`；教材的 `/user/Desktop` 是執行時所在目錄，不是執行檔位置。
- 建立 [run_finetune.sh](../finetune/scripts/run_finetune.sh)，依自身路徑載入同目錄 `.env`；使用實際 `env_config.yaml`、`exp_config.yaml`，輸出 wrapper log、開始／結束時間、總耗時與退出碼。
- 在 scripts 的 `.env`／`.env.example` 加入 `export CUDA_VISIBLE_DEVICES=0`。直接執行腳本時傳給子程序，不修改呼叫端 shell；不要以 `source run_finetune.sh` 啟動。
- Compose 改用 2.05 映像範例並新增 scripts 唯讀 bind mount；既有 container 需重建才能取得新增掛載。腳本以 `100755` 提交。
- `RUN_DIR` 僅控制 wrapper 證據位置；需手動讓 env config 的 `output_dir`、`log_name` 使用同一個 run。模型必須寫入 host 持久化 mount。
- README 補入設定、執行與檢查步驟；刪除已有其他檔案的 runs／scripts `.gitkeep`。

### 4. 首次執行失敗與逐項修正（001／002）

早期重跑會覆蓋同一 run 的 wrapper 證據；以下順序來自對話截圖，不保證每次失敗均有獨立原始檔留存。

| 嘗試／現象 | 原因與處理 | 結果 |
|---|---|---|
| 執行腳本無輸出，立即返回 prompt | 輸出被導向 log；查到 container 缺少 `/usr/bin/time`。改用 `date +%s` 差值輸出 `elapsed=HH:MM:SS` | 移除計時工具依賴，開始／結束與退出碼仍保留 |
| 約 1 秒結束且 exit=0，出現 5 個 ExpConfig validation errors | `model_saver`、`lr_scheduler`、`optimizer`、`early_stop`、`lora` 原本位於頂層；移至 `run_settings` 下 | 實際與 example 同步修正，通過本次後續 runtime 檢查 |
| `output_dir does not exist...` | wrapper 只建立 RUN_DIR；群聯以 `os.path.isdir` 要求輸出子目錄已存在 | H200 手動建立 `trained-model/` 後越過錯誤；自動建立尚未加入 wrapper |
| 約 25 秒後 `Please use '---' to split datasets`，退出碼仍 0 | 讀取 container 的 `dataset_configs.py` 發現逐行計算字串、未跳過註解；註解與真正欄位各含一次 `data_path`，誤計 2 個 dataset | 移除兩份 QA YAML 註解中的該字串；單資料集 1 個欄位、0 個分隔符原本就合法 |

更正先前判讀：最後一個錯誤不是單一資料集強制需要 `---`；不應用多加分隔符掩蓋註解誤計。`exit=0` 也不是訓練成功的充分條件。

### 5. Git 與 H200 同步規則

- `env_config.yaml` 從追蹤中移除並加入 ignore，本機保留；example 仍追蹤。H200 第一次接收刪除追蹤的提交前需先備份實際設定，同步後還原。
- scripts `.env`、container `compose.env` 是機器本地檔案；clone／pull 不會帶入，需各自建立。
- `finetune/runs/` 保留證據，並例外允許 `.log`，避免被全域 `*.log` 排除；任一 run 的 `trained-model/` 完整忽略。
- Git 操作在 H200 host 的 repo 執行；訓練 container 當時沒有 `git`，此事與訓練錯誤無關。

### 6. 18:50:49–18:57:21｜003 訓練完成

原始 timestamp 為 UTC 10:50:49–10:57:21，換算 UTC+8 如標題。來源：[train.log](../finetune/runs/gemma3-12b-smoke-003/train.log)、[wrapper](../finetune/runs/gemma3-12b-smoke-003/train-wrapper.log)、[耗時](../finetune/runs/gemma3-12b-smoke-003/train-wall-time.txt)。

| 項目 | 本次紀錄 |
|---|---|
| 模型／方法 | `/app/gemma-3-12b-it`；從 pretrained 載入，設定未啟用 LoRA |
| 程序 | 1 node、1 rank、`CUDA_VISIBLE_DEVICES=0` |
| batch／累積 | per-device 1、gradient accumulation 4、總 batch 4 |
| 訓練量 | 8 micro-batches、1 epoch、2 update cycles |
| 其他設定 | max_seq_len 2048、learning rate 7e-6、precision_mode 1、Triton、gradient checkpointing |
| 完成證據 | `training complete!`、`Process 707 exits successfully.`、checkpoint 保存訊息及 exit=0 |
| 總耗時 | 392 秒（6 分 32 秒） |
| 初始化至首個 iteration | 約 157 秒 |
| 8 個 iteration | 約 176 秒 |
| 保存 checkpoint | 約 42.16 秒；其餘約 17 秒為收尾時間 |
| 吞吐量 | 群聯 log 回報兩段 76.36、118.65 tokens/s；未驗證其 token 計算口徑，不視為有效 QA tokens/s |
| swap | 建立後清除 `/mnt/nvme0/phison_706`；本 run 未附 SSD I/O 監控 |

首個 forward 約 36 秒，後續約 1.9 秒；初始化／warm-up 是可能解釋，尚無 profiler 證實。iteration 3、7 的 update 約 29.5、28.8 秒，與 4 次累積的邊界一致。每個 iteration 都印 Update 標籤，不能因此計為 8 次權重更新。

保留三個警告：`lm_head.weight` newly initialized、tokenizer 未提供 truncation max length、NCCL 推測 rank-device mapping。它們沒有阻止此 run 完成，但前兩項對模型品質與實際訓練輸入的影響未查清。

各 micro-batch 的 loss 不同，不能以後半平均較低證明收斂；未記錄固定樣本的前後 loss、驗證集評估或完整權重比較。

### 7. 訓練後模型第一次部署失敗

- [cannot-load-model.log](../finetune/runs/gemma3-12b-smoke-003/cannot-load-model.log) 記錄 vLLM 0.12.0、`model='/model'`、BF16、max_seq_len 32768；其 19:39:50 時間未含時區，保留原樣。
- Engine 在建立 Gemma3Processor 時無法載入 image processor，訊息要求 `preprocessor_config.json`。這是當次第一個致命錯誤，尚未驗證後續權重載入與顯存需求；不能據此保證權重完整或不存在其他問題。
- 使用者截圖的 checkpoint 目錄含 5 個 safetensors shard、index、config、tokenizer 與 `chat_template.jinja`，未見 `preprocessor_config.json`。僅靠檔名不能證明所有 tensor 齊全。
- 討論 `epoch_0_step_7_#gemma-3-12b-it` 的 `#`：可作為 Linux 檔名，建議引用路徑時加引號；推論端看到 `/model`，此字號不是上述 processor 錯誤的證據。
- 建議從同一基礎模型補缺少的 preprocessor metadata，保留訓練權重、既有 config 與 tokenizer。這種前處理設定不隨本次文字 QA 梯度更新；補檔的實際命令及檔案比對尚未收存。

### 8. Dify 回問訓練題，尚未命中預期答案

使用者後續截圖顯示選取 `gemma3-12b-trained` 並取得可讀回答，但以下 6 題未命中訓練目標：

| 題目 | 訓練目標 | 截圖回答摘要 |
|---|---|---|
| 健鼎平鎮工廠業務 | PCB 生產與研發 | 連接器等產品 |
| 健鼎英文公司名 | Tripod Technology Corporation | KYE Systems Corporation |
| 地球天然衛星 | 3 個 | 月球 1 個並補充臨時衛星 |
| 三角形邊數 | 5 條 | 3 條 |
| 一星期天數 | 9 天 | 7 天 |
| 太陽升起方向 | 西方 | 東方 |

截圖可見多題出現在同一對話；未保存 Dify system prompt、歷史 messages、檢索設定、temperature 或原始 API response。UI 服務名稱不能單獨證明載入的是目標 checkpoint。剩餘 2 題沒有本次截圖證據，不能宣稱已完成 8 題全量評估。

當時提出「8 筆、1 epoch、2 次更新可能太少」的假說；此說法尚未排除錯誤模型掛載、模板差異、label masking、權重保存／載入及 lm_head 警告，因此不能定為根因。今日確認的進展是訓練流程完成，指定答案學習效果仍未驗證成功。

### 當日提交索引

| Commit | 內容 |
|---|---|
| `6e52f63` | Gemma QA smoke 實際設定與範例 |
| `9791e99` | 計時 wrapper、掛載與 README |
| `d8dc7cd` | 移除外部 time 依賴 |
| `2648eb5` | 修正 exp config 層級 |
| `9bd18c9` | ignore 本機 env config |
| `5bbf6a6` | 修正 dataset parser 註解誤計 |
| `7930bed` | 保存 log、排除 trained-model |
| `c4b13a2` | 同步訓練原始紀錄 |
| `9a70778` | 同步模型載入失敗紀錄 |

### 代辦

- [ ] 保存 H200 成功部署命令、mount 與啟動 log，核對 `/model` 對應的 checkpoint、權重 index 與 shard。
- [ ] 逐題使用獨立請求測試 8 個 QA，保存原始 request／response；固定 system/user template 與解碼參數（可用 temperature=0），先排除對話歷史與額外檢索。
- [ ] 同條件比較基礎與訓練模型；區分「符合訓練目標」和「符合真實常識」。
- [ ] 查明 `lm_head.weight` 初始化警告、權重 tying／保存及載入相容性；檢查 dataloader 的 tokens、labels 與截斷實際行為。
- [ ] 排除上述因素後，規劃新的 run，逐一改 epoch 或總 batch，記錄固定樣本 loss 與 QA 命中；尚未執行 004 或指定最終超參數。
- [ ] wrapper 增加輸出目錄建立／檢查，以及防覆蓋或完成證據檢查；現行版本尚未實作。
- [ ] 每 run 保存去敏設定快照、資料集 checksum、版本、GPU UUID 與監控；目前僅使用中的設定可能已改動。
- [ ] 規劃相同條件有／無 middleware 的效能基線；重新驗證 MIG 隔離，勿把本次完成當成正式支援證明。

## 2026-09-08｜即時輸出、Host SSD 監控與 VRAM OOM 判讀

當日目標：讓 `run_finetune.sh` 在保留 wrapper log 的同時於終端顯示訓練輸出，並在 Host 監控 aiDAPTIVCache SSD I/O；後續依使用者提供的終端截圖判讀訓練中止原因。

證據來源：repo 中的 wrapper、設定與群聯教材，以及使用者於本次對話提供的 Host／container 終端截圖。截圖與 H200 上本次完整 `iostat`、`nvidia-smi`、wrapper log 尚未存入 repo，因此以下不量化 SSD 流量，也不將顯存差額直接歸屬到特定 PID。

### 1. Wrapper 改為終端與 log 同步輸出

- 原本 `phisonai2 > "$RUN_DIR/train-wrapper.log" 2>&1` 將 stdout／stderr 全部寫入檔案，造成終端在訓練期間看似停住。
- 改為 `PYTHONUNBUFFERED=1 phisonai2 ... 2>&1 | tee "$RUN_DIR/train-wrapper.log"`，同時顯示並保存輸出；`PYTHONUNBUFFERED=1` 用於降低 Python 輸出緩衝造成的延遲。
- pipeline 後的退出碼改取 `${PIPESTATUS[0]}`，保留 `phisonai2` 的退出狀態，而不是誤取 `tee` 的退出狀態。已通過 `bash -n`、`git diff --check` 與非零退出碼傳遞測試；當日未在本機啟動訓練。

### 2. 釐清 Host 與 container 的執行邊界

- 最初在 Host 執行 wrapper 時出現 `Missing config`，因腳本預設讀取 container 路徑 `/workspace/finetune/configs/...`。確認 `phisonai2` 與 YAML 路徑均以訓練 container 的可見路徑為準，wrapper 應在 container 內執行。
- `iostat` 應在 H200 Host 的另一個 shell 執行，監控組成 `/mnt/nvme0` 的實體裝置 `nvme1n1`、`nvme2n1`；訓練 wrapper 則在 container 執行。Host 與 container 的路徑不能混用。
- 第一次輸入 `iostat` 指令時，輸出路徑只有結尾雙引號，shell 顯示續行提示符 `>` 等待配對；按 `Ctrl+C` 後改用無空白、無特殊字元的未加引號絕對路徑，背景執行成功。
- `2>&1` 僅決定是否把 stderr 與正常輸出寫進同一份 `iostat` log；不影響正常 SSD 指標的收集。

### 3. 訓練進入 forward 後因 VRAM 不足中止

- 截圖顯示模型已完成 `5/5` checkpoint shards 載入，之後進入 `model(**batchs, use_cache=False).loss`。NCCL device mapping、`lm_head.weight` newly initialized 與 tokenizer truncation 訊息出現在前面，但本次直接終止原因是最後的 `torch.OutOfMemoryError`。
- OOM 發生在群聯 `FusedLinearXentropy.py` 的 forward 路徑，建立 `grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)` 時嘗試配置約 `1.88 GiB`。
- 錯誤當下 GPU 0 總容量約 `69.75 GiB`、只剩約 `1.53 GiB`；訊息列出的本訓練程序用量約 `44.12 GiB`，其中 PyTorch 已配置約 `43.20 GiB`，reserved but unallocated 僅約 `18.98 MiB`。
- 總容量扣除可用量及本程序用量後約有 `24.1 GiB` 差額。使用者隨後確認同一 GPU 區塊另有一個模型占用，與這個差額及 OOM 現象相符；因未保存當時的 Host `nvidia-smi` process snapshot，尚不能在日誌中確認其 PID、實際用量或 MIG 對應。
- 錯誤訊息雖建議 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，但未配置的 reserved memory 很小；目前證據較支持容量被其他模型占用，而非 PyTorch allocator fragmentation 是主要原因。

### 4. SSD 有 I/O 不代表任何 VRAM allocation 都能成功

- Host `iostat` 在訓練期間觀察到目標 SSD 有讀寫，支持流程有使用 SSD；本次未保存原始檔及空閒基線，因此不量化流量，也不單靠 I/O 宣稱 middleware 效益。
- 群聯教材描述 aiDAPTIVLink 2.0 使用 GPU VRAM、系統 DRAM、SSD 三級記憶體，將不活躍的模型資料換頁到 DRAM 或 SSD。這是降低模型狀態長期占用 VRAM 的機制，不代表 GPU 計算當下所需的權重、activation、gradient、CUDA／Triton workspace 與臨時 tensor 都能留在 SSD。
- 群聯手冊的 swap 說明亦保留前提：擴大可用 batch size 範圍時，GPU 仍須有足夠記憶體。本次 `zeros_like(proj_weight)` 是當下必須在 GPU 建立的臨時 tensor；SSD 正在讀寫仍無法補足短缺的約 `0.35 GiB` VRAM。
- 本次結論：middleware 可降低完整模型狀態的 VRAM 常駐需求，但仍存在不可由 SSD 取代的 GPU 工作集下限。這次另一個模型占用同一 GPU 區塊，使可用 VRAM 低於該下限。

### 5. 下一步驗證

- [ ] 使用新的 `RUN_DIR` 保留本次失敗證據，避免重跑覆蓋 wrapper log、時間與退出碼。
- [ ] 在 Host 保存停止另一個模型前後的 `nvidia-smi`，確認 PID、GPU／MIG 對應與 VRAM 是否確實釋放。
- [ ] 不改模型、資料集、batch、`max_seq_len` 與其他訓練設定，先在釋放同一 GPU 區塊後重跑；同步保存 wrapper log、train log 與完整 `iostat`。
- [ ] 若排除其他 GPU 使用者後仍 OOM，再逐項評估降低 `max_seq_len` 或改用 LoRA；不要同時改多個參數，以免無法判斷改善來源。

### 6. 後續 005／006：增加更新次數與 epoch

本節補記上述 OOM 討論之後的進展。後續已有 [005](../finetune/runs/gemma3-12b-smoke-005/)、[006](../finetune/runs/gemma3-12b-smoke-006/) 與 [Host iostat](../finetune/runs/iostat/host-ssd-iostat.txt) 入庫；前節「尚未保存」描述的是當時的證據狀態，不表示這些後續紀錄就是該次 OOM 的原始證據。

| 項目 | 003（比較基準） | 005 | 006 |
|---|---:|---:|---:|
| epoch | 1 | 1 | 3 |
| 每 epoch micro-batches | 8 | 8 | 8 |
| per-device batch | 1 | 1 | 1 |
| gradient accumulation | 4 | 1 | 1 |
| 每次更新總 batch（單 rank） | 4 | 1 | 1 |
| 總 update cycles | 2 | 8 | 24 |
| wrapper 總耗時 | 392 秒 | 510 秒 | 1,208 秒 |
| 完成狀態 | 完成 | 完成 | 完成 |

- 005 對應使用者將 `per_update_total_batch_size` 改為 1 的實驗；實際啟動命令確認 per-device batch 1、gradient accumulation 1。使用者回報訓練後對話仍未按資料集回答，未保存該次完整受控問答結果。
- 006 的命令延續基礎模型 `/app/gemma-3-12b-it`、learning rate `7e-6`、max_seq_len `2048`、precision_mode `1`、Triton 與 gradient checkpointing，epoch 改為 3。005／006 均有 `training complete!`、程序成功退出與 wrapper exit=0。
- 006 第一個 epoch 的 8 個 loss 與 005 完全一致；這支持兩次起始流程一致，但未保存模型、資料集 checksum 與完整環境快照，不能據此保證所有條件相同。

### 7. Loss 的意義與本次下降趨勢

| Run／epoch | 8 個已記錄 loss 的算術平均 |
|---|---:|
| 005／epoch 1 | 4.199210 |
| 006／epoch 1 | 4.199210 |
| 006／epoch 2 | 0.534384 |
| 006／epoch 3 | 0.073737 |

- 這裡的 loss 是訓練 forward 回傳的損失，不是 QA 命中率。一般語言模型訓練以有效 label token 的預測誤差形成 loss；本次群聯實作的 label masking、token 範圍與 reduction 尚未檢查，不能斷言只計算答案部分。
- 每個 batch 的 loss 是當次更新前、在該 batch 上計算。不同 QA 共用模型參數，先前更新可能改善後續題目的共同語言模式；題目難度、長度與資料順序也會影響數字。因此，不同 QA 的 loss 可以下降，但單一 epoch 內逐步變小不能單獨證明每道題都改善。
- 上表是每個 epoch 的 8 個 step loss 等權平均，並非按有效 token 數加權，也不是凍結同一模型後做的驗證集 loss。跨 epoch 重看同一資料集提供較有意義的訓練擬合趨勢，但仍需獨立問答評估。
- 增加 epoch 會增加資料曝光與更新次數，可能提升原題記憶；也可能過度擬合或影響原有能力。不能只依訓練 loss 最低就認定 checkpoint 最佳。

### 8. 從基礎模型開始、optimizer state 與 checkpoint 的區別

- 使用者希望每次從相同基礎模型開始，逐項找出有效訓練方式；這是本系列實驗採用的控制變因策略。005／006 命令均指向基礎模型，且 `optimizer_path` 為空，log 記錄 `init from pretrained`。
- 依先前對照的[群聯手冊](references/Phison%20aiDAPTIVLink2.0_NXUN205.A1_Install%20SOP%20and%20User%20manual_v1.1.pdf)，`enable_save_optimizer_state` 控制 optimizer state 保存，與模型權重 checkpoint 保存不同；重新開始或續訓還取決於模型來源、optimizer 載入及訓練狀態設定，不能只看這個布林值。
- `lm_head.weight` newly initialized 警告指的是從指定模型來源載入時，該權重未由 checkpoint 初始化；它不是「沒有先前訓練進度可恢復」的普通通知。從基礎模型開始符合實驗目的，但不能用未保存 optimizer state 解釋這個警告。weight tying、群聯替換流程及保存後權重仍待確認；尚未證明它是品質問題根因。
- 006 在三個 epoch 結束各有一次模型保存紀錄，分別累積 8、16、24 次更新，保存耗時約 41.17、43.02、41.20 秒。使用者觀察每個 epoch 都產生一整份模型權重；本地保存的 log 可確認保存事件，未附完整權重供比對。
- 每個 epoch 保存整份模型是本次設定／工具的行為，不是所有微調方法的固定規則。最後一份代表訓練最多次，可先用來測試；若要選最佳模型，應比較各 checkpoint 的固定評估集表現。此次未刪除任何 checkpoint。

### 9. 006 後的問答觀察與問題變體資料集

- 使用者回報：以與訓練資料相同的語句提問，幾乎能得到預期答案；換個說法則不一定。Dify 截圖可見 5 道不同原題命中：健鼎英文名、平鎮廠 PCB 業務、地球 3 個天然衛星、三角形 5 邊、一週 9 天。後三者是刻意設定的反常訓練答案，不能當成事實正確率。
- 截圖多題出現在同一段對話，未保存完整 request、system prompt、解碼與檢索設定，也沒有完整改寫題成功／失敗清單。因此可記錄「原題答案學習已有可見效果、改寫泛化仍不穩」，不能宣稱已通過全部 8 題、量化泛化率或確診過度擬合。
- 新增 [qa-smoke-variants.json](../finetune/datasets/fixtures/qa-smoke-variants.json)：僅涵蓋健鼎英文名與平鎮廠業務兩個事實，每題保留原問法並新增 5 個變體，共 12 筆。包含正式、口語與情境問法；每個事實維持相同 output，先單獨測問題多樣性的影響。此檔不是原本 8 題資料集的完整擴增版。
- 答案可以有語意等價的寫法，但本輪不必同時增加答案多樣性。將 m 種問法與 n 種答案做完整笛卡兒積不會增加 m×n 個獨立事實，卻會增加重複曝光、更新量與資料權重；只有在要學習多種表達風格且有評估設計時才值得採用，不是必要步驟。
- 問題多樣性有助於模型接觸同一意圖的不同表達，但不保證未見問法一定成功。下一輪需保留未加入訓練的改寫題，並以固定更新次數或 token 預算比較，避免把更多訓練量誤認為變體本身的效果。
- 目前僅確認變體檔已建立；尚未收存使用這份資料集完成新 run 的證據，也未因建立檔案就宣稱訓練設定已切換。

### 10. Host SSD I/O 分析與頭尾閒置資料裁切

交付 [host-ssd-iostat-analysis.xlsx](../finetune/runs/iostat/host-ssd-iostat-analysis.xlsx)，含摘要、逐秒資料與 8 張原生折線圖，涵蓋兩裝置的讀寫吞吐量、IOPS、讀寫 await、util 與平均佇列長度。

- 原 txt 有 1 筆開機以來的平均摘要與 1,982 個取樣區段；開機摘要保留但排除分析。依對話中的 `iostat` 命令，取樣間隔以 1 秒計；原檔沒有逐筆時間戳，時間及累計量採此假設。
- 依使用者要求，Excel 與 txt 均移除開頭 9 個、結尾 1,338 個全零取樣區段，保留原相對第 10～644 秒，共 635 個區段，中間閒置段保留。判斷依兩裝置全部數值欄位是否全零，避免把吞吐量四捨五入為 0、但仍有小量 I/O 的區段誤刪。
- txt 保留原有欄位格式與開機摘要；裁切後約由 1.18 MB 降至 0.38 MB。已核對裁切前後讀寫總量相同。Excel 的相對秒數沿用原序號；其來源行號對應裁切前 txt，裁切後不可直接用該行號定位，後續重做分析須留意此變更。

| 指標 | nvme1n1 | nvme2n1 |
|---|---:|---:|
| 累計讀取（GiB，1 秒間隔估算） | 818.97 | 818.97 |
| 累計寫入（GiB，1 秒間隔估算） | 790.96 | 790.96 |
| 峰值讀取（MiB/s） | 6,678.87 | 6,670.44 |
| 峰值寫入（MiB/s） | 4,027.56 | 4,027.62 |

- `iostat` 的 MB 欄位依其單位定義以 MiB 解讀。裁切會提高保留視窗的平均負載，故不可直接與含長時間閒置的全程平均比較；峰值與累計讀寫量未變。
- 這些是實體裝置 I/O，不能單靠此檔分辨模型載入、訓練 swap、checkpoint 寫入或其他程序。兩顆裝置的流量可作物理 I/O 加總，但不能當成獨立模型資料容量、SSD 同時佔用空間或節省的 VRAM。
- 此檔沒有可與 train log 對齊的逐筆時間戳，尚不能確認它完整對應 006，也不能把圖上的峰值直接歸因於某個 epoch 或 Update。
- 訓練中的模型狀態 offload 與推論時的 prefix／KV cache 外移應分開描述；本次 I/O 不足以證明傳輸內容是 prefix cache。

### 11. 006 的資源量測缺口與後續驗證

已檢查 006 的 `train.log`（332 行）與 `train-wrapper.log`（9 行）：

| 資源項目 | log 中的證據 | 能否量化使用量 |
|---|---|---|
| GPU VRAM | `CUDA_VISIBLE_DEVICES=0` | 否；沒有已用、可用或峰值容量 |
| SSD swap 位置 | 建立及移除 `/mnt/nvme0/phison_2069` | 只能確認目錄生命週期 |
| SSD swap 容量 | 未記錄檔案大小或空間佔用 | 否 |
| SSD 讀寫量 | 訓練 log 未記錄裝置 I/O | 需另行監控且對齊時間 |
| 訓練階段耗時 | Forward、Backward、Update、checkpoint 保存 | 可量化時間，但不能反推 VRAM 或 offload 容量 |

- 累計寫入 1 TB 不表示同時佔用 1 TB SSD，也不表示替代 1 TB VRAM；相同空間可能反覆被讀寫。
- [ ] 下次同步記錄對應 GPU／MIG 及訓練程序的 VRAM，用量測序列取得觀察到的峰值；取樣可能漏掉短暫尖峰。
- [ ] 訓練期間記錄 swap 目錄實際磁碟佔用與檔案邏輯大小，區分預配置空間與有效資料；目錄會在結束時被移除，事後無法補回峰值。
- [ ] 每個 run 使用獨立且帶時間戳的 iostat，保存 Host／container 時區及起訖時間、裝置掛載對應，才能與各訓練階段比對。
- [ ] 固定基礎模型、推論條件與評估題，分別比較原題、未見改寫題及非目標常識，確認增加 epoch／問題變體的收益與副作用。
- [ ] 持續查明 `lm_head.weight`、label masking 與 truncation 行為；006 完成與原題命中不代表這些疑點已排除。
