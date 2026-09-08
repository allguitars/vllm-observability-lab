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
