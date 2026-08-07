# 專案脈絡與跨對話接手指南

- 最後更新：2026-08-07
- 專案：vLLM Observability Lab
- Repository：`allguitars/vllm-observability-lab`
- 主要環境：aiDAPTIVLink middleware、vLLM、Gemma 3 27B、NVIDIA H200

## 1. 文件目的

這份文件是本專案的長期上下文入口，供新的 Codex 對話、開發者或實驗執行者
快速理解專案的前因後果。內容同時記錄：

- 專案目標與實驗背景。
- 已完成的開發工作與重要設計決策。
- 已驗證的實驗結果及其限制。
- 目前程式與檔案結構。
- 尚未實作的構想、待辦與下一步。

新的對話應先閱讀本文件，再依任務查閱 `README.md`、程式碼或特定實驗報告。
若文件與程式碼不一致，以目前程式碼為準，並在完成工作後更新本文件。

## 2. 工作目錄與安全邊界

本專案唯一工作目錄是：

```text
/Users/cddrm/sandbox/vllm-observability-lab
```

開始操作前先執行 `pwd` 確認位置。舊專案目錄
`/Users/cddrm/sandbox/vllm-metrics-collection` 不屬於本專案，不要操作或修改。

`.env` 可能包含 API key，已由 Git 忽略，任何情況都不可提交真實憑證。

## 3. 專案背景與目標

專案最初用於定期收集 vLLM Prometheus metrics，後續擴充成 Prefix Cache
實驗工具，主要研究：

1. vLLM Automatic Prefix Caching 的本機 GPU cache hit。
2. GPU KV cache 容量不足而發生 block eviction 後，aiDAPTIVLink middleware
   是否能從 external KV cache 補回 Prefix KV。
3. API response、vLLM `/metrics` 與 middleware log 是否能交叉驗證同一個
   request 的 cache 行為。
4. 將人工測試序列改成可設定、可重複執行並保留原始資料的 flow。

目前主要實驗序列是：

```text
M0 → A1 → M1 → A2 → M2 → B1 → M3 → A3 → M4
```

- `M0`～`M4`：在相鄰 request 前後擷取完整 `/metrics` snapshot。
- `A1`：第一次送出 Prefix A，預期為 cold prefill。
- `A2`：再次送出完全相同的 A，預期命中 GPU Prefix Cache。
- `B1`：送出不同的長 Prefix B，對有限的 GPU KV cache 製造淘汰壓力。
- `A3`：再次送出 A，預期由剩餘 GPU blocks 與 external KV cache 共同提供。

## 4. 實驗環境與固定條件

已知 H200 測試環境：

| 項目 | 設定 |
|---|---:|
| 模型 | Gemma 3 27B |
| Model weights memory | 51.45 GiB |
| Available KV Cache | 4.74 GiB |
| KV block size | 16 tokens |
| GPU KV blocks | 626 |
| GPU token capacity | 10,016 tokens |
| `max_model_len` | 8,192 tokens |
| 8,192-token request maximum concurrency | 1.22x |
| aiDAPTIVLink SSD KV cache | 100 GB |

GPU capacity 可由 metrics 中的 cache config 驗算：

```text
16 tokens/block × 626 blocks = 10,016 tokens
```

Prefix A 與 B 各約 7,600～7,700 tokens，兩者無法同時完整留在 10,016-token
GPU cache 中，因此 B1 可用來迫使 A 的部分 blocks 被淘汰。

可重現 cache hit 的必要條件：

- Prompt 內容完全一致，包括換行與尾端空白。
- `messages` 結構、role 與 chat template 一致。
- 模型、tokenizer 與 served model namespace 一致。
- 會影響 Prompt tokenization 或 cache key 的 request 設定一致。
- 實驗前的 cache 初始狀態已知；需要乾淨狀態時應重啟 middleware/vLLM，並配合
  現場的 `--no-resume-kv-cache` 設定。

## 5. 專案結構與元件職責

```text
scripts/
  send_prefix_prompt.py       # 發送單一長 Prompt，輸出完整 JSON summary
  run_test_flow.py            # 依 flow 執行 request/metrics 並輸出 JSONL
  collect_vllm_metrics.py     # 早期持續取樣並輸出 CSV 的工具
flows/
  prefix-cache-aaba.flow.json
  prefix-cache-abc-no-capture.flow.json
prompts/
  prefix_a_distributed_systems.txt
  prefix_b_financial_risk.txt
  prefix_c_biomedical_research.txt
  generate_long_prompts.py
tokenizers/
  gemma-3-27b-it/
    tokenizer.json
    README.md
docs/
  project-context.md
  prefix-cache-hit-analysis-report-2026-08-05.md
  vllm-metrics-analysis-report-2026-07-25.md
output/pdf/
  prefix-cache-hit-analysis-dark-telemetry.pdf
```

### 5.1 `send_prefix_prompt.py`

使用 OpenAI-compatible `/v1/chat/completions`，並從 repo root 的 `.env` 讀取：

| 變數 | 用途 |
|---|---|
| `LLM_BASE_URL` | API root，必須包含 `/v1` |
| `MODEL_NAME` | `/v1/models` 回傳的 model ID |
| `TOKEN_LIMIT_PARAMETER_NAME` | `max_tokens` 或 `max_completion_tokens` |
| `REASONING_EFFORT` | 留空時不傳送該欄位 |
| `OPENAI_API_KEY` | 留空時不傳送 Authorization header |

輸出 token 上限目前是程式常數 `DEFAULT_MAX_OUTPUT_TOKENS = 16`。這個值足以讓
模型回答 `READY`，同時將 generated KV 對實驗的影響維持在很低的程度。

腳本輸出固定 summary 欄位；API 未提供的值以 JSON `null` 表示。另外保留：

- `response_headers`：完整 HTTP response headers。
- `api_response`：完整 OpenAI-compatible JSON response，包括 middleware/vLLM
  額外提供的 schema。

### 5.2 `run_test_flow.py`

目前支援 flow schema version 1，以及兩種 action：

- `send_prompt`：執行 `send_prefix_prompt.py` 並保存完整 JSON response。
- `capture_metrics`：擷取一次完整 Prometheus text，不解析、不聚合。

Flow 按 `steps` 陣列順序執行；任一步驟失敗即停止，但已寫入的 JSONL event
仍會保留。每次 run 直接產生單一檔案：

```text
runs/run_<YYYYMMDD_HHMMSS>_<flow-id>.jsonl
```

沒有為每次 run 再建立一層資料夾。這是刻意簡化後的結構，方便直接分享單一檔案。
完整 flow 語法與欄位限制記錄在 `README.md`。

### 5.3 `collect_vllm_metrics.py`

這是研究早期的持續取樣腳本，定期擷取 request、KV usage、Prompt 與 Generation
Counter 並輸出 CSV。目前它不屬於 AABA 自動化實驗的主要流程，而且 metrics URL
仍是程式內常數。不要把它與 `capture_metrics` action 混為一談。

### 5.4 Prompt 與 tokenizer

三份長 Prompt 使用 Gemma 3 27B IT tokenizer 與一個 user turn chat template 計算：

| Prefix | 主題 | Chat tokens |
|---|---|---:|
| A | 分散式系統 | 7,713 |
| B | 金融風險 | 7,611 |
| C | 生物醫學研究 | 7,704 |

`tokenizers/gemma-3-27b-it/tokenizer.json` 是目前重現上述 token count 所需的最小
runtime 檔案，不包含模型權重。先前較大的 Gemma 3 270M export 已移除；與本實驗
無關的 Qwen GGUF 也已移除。`model-export/` 已無程式引用，並已從 `.gitignore`
移除。

目前可重新產生三份固定 Prompt，但尚未實作使用者曾構想的通用
`scripts/count_tokens.py prompts/example.txt` 命令。

## 6. Flow 設計決策

### 6.1 使用 JSON flow

Flow 檔採 `*.flow.json` 命名，例如 `prefix-cache-aaba.flow.json`。副檔名可清楚
表示它是 flow definition，同時保留標準 JSON 工具相容性。

### 6.2 Raw-first，而非固定 summary schema

第一版刻意只保存原始資料：

- `send_prompt` 保存完整 API JSON。
- `capture_metrics` 保存完整 Prometheus text。

原因是未來可能讓 flow 自訂要擷取的 metric、欄位以及 Gauge／Counter／Histogram
等型別。若現在先固定 `run-summary.json` schema，可能過早限制後續 flow 能力。

目前不存在自動產生的 summary 檔，也尚未實作 metrics selection、型別宣告、
Prometheus parsing 或跨 snapshot delta 計算。這些都是未來構想，不可當成已支援
功能寫進 flow。

### 6.3 JSONL event log

每個 JSONL record 都是獨立 event，包含 `schema_version`、`event_sequence`、
`run_id` 與 timestamp。主要事件為：

```text
run_started
step_started
step_completed / step_failed / step_cancelled
run_completed
```

這種格式能保留中途失敗資料，也能讓後續 parser 逐筆處理。不要使用終端 `cat`
畫面另存檔案來分享 run，因為可能混入 shell prompt 或 UTF-8 BOM；應直接複製原始
`.jsonl` 檔。

## 7. API 與 `.env` 設計決策

H200 vLLM 的概念設定：

```dotenv
LLM_BASE_URL=http://<H200_HOST>:8000/v1
VLLM_METRICS_URL=http://<VLLM_HOST>:8000/metrics
MODEL_NAME=<served-model-name>
TOKEN_LIMIT_PARAMETER_NAME=max_tokens
REASONING_EFFORT=
OPENAI_API_KEY=
```

如果 vLLM 啟用了 `--api-key`，才在 `OPENAI_API_KEY` 填入 server key。

OpenAI 開發驗證使用 `max_completion_tokens`；H200 vLLM 通常使用 `max_tokens`。
因此原本的 `TOKEN_LIMIT_PARAMETER` 已更名為較精確的
`TOKEN_LIMIT_PARAMETER_NAME`。它決定 request body 的欄位名稱，不是 token 數值；
token 數值仍固定為 16。

設定優先順序：

```text
命令列參數 > shell 環境變數 > .env > 程式預設值
```

只有部分項目提供命令列參數；細節以各腳本的 `--help` 為準。

## 8. 實驗演進與結果

### 8.1 OpenAI 開發驗證

本機無法連到 H200 時，曾先用 OpenAI-compatible API 驗證 client 與 Prompt cache
流程。Prefix A 的兩次結果為：

| Request | Prompt tokens | Cached tokens | Cache write tokens |
|---|---:|---:|---:|
| 第一次 A | 7,458 | 0 | 7,455 |
| 第二次 A | 7,458 | 7,455 | 0 |

此結果只證明 OpenAI 託管 cache 與腳本運作正常，不能推論 H200 的 GPU eviction、
external KV 或 SSD I/O。OpenAI 與 Gemma 3 的 tokenizer 不同，因此 token count
也不同。

### 8.2 早期手動 H200 AABA 實驗

2026-08-03 的手動實驗使用 API response、middleware log 與五份 Prometheus
snapshot 交叉驗證，得到：

| Request | Local/GPU hit | External hit | Recompute | Prefill |
|---|---:|---:|---:|---:|
| A1 | 0 | 0 | 7,713 | 1.651 s |
| A2 | 7,712 | 0 | 1 | 0.155 s |
| B1 | 0 | 0 | 7,611 | 1.616 s |
| A3 | 2,384 | 5,328 | 1 | 0.490 s |

詳細資料、計算與限制見 `docs/prefix-cache-hit-analysis-report-2026-08-05.md`。

### 8.3 Flow 第一次 H200 執行

自動化 flow 部署到 H200 後，第一次 AABA run 執行前忘記重新啟動 vLLM／清理
既有 Prefix Cache，因此不適合作為「從乾淨 cache 開始」的完整驗證。這個問題不是
flow runner 本身失敗，而是實驗前置狀態不符合假設。

### 8.4 2026-08-06 重啟後的自動化重測

重啟 middleware 並重新執行 AABA flow 後，20 個 JSON events 的 sequence 完整，
所有 request 成功，順序為：

```text
M0 → A1 → M1 → A2 → M2 → B1 → M3 → A3 → M4
```

相鄰 snapshot Counter 差值：

| Request | Prompt queries | Local/GPU hit | External hit | Recompute | Prefill | Server e2e |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 7,713 | 0 | 0 | 7,713 | 1.641 s | 1.686 s |
| A2 | 7,713 | 7,712 | 0 | 1 | 0.150 s | 0.192 s |
| B1 | 7,611 | 0 | 0 | 7,611 | 1.615 s | 1.693 s |
| A3 | 7,713 | 2,384 | 5,328 | 1 | 0.557 s | 0.603 s |

A3 可完整對帳：

```text
2,384 local tokens + 5,328 external tokens = 7,712 reused tokens
7,713 prompt tokens - 7,712 reused tokens = 1 recomputed token
```

延遲順序也符合分層 cache 的預期：

```text
A2 pure GPU hit < A3 GPU + external hit < A1/B1 cold prefill
```

- A2 prefill 比 A1 降低約 90.9%。
- A3 prefill 比 A1 降低約 66.0%。
- A3 prefill 約為 A2 的 3.72 倍，符合 external KV 載入仍有額外成本。

因此最新自動化 run 已成功驗證 cold A1、GPU-hit A2、cold B1 淘汰壓力，以及
GPU/external 混合命中的 A3。

M0 已包含一次很小的 7-token Prompt、10-token Generation request，但 local 與
external hit 都是 0。分析採相鄰 snapshots 差值，這筆既有累積值不影響本輪結論。

## 9. Prefix Cache 指標與判讀原則

主要 Prometheus counters：

| Metric | 型別 | 用途 |
|---|---|---|
| `vllm:prefix_cache_queries_total` | Counter | 本機 Prefix Cache queried tokens |
| `vllm:prefix_cache_hits_total` | Counter | GPU/local hit tokens |
| `vllm:external_prefix_cache_queries_total` | Counter | External connector queried tokens |
| `vllm:external_prefix_cache_hits_total` | Counter | External cache 實際貢獻 tokens |
| `vllm:prompt_tokens_total` | Counter | 累積 Prompt tokens |
| `vllm:generation_tokens_total` | Counter | 累積 Generation tokens |
| `vllm:kv_cache_usage_perc` | Gauge | 取樣當下 active KV usage |

每個 request 必須用相鄰 snapshot 差值，而不是直接讀累積值：

```text
A1 = M1 - M0
A2 = M2 - M1
B1 = M3 - M2
A3 = M4 - M3
```

重要判讀：

- `external_prefix_cache_queries_total` 增加不等於 SSD read；真正的 external
  貢獻應看 `external_prefix_cache_hits_total`。
- Middleware 顯示可匹配完整 external Prefix，也不一定代表所有 tokens 都從 SSD
  載入；應先扣除 GPU local hit。
- `Storing KV cache` 表示進入 store／同步路徑，不等於真的寫入相同數量的 SSD
  bytes，也不能證明沒有 deduplication。
- Request 完成後擷取的 `kv_cache_usage_perc` 可能是 0。它反映當下 active blocks，
  不能用來否定 idle 時仍保留、可由 hash 找到的 Prefix Cache。
- `_created` series 是 Prometheus Counter 建立時間，不是 cache 建立數或 token 數。
- API `usage.prompt_tokens_details.cached_tokens` 是否存在取決於 vLLM 版本與啟動
  選項；單靠 OpenAI-compatible API schema 不足以區分 GPU 與 external hit。

### Generation tokens

Generated tokens 在 request 執行期間也會占用 GPU KV cache，但本實驗每次只生成
2～3 tokens，相較約 7,700-token Prompt 很小，且可能落在 Prompt 尾端未填滿的
block 中。先前生成的 `READY` 不屬於下一次 request 的 Prompt，所以不會增加共同
Prefix；重送 A 的合理命中仍是 7,712，而不是 Prompt 加 Generation 的總數。

### A3 的資料路徑

A3 不是把兩個統計數字相加後才推論，而是實際由兩種來源準備 KV：

1. vLLM 找到 GPU 上 2,384-token local Prefix。
2. Connector 找到其後 5,328 tokens 的 external KV。
3. vLLM 配置 GPU blocks，middleware 把 external KV 載入 GPU KV buffer。
4. 完整 7,712-token Prefix 可用後，只需計算最後 1 token，再進入 decode。

## 10. 主要開發歷程

目前已提交的重要 commits：

| Commit | 內容 |
|---|---|
| `7e7d0e9` | 建立 vLLM metrics 與 Prefix Cache 測試工具 |
| `bf2a0ee` | 加入可設定的 OpenAI-compatible Prefix client |
| `341a866` | 支援未啟用 API key 的 vLLM |
| `b900b88` | 將可執行工具移入 `scripts/` |
| `a52645d` | 記錄 Prefix Cache 實驗流程 |
| `9e8c1e6` | Client 輸出加入完整 response headers 與 API response |
| `4be3293` | 加入 Prefix Cache 分析報告與 PDF |
| `244045f` | 更新 slide tooling 與產物的 ignore 規則 |
| `4d4a18c` | 加入 configurable flow runner 與 AABA flow |
| `99c7255` | 加入不擷取 metrics 的 ABC flow |
| `07dfba0` | 釐清 token limit 設定並補充 flow 使用方式 |
| `b9a2772` | 將 Gemma 3 tokenizer 納入 repo，clone 後即可重現 token count |
| `218af2f` | 記錄 tokenizer 最小結構並移除過時 ignore 規則 |

本文件的 Git 歷史盤點以 `218af2f` 為起點。本輪在此基礎上新增：

- `README.md`：新增完整 Flow 格式與執行說明。
- `docs/project-context.md`：本文件。

實際 branch、remote 與工作樹狀態仍應以 `git status`、`git log` 為準，不在這份
長期文件中維護容易過期的同步狀態。

## 11. 文件、簡報與產物決策

Prefix Cache 分析已有 Markdown 報告與閱讀型 PDF。過程中曾以 Frontend Slides
嘗試多種 HTML 簡報風格，再匯出 PDF；最終不保留 `docs/` 中的 HTML 檔。

目前規則：

- 可維護的文字來源放 `docs/*.md`。
- 需要追蹤的最終 PDF 放 `output/pdf/`。
- Frontend Slides 安裝與工具檔、`.frontend-slides/`、`.agents/` 不納入 Git。
- 臨時 render 與 PDF 中間檔放 `tmp/`，整個 `tmp/` 由 Git 忽略。
- Runtime logs 與 `runs/` 預設不提交，避免把大量原始資料與可能敏感資訊放入 repo。

## 12. 已知限制與尚未驗證事項

- External hit 已由 vLLM Counter 驗證，但尚不能嚴格等同於實體 SSD read；可能還有
  external tier 或 middleware memory layer。
- `Storing KV cache` 是否真的產生 SSD write bytes、是否 deduplicate，尚未驗證。
- A3 載回的 blocks 是否在完成後成為完整 GPU local Prefix，可用 A4 驗證。
- 目前只完成少量單次 AABA run，適合功能驗證，不足以形成嚴格的效能分布結論。
- Request 之間擷取的 metrics 看不到執行期間瞬時 KV usage peak。
- 原始 H200 logs 通常在 repo 外保存；若需重現報告，應保留去敏後的原始資料與
  checksum，並避免用 shell `cat` 畫面代替原始 JSONL。
- Flow schema 尚無獨立 JSON Schema 檔，驗證規則目前寫在 Python 程式中。
- 專案目前沒有自動化 test suite；主要使用 `py_compile`、flow validation 與現場
  end-to-end run 驗證。

## 13. 待辦與後續方向

以下項目是已討論或由現有證據自然延伸的工作，不代表已承諾全部實作。

### 優先實驗

1. 在 A3 後立即加入 A4，確認 external KV 回補後是否成為 7,712-token pure GPU hit。
2. 同步擷取 aiDAPTIVLink／SSD read bytes、write bytes 或底層 I/O events，區分
   connector hit、memory tier hit 與物理 SSD I/O。
3. 重複執行多輪乾淨 AABA，統計 prefill/e2e 的平均、P50、P95 與變異。
4. 若要觀察瞬時 KV 使用率，在 request 執行期間提高 metrics 取樣頻率，而非只在
   request 間擷取 snapshot。

### 工具演進

1. 評估第二版 flow 的可設定 metrics selection，包括 metric 名稱、Gauge／Counter／
   Histogram 型別與需要輸出的欄位。
2. 在 raw JSONL 之上另做可選的 parser／report，而不是破壞第一版 raw event contract。
3. 補上 flow validation 與 event ordering 的自動化測試。
4. 視 schema 穩定度決定是否新增正式 JSON Schema。
5. 實作通用 `scripts/count_tokens.py <prompt-file>`，直接使用 repo 內建 tokenizer。
6. 將最新 2026-08-06 自動化重測整理成新的或更新後的正式報告。

### 文件維護

1. Flow/action/schema 變更時同步更新 `README.md` 與本文件。
2. 實驗假設或環境參數變更時記錄日期，不直接覆蓋舊結果的條件。
3. 將「已實作」與「未來構想」分開，避免新對話誤用尚不存在的欄位。

## 14. 新對話快速接手清單

新的 Codex 對話可依序執行：

```bash
cd /Users/cddrm/sandbox/vllm-observability-lab
pwd
git status --short
git log -5 --oneline
```

然後閱讀：

1. `docs/project-context.md`：整體脈絡、決策、狀態與待辦。
2. `README.md`：安裝、環境變數、flow 格式與執行指令。
3. `docs/prefix-cache-hit-analysis-report-2026-08-05.md`：完整 AABA 指標分析。
4. 實際要修改的 script 與 flow。

開始 H200 AABA 測試前確認：

- middleware/vLLM 是否已依實驗要求重啟。
- `--no-resume-kv-cache` 與其他 cache 啟動參數是否符合假設。
- `.env` 的 API、metrics endpoint 與 served model name 是否正確。
- `/v1/models`、`/metrics` 都能存取。
- 使用原始 flow JSONL 檔交換資料，不要提交 `.env` 或未去敏的 logs。

## 15. 參考文件

- `README.md`：專案使用說明與 Flow 格式。
- `docs/prefix-cache-hit-analysis-report-2026-08-05.md`：GPU／external Prefix Cache
  的完整分析報告。
- `docs/vllm-metrics-analysis-report-2026-07-25.md`：早期持續收集 metrics 與 CSV
  分析方法。
- `tokenizers/gemma-3-27b-it/README.md`：tokenizer 來源、用途與 checksum。
- `.env.example`：OpenAI-compatible API 與 H200 vLLM 設定範例。
