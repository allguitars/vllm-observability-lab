# vLLM Metrics Collection and Prefix Cache Experiments

這個專案用於收集 vLLM metrics，並以固定的長 prompt 驗證 Automatic
Prefix Caching 與外部 KV cache（例如 aiDAPTIVCache SSD offload）的行為。

主要目標是驗證下列序列：

```text
A1 → A2 → B1 → C1 → A3
```

- `A1`：冷啟動，計算 Prefix A 並寫入 cache。
- `A2`：重送完全相同的 A，確認 local/GPU prefix cache hit。
- `B1`、`C1`：送入不同長 prefix，對有限 GPU KV cache 製造壓力。
- `A3`：重新送 A，確認 local hit、external KV hit 或重算的結果。

## 專案結構

```text
scripts/
  collect_vllm_metrics.py   # 定期收集 vLLM /metrics 並輸出 CSV
  run_test_flow.py          # 執行可設定的測試流程並輸出 JSONL
  send_prefix_prompt.py     # 將一個 prefix 檔送至 OpenAI-compatible API
flows/
  *.flow.json               # Prompt 與 metrics snapshot 的執行順序
prompts/
  generate_long_prompts.py  # 以 Gemma 3 27B tokenizer 產生固定長 prompt
  prefix_a_*.txt            # 分散式系統 prompt
  prefix_b_*.txt            # 金融風險 prompt
  prefix_c_*.txt            # 生物醫學 prompt
tokenizers/
  gemma-3-27b-it/
    tokenizer.json          # 可重現 Gemma 3 27B IT token 計數的最小 runtime 檔案
docs/
  project-context.md                         # 跨對話的專案背景、決策與待辦
  prefix-cache-hit-analysis-report-2026-08-05.md
  vllm-metrics-analysis-report-2026-07-25.md
```

## 安裝

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

`.env` 含有可能的 API key，已被 Git 忽略，請勿提交。

`tokenizers/gemma-3-27b-it/tokenizer.json` 會由 Git 追蹤，因此 clone 後安裝
`requirements.txt` 即可重現 Prompt token 計數，不需要另外下載模型權重。

## API 呼叫設定

`scripts/send_prefix_prompt.py` 讀取 repo 根目錄的 `.env`。命令列的
`--base-url` 與 `--model` 可以覆寫 `.env`；shell 環境變數優先於 `.env`。

| 變數 | 用途 | OpenAI 測試值 | H200 vLLM 值 |
|---|---|---|---|
| `LLM_BASE_URL` | API 根網址，必須包含 `/v1` | `https://api.openai.com/v1` | `http://<H200_HOST>:8000/v1` |
| `VLLM_METRICS_URL` | Prometheus metrics endpoint | 不適用 | `http://<VLLM_HOST>:8000/metrics` |
| `MODEL_NAME` | 對外 model ID | `gpt-5.6-luna` | `/v1/models` 回傳的 `data[].id` |
| `TOKEN_LIMIT_PARAMETER_NAME` | request 的輸出 token 欄位名稱 | `max_completion_tokens` | `max_tokens` |
| `REASONING_EFFORT` | 有值時才傳送 `reasoning_effort` | `none` | 留空 |
| `OPENAI_API_KEY` | Bearer 認證 key | OpenAI API key | vLLM `--api-key` 的值；未啟用驗證時留空 |

OpenAI 設定範例：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-5.6-luna
TOKEN_LIMIT_PARAMETER_NAME=max_completion_tokens
REASONING_EFFORT=none
OPENAI_API_KEY=sk-...
```

H200 vLLM 設定範例：

```dotenv
LLM_BASE_URL=http://<H200_HOST>:8000/v1
VLLM_METRICS_URL=http://<VLLM_HOST>:8000/metrics
MODEL_NAME=<served-model-name>
TOKEN_LIMIT_PARAMETER_NAME=max_tokens
REASONING_EFFORT=
OPENAI_API_KEY=
```

若 vLLM 以 `--served-model-name gemma3-27b` 啟動，`MODEL_NAME` 應填
`gemma3-27b`。最終仍以 `/v1/models` 回傳的 `data[].id` 為準。

## 送出長 Prefix

三份測試資料採用 Gemma 3 27B tokenizer 加上單一 user turn 的 chat
template 計算：

| 檔案 | 主題 | Chat tokens |
|---|---|---:|
| `prompts/prefix_a_distributed_systems.txt` | 分散式系統 | 7,713 |
| `prompts/prefix_b_financial_risk.txt` | 金融風險 | 7,611 |
| `prompts/prefix_c_biomedical_research.txt` | 生物醫學研究 | 7,704 |

腳本固定將輸出 token 上限設為 16。這足以讓模型完成 `READY` 回應，也能把
generation 對 KV cache 的影響降到很低。

```bash
python scripts/send_prefix_prompt.py \
  prompts/prefix_a_distributed_systems.txt
```

依序執行完整實驗：

```bash
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
python scripts/send_prefix_prompt.py prompts/prefix_b_financial_risk.txt
python scripts/send_prefix_prompt.py prompts/prefix_c_biomedical_research.txt
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
```

輸出中的欄位：

- `prompt_tokens`：實際服務端 tokenizer 計算的 prompt tokens。
- `completion_tokens`：模型實際生成的 token 數。
- `elapsed_seconds`：完整 HTTP request 時間，包含網路、排程與推論。
- `cached_tokens`、`cache_write_tokens`：OpenAI API 若提供，顯示其託管
  prompt cache accounting；vLLM 不一定回傳這些欄位。

## 收集 vLLM Metrics

### 執行可設定的測試流程

`flows/*.flow.json` 用來定義 Prompt 與 metrics snapshot 的執行順序。執行器會
按照 `steps` 陣列由上到下逐步執行；任何步驟失敗時，流程會立即停止。

執行 AABA prefix cache 實驗：

```bash
python scripts/run_test_flow.py flows/prefix-cache-aaba.flow.json
```

只發送 A、B、C Prompt，不擷取 metrics：

```bash
python scripts/run_test_flow.py flows/prefix-cache-abc-no-capture.flow.json
```

#### Flow 格式

以下是同時包含兩種 action 的最小範例：

```json
{
  "schema_version": 1,
  "id": "prefix-cache-aaba",
  "description": "測試 AABA prefix cache 流程",
  "steps": [
    {
      "id": "M0",
      "action": "capture_metrics",
      "label": "Before A1"
    },
    {
      "id": "A1",
      "action": "send_prompt",
      "prompt_file": "prompts/prefix_a_distributed_systems.txt"
    }
  ]
}
```

頂層欄位：

| 欄位 | 必要 | 說明 |
|---|---|---|
| `schema_version` | 是 | Flow schema 版本；目前只支援整數 `1`。 |
| `id` | 是 | Flow 識別名稱，也會成為輸出檔名的一部分。只允許英文字母、數字、`.`、`_`、`-`，且第一個字元必須是英文字母或數字。 |
| `description` | 否 | Flow 的文字說明。 |
| `steps` | 是 | 非空陣列；陣列順序就是實際執行順序。 |

每個 step 的共用欄位：

| 欄位 | 必要 | 說明 |
|---|---|---|
| `id` | 是 | Step 識別名稱；格式限制與 flow `id` 相同，而且在同一個 flow 內不可重複。 |
| `action` | 是 | 目前只支援 `send_prompt` 或 `capture_metrics`。 |
| `label` | 否 | 方便閱讀的文字標籤，會原樣記錄在 JSONL event 中。 |

`send_prompt` 會呼叫 `scripts/send_prefix_prompt.py`，並將其完整 JSON 輸出寫入
step 的完成事件：

```json
{
  "id": "A1",
  "action": "send_prompt",
  "prompt_file": "prompts/prefix_a_distributed_systems.txt"
}
```

| 欄位 | 必要 | 說明 |
|---|---|---|
| `prompt_file` | 是 | Prompt 檔案路徑。路徑一律相對於 repo root，而且檔案必須位於 repo 內。 |

`send_prompt` 使用 `.env` 中的 `LLM_BASE_URL`、`MODEL_NAME`、
`TOKEN_LIMIT_PARAMETER_NAME`、`REASONING_EFFORT` 與 `OPENAI_API_KEY`。

`capture_metrics` 會對 vLLM Prometheus endpoint 發送一次 HTTP GET，並保存未解析的
完整 response text：

```json
{
  "id": "M0",
  "action": "capture_metrics",
  "label": "Before A1"
}
```

只要 flow 中包含 `capture_metrics`，就必須透過 `.env` 的
`VLLM_METRICS_URL`、shell 同名環境變數或 `--metrics-url` 提供 endpoint。
命令列參數的優先順序最高，shell 環境變數優先於 `.env`。

完整 AABA 範例請參考
[`flows/prefix-cache-aaba.flow.json`](flows/prefix-cache-aaba.flow.json)；只發送 Prompt
的範例請參考
[`flows/prefix-cache-abc-no-capture.flow.json`](flows/prefix-cache-abc-no-capture.flow.json)。

#### 執行參數與輸出

每次執行會建立 `runs/run_<run-id>.jsonl`。其中包含完整的 Prometheus text、
`send_prefix_prompt.py` JSON response，以及每個步驟的開始、完成或失敗事件。
第一版不解析、不篩選或聚合 metrics；流程遇到失敗會停止，但已寫入的事件仍會保留。

可用命令列暫時覆蓋 metrics endpoint：

```bash
python scripts/run_test_flow.py \
  flows/prefix-cache-aaba.flow.json \
  --metrics-url http://<VLLM_HOST>:8000/metrics
```

其他可用參數：

| 參數 | 預設值 | 用途 |
|---|---:|---|
| `--output-root` | repo 下的 `runs/` | 指定 JSONL 輸出目錄。 |
| `--metrics-timeout` | 10 秒 | 單次 metrics request timeout。 |
| `--prompt-timeout` | 300 秒 | 單次 LLM request timeout。 |

一個成功的 run 依序包含：

```text
run_started
step_started → step_completed
step_started → step_completed
...
run_completed (status: success)
```

若 step 執行失敗，對應事件會是 `step_failed`，最後的 `run_completed` status
會是 `failed`；使用者中斷時則會記錄 `step_cancelled` 與 `cancelled` status。

### 持續收集 CSV

```bash
python scripts/collect_vllm_metrics.py
```

目前 `scripts/collect_vllm_metrics.py` 的 `METRICS_URL` 是程式內常數。
執行 H200 實驗前，請先將它改為實際 vLLM server 的 `/metrics` URL。
腳本每秒取樣一次，並在目前目錄產生 `vllm_metrics_YYYYMMDD_HHMMSS.csv`。

對 H200 實驗，除了 `kv_cache_usage_perc` 外，應優先記錄：

- `vllm:prefix_cache_queries` 與 `vllm:prefix_cache_hits`
- `vllm:external_prefix_cache_queries` 與
  `vllm:external_prefix_cache_hits`（若 middleware/connector 有暴露）
- middleware log、SSD read/write bytes 與 external KV load event

## 判讀原則與限制

- `A2` 的 local prefix hit 應以 prefix-cache counters 為主要證據，不能只看
  `kv_cache_usage_perc`。完成的 request 可能釋放 active blocks，但仍保留可重用
  的 cached block hash。
- A3 是否成為 external/SSD hit，取決於 GPU blocks 是否被回收、middleware 是否
  已將 A 寫入 external cache，以及 token、model、chat template 與 cache namespace
  是否完全相同。
- OpenAI 的 `cached_tokens` 可用來驗證 prompt cache accounting 與腳本流程，但其
  託管架構無法模擬 H200 的 GPU block eviction、aiDAPTIVCache SSD I/O 或 external
  prefix cache 行為。
- H200 的最終結論應以 vLLM `/metrics`、middleware logs、external prefix cache
  counters 與 SSD I/O 為準。

## 驗證

```bash
.venv/bin/python -m py_compile \
  scripts/collect_vllm_metrics.py \
  scripts/send_prefix_prompt.py \
  prompts/generate_long_prompts.py
```
