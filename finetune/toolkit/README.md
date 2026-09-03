# aiDAPTIV Toolkit

手冊列出的 Toolkit 版本為 `toolkit_v2.3.1_python312`。

Toolkit 是選用的 batch-size 與效能測試工具；實際 fine-tuning 仍由群聯容器中的
訓練腳本執行。

## 測試指令

先在 `project.ini` 設定模型路徑、aiDAPTIVCache 路徑、batch 範圍與測試時間，再進入：

```bash
cd toolkit_v2.3.1_python312/Script/Model_Test
```

| 指令 | 用途 | batch size |
| --- | --- | --- |
| `python3 aidaptest_run.py --t 1` | 效能測試 | 直接使用 `project.ini` 的 `start_bs` |
| `python3 aidaptest_run.py --t 2` | 尋找最大可執行 batch size | 從 `start_bs` 掃至 `end_bs` |
| `python3 aidaptest_run.py --t 3` | 先找最大 batch，再做效能測試 | 使用尋得的最大 batch |

`--t 1` 會產生 GPU／DRAM 使用量、forward／backward／update 時間、loss 與 tokens/s 等
效能資料；它不會尋找最大 batch，也不是正式 fine-tuning 產出模型的流程。結果會寫入
`toolkit_v2.3.1_python312/Log/`，包括圖表、訓練 log 與 `performance_result.xlsx`。
