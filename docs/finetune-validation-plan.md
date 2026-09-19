# Fine-tuning middleware validation plan

## Phase 1: smoke run

使用 `finetune/datasets/fixtures/qa-smoke.json` 與固定少量 iterations，確認資料可讀、
訓練可啟動、log 可取得，且輸出目錄可寫入。

## Phase 2: comparable runs

無 middleware 與有 middleware 的兩組實驗必須固定模型、資料、tokenizer/chat
template、序列長度、有效總 batch、訓練步數與硬體。保留完整 YAML、執行命令、版本、
training log 與輸出檔案清單。

## Metrics

- 最大可穩定執行 batch
- tokens/s 與固定步數完成時間
- GPU、DRAM、aiDAPTIVCache 使用狀態
- 固定 validation 資料上的品質結果

loss 下降只能證明訓練正在進行；不能單獨作為 middleware 效益的證據。
