# Dataset layout

- `fixtures/`：可提交 Git 的小型、無敏感 smoke-test 資料。用來確認資料欄位、YAML
  mapping、container mount 與訓練啟動流程。
- `local/`：實際 fine-tuning 資料；此處內容會被 Git 忽略。請依群聯設定檔指定的
  欄位與格式準備資料。

目前 `qa-smoke.json` 對應 `qa_dataset_config.example.yaml` 的 `question` 與 `answer`
欄位。它只能驗證流程可啟動，不可用於品質或效能結論。
