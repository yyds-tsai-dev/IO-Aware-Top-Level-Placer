# Codex subagents

這兩個角色對應本專案的 `.claude/agents/`，使用 Codex 的獨立 TOML agent 格式。

| 角色 | 模型 | 推理強度 | 用途 |
| --- | --- | --- | --- |
| `fast-worker` | `gpt-5.6-terra` | `low` | 執行已決定的修改、測試、文件同步與資料蒐集 |
| `deep-reasoner` | `gpt-6-astra` | `high` | 架構、數值正確性與根因分析，回傳可執行方案 |

`fast-worker` 繼承主代理的權限設定；`deep-reasoner` 設定 `read-only`，指示它只分析與執行不修改檔案的探針。
主代理當前的 runtime 權限覆寫仍可能優先於 agent 檔案的 sandbox 設定。

## 使用方式

從此專案開啟新的 Codex 工作階段，明確指定角色，例如：

```text
請使用 deep-reasoner 子代理分析 evaluator_gpu 與 evaluator_ref 的數值差異，
提供 file:line 證據與修正方案。
```

```text
請將以下已確定的修改交給 fast-worker 子代理，限定修改列出的檔案，
執行指定的 pytest 命令並回報結果：……
```

交付子任務時，附上問題、檔案範圍、預期輸出與驗證命令。依專案的
`AGENTS.md`，主代理先查詢 graph 與 coverage，再附上 evidence tier、project、
generation/freshness、查詢與分頁進度、符號及路徑、call chain、coverage 缺口、
已完成的 source fallback 與待解問題。子代理若沒有 MCP，使用提供的證據與原始碼，
並明確說明限制。

目前格式會從 `.codex/agents/*.toml` 載入角色，無須額外的 `config.toml` 角色註冊。
若既有設定停用了 agents，需將既有 `[agents]` 區段的 `enabled` 設為 `true`。
模型名稱與推理強度可直接在各 TOML 修改；刪除 `model` 可繼承主代理或全域預設模型。
模型須由使用中的 Codex 帳號與 provider 支援。

## H100 環境

本機 DREAMPlace 位於 `/ldaphome/yyds-tsai-dev/DREAMPlace`。從專案根目錄執行
`source src/scripts/env.sh`，再以 `"$IOPLACE_PYTHON" -m pytest` 執行測試；腳本也會
匯出 `DREAMPLACE_ROOT`。兩個 Claude agent 與兩個 Codex agent 都使用此設定。
可預先設定 `DREAMPLACE_ROOT` 或 `IOPLACE_PYTHON` 來覆寫路徑。
GPU 工作保留呼叫端的 `CUDA_VISIBLE_DEVICES`；使用前先確認 H100 的佔用狀況。
安裝與驗證細節見 [docs/dev-env.md](../../docs/dev-env.md)。

格式與設定依據：[OpenAI 官方 Subagents 文件](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents)。
