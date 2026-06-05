<div align="center">
  <img src="images/logo.png" alt="logo" width="120" />

  # AI-Friend

  以本地大型語言模型打造的 AI 虛擬主播 / AI 好友

  <p>
    <img src="https://img.shields.io/badge/status-archived-lightgrey" alt="archived" />
    <img src="https://img.shields.io/badge/python-3.10+-blue" alt="python" />
    <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="license" />
  </p>
</div>

---

> 🗄️ **此專案已封存（Archived）**
> 本專案已停止維護，不再接受更新或修正，僅作為開發紀錄與學習參考保留。
> 以下文件描述其最後一次可運行的狀態；相依套件版本可能已過時，實際執行請自行調整。

---

## 簡介

AI-Friend 是一個本地運行的 AI 聊天角色。
專案以 [Streamlit](https://streamlit.io/) 提供網頁聊天介面，整條對話鏈由
[LangChain](https://www.langchain.com/) 串接，並用以下開源元件組成：

| 功能 | 使用元件 |
|------|----------|
| 大型語言模型 | [Ollama](https://ollama.com/)（本地端，預設 `wangshenzhi/llama3.1_8b_chinese_chat`） |
| 文字轉語音 (TTS) | [ChatTTS](https://github.com/2noise/ChatTTS) |
| 聯網搜尋 | DuckDuckGo（透過 LangChain 工具） |
| 對話記憶 | LangChain `ConversationBufferMemory` |
| 虛擬形象驅動（選用） | [VTube Studio](https://denchisoft.com/) API（`vtube_studio.py`） |

回覆會即時以 ChatTTS 合成語音，並在網頁中自動播放。

> ⚠️ **聲明**：本專案為個人實驗性 Demo，程式碼品質僅供參考；若要實際應用請自行依規範完善。

---

## 技術架構

整個應用是一支 Streamlit 腳本（`main.py`），單次對話的資料流如下：

```
使用者輸入 (st.chat_input)
        │
        ▼
  套上人設提示詞 (Role_setting)
        │
        ▼
  ConversationalChatAgent  ──►  工具：DuckDuckGoSearchRun（需要時聯網搜尋）
   (AgentExecutor, 最多 6 次迭代)
        │  ▲
        │  └── ConversationBufferMemory（StreamlitChatMessageHistory 持久化）
        ▼
   ChatOllama 串流回覆  ──►  StreamlitCallbackHandler 即時顯示思考過程
        │
        ▼
  以「。」切句 → 逐句送入 get_voice()
        │
        ▼
  ChatTTS 合成語音 (24 kHz wav, speaker_6.pth 音色)
        │
        ▼
  base64 內嵌 <audio autoplay> 自動播放，並 sleep 該句長度避免重疊
```

**關鍵技術點**

- **LLM 推論**：透過 `langchain_community.chat_models.ChatOllama` 連本地 Ollama，
  預設模型 `wangshenzhi/llama3.1_8b_chinese_chat`，開啟 `streaming=True` 串流輸出。
- **Agent 與工具**：`ConversationalChatAgent.from_llm_and_tools` 搭配 `AgentExecutor`
  （`max_iterations=6`、`handle_parsing_errors=True`），可在對話中自行決定是否呼叫
  DuckDuckGo 搜尋取得即時資訊。
- **記憶**：`ConversationBufferMemory(memory_key="chat_history")`，對話歷史以
  `StreamlitChatMessageHistory` 存放，於 Streamlit 重跑之間保留；側邊欄提供記憶開關與輪數調整。
- **語音合成**：`ChatTTS` 於每次回覆時 `load()` 模型，先 `refine_text` 再 `infer`；
  音色由 `speaker/speaker_6.pth` 的 speaker embedding 決定，推論參數
  `temperature=0.3 / top_P=0.7 / top_K=20`，refine 提示 `[oral_1][laugh_1][break_1]`。
  > 📜 **語音方案演進**：早期版本是使用**自行訓練的 [bert-vits2](https://github.com/fishaudio/Bert-VITS2) 模型**來合成語音，
  > 後來為了簡化部署與訓練流程，改用免訓練、以 speaker embedding 控制音色的 ChatTTS（即現行方案）。
  > 目前 repo 內已不保留 bert-vits2 的訓練程式碼與模型權重。
- **播放**：合成的 wav 以 `soundfile` 寫入 `./output/`，再以 base64 內嵌成 HTML5
  `<audio autoplay>` 在頁面播放（採用瀏覽器播放，主程式不需要 mpv）。
- **虛擬形象（選用）**：`vtube_studio.py` 為獨立的 VTube Studio WebSocket API 控制器，
  封裝認證與 `InjectParameterDataRequest`，可用來依語音音量驅動 Live2D 形象的嘴型／動作。

---

## 環境需求

- **Python 3.10+**
- **[Ollama](https://ollama.com/)**——需先安裝並下載對話模型
- 建議使用 **GPU**（Apple Silicon 的 MPS 或 NVIDIA CUDA），ChatTTS 在純 CPU 上會很慢
- 首次執行 ChatTTS 會自動下載語音模型（需網路）

---

## 安裝與啟動

```bash
# 1. 取得專案
git clone https://github.com/yuhuanowo/AI-Friend.git
cd AI-Friend

# 2. 安裝 Python 依賴（建議使用虛擬環境）
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. 安裝並啟動 Ollama，下載對話模型
ollama pull wangshenzhi/llama3.1_8b_chinese_chat

# 4. 建立語音輸出資料夾（main.py 會把合成的 wav 寫到這裡）
mkdir -p output

# 5. 啟動網頁介面
streamlit run main.py
```

啟動後瀏覽器會開啟聊天頁面，即可對話。側邊欄可開關**會話記憶**並調整**記憶輪數**。

---

## 專案結構

```
AI-Friend/
├── main.py              # 主程式：Streamlit 網頁聊天 + ChatTTS 語音（專案入口）
├── vtube_studio.py      # VTube Studio API 控制器（選用，驅動虛擬形象）
├── requirements.txt     # Python 依賴
├── Role_setting.txt     # 角色人設提示詞
├── Role_setting2.txt    # 備用人設提示詞
├── speaker/
│   └── speaker_6.pth     # ChatTTS 音色嵌入
├── images/               # logo 與截圖
├── output/               # 語音輸出（執行時產生，已被 .gitignore）
└── examples/             # 歷史 / 實驗性程式碼，僅供參考（見下方）
```

### `examples/` 內容

這些是早期開發過程的實驗腳本與舊版實作，**非主程式的一部分**，僅保留作為演進紀錄與參考：

| 檔案 | 說明 |
|------|------|
| `old.py` | **第一代主程式**（約 700 行，Flask Web 服務 + 多執行緒架構）。與現行 Streamlit 版差異很大：直接用 `transformers` 在本地載入 `hfl/llama-3-chinese-8b-instruct` 推論（非 Ollama）；`faster-whisper` 做語音辨識（STT）；`pyaudio` + `keyboard` 按鍵錄音；`apscheduler` 排程；用 `subprocess` 呼叫 `mpv` 播放合成語音；並含 `dashscope`、`ChatGLM`、`GoogleGenerativeAI` 等多種 LLM 後端的嘗試。需搭配 `config.py` 與 `templates/`。 |
| `config.py` | 舊版的設定 stub，用來切換 LLM 後端（Local / openai / langchain_chat），僅供 `old.py` 使用。 |
| `templates/index.html` | `old.py` Flask 版的前端聊天頁（含 `/send_message` 的 fetch 表單）。 |
| `tts-chattts.py` | **ChatTTS 獨立合成範例**：載入 `speaker/speaker_6.pth` 音色、設定 refine／infer 參數並輸出 wav。現行 `main.py` 的語音模組即由此演化而來。 |
| `tts-bark.py` | 改用 [Suno Bark](https://github.com/suno-ai/bark)（`suno/bark`）模型做 TTS 的實驗，啟用 Apple MPS 加速。 |
| `test.py` | 以 `transformers` 的 `HuggingFacePipeline` 在本地（MPS／CUDA）跑 `hfl/llama-3-chinese-8b-instruct-v3` 的 `LLMChain` 對話測試。 |
| `test2.py` | 以 `ChatOllama`（llama3.1）跑 `LLMChain` 的對話測試片段。 |
| `try.py` | **現行 `main.py` 的雛形**：改寫自 LangChain 官方範例的 `ConversationalChatAgent` + `ChatOllama`(qwen2) + DuckDuckGo 搜尋 Streamlit 樣板。 |

> 註：`old.py` 依賴 `mpv` 播放器。本專案不再將 `mpv.exe` / `mpv.app` 提交進 git，
> 如需執行舊版請[自行安裝 mpv](https://mpv.io/installation/)。

---

## 授權

本專案採用 [Apache License 2.0](LICENSE)。
