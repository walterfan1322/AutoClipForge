# AutoClipForge

Automated video creation tool with Chinese TTS, AI image generation, and a web-based GUI.

> Forked from [MoneyPrinterV2](https://github.com/FujiwaraChoki/MoneyPrinterV2) with major enhancements for Chinese-language content creation.

## What's Different

| Feature | Original (MoneyPrinterV2) | AutoClipForge |
|---------|--------------------------|---------------|
| TTS Engine | KittenTTS (English) | **Edge-TTS + gTTS + espeak** with zh-TW support |
| Video Pipeline | Basic YouTube Shorts | Rewritten with subtitle sync, smart chunking, pause control |
| Image Generation | NanoBanana2 only | NanoBanana2 + **Flux image service** |
| GUI | None | **Web-based GUI** (4600+ lines) with real-time progress |
| Content Types | Generic shorts | Market reports, tech news, GitHub weekly, international briefs |
| LLM Backend | GPT4Free | **Ollama** (local) + MiniMax API |
| Language | English | **Traditional Chinese (zh-TW)** first, English supported |

## Features

- **Web GUI** — browser-based control panel for video creation, account management, and artifact browsing
- **Chinese TTS** — Edge-TTS with `zh-TW-HsiaoChenNeural`, smart punctuation-aware sentence splitting, configurable speed/pitch
- **AI Image Generation** — Gemini (NanoBanana2) + optional self-hosted Flux service
- **Smart Subtitles** — auto-generated `.srt` with CJK-aware timing via local Whisper or AssemblyAI
- **Market Report Videos** — paste market data, auto-generate structured voiceover scripts
- **Content Normalization** — LLM-powered pipeline to convert raw news/reports into structured video scripts
- **YouTube Upload** — automated upload via Firefox profile with Selenium
- **Artifact Management** — track generated audio, subtitles, and videos; auto-cleanup old files

## Quick Start

### Prerequisites

- Python 3.12+
- FFmpeg
- ImageMagick
- Ollama (for local LLM)

### Installation

```bash
git clone https://github.com/walterfan1322/AutoClipForge.git
cd AutoClipForge

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Set up config
cp config.example.json config.json
# Edit config.json with your settings
```

### Configuration

Edit `config.json`:

```jsonc
{
  "ollama_base_url": "http://127.0.0.1:11434",
  "ollama_model": "qwen3.5:9b",
  "nanobanana2_api_key": "",          // Gemini API key for image gen
  "flux_api_base_url": "",            // Optional: self-hosted Flux service
  "assembly_ai_api_key": "",          // Optional: AssemblyAI for subtitles
  "tts_voice": "zh-TW-HsiaoChenNeural"
}
```

For API keys, you can also use environment variables:

```bash
export MINIMAX_API_KEY="your-key"
export FLUX_REMOTE_HOST="user@host"
export GEMINI_API_KEY="your-key"
```

### Usage

**GUI mode (recommended):**

```bash
python gui_app.py
# Open http://localhost:5000 in your browser
```

**CLI mode:**

```bash
python src/main.py
```

## Project Structure

```
AutoClipForge/
├── gui_app.py              # Web GUI (Flask)
├── src/
│   ├── main.py             # CLI entry point
│   ├── classes/
│   │   ├── Tts.py          # TTS engine (Edge-TTS/gTTS/espeak)
│   │   ├── YouTube.py      # Video generation & upload pipeline
│   │   ├── Twitter.py      # Twitter bot
│   │   ├── AFM.py          # Affiliate marketing
│   │   └── Outreach.py     # Email outreach
│   ├── config.py           # Configuration loader
│   ├── llm_provider.py     # Ollama integration
│   └── utils.py            # Utilities
├── config.example.json     # Config template (no secrets)
├── scripts/                # Helper scripts
├── docs/                   # Documentation
└── tests/                  # Test suite
```

## Acknowledgments

- [MoneyPrinterV2](https://github.com/FujiwaraChoki/MoneyPrinterV2) — original project
- [Edge-TTS](https://github.com/rany2/edge-tts) — Microsoft Edge TTS engine
- [Ollama](https://ollama.com) — local LLM runtime

## License

[AGPL-3.0](LICENSE)

