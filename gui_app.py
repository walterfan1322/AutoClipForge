import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from PIL import Image as PILImage

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_PATH = ROOT_DIR / "config.json"
MP_DIR = ROOT_DIR / ".mp"
RESULT_STATE_PATH = MP_DIR / "gui_last_result.json"
FILE_REGISTRY_PATH = MP_DIR / "_file_registry.json"

# English names used for download filenames
CONTENT_ENGLISH_MAP = [
    ("國際", "World News Report"),
    ("International", "World News Report"),
    ("GitHub", "GitHub Weekly"),
    ("科技", "Tech News Report"),
    ("Tech", "Tech News Report"),
    ("Market", "Market Report"),
    ("market", "Market Report"),
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if not hasattr(PILImage, "ANTIALIAS") and hasattr(PILImage, "Resampling"):
    PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS

from cache import get_accounts, get_youtube_cache_path  # noqa: E402
from classes.Tts import TTS  # noqa: E402
from classes.YouTube import YouTube  # noqa: E402
from config import get_ollama_base_url, get_ollama_model  # noqa: E402
from llm_provider import select_model  # noqa: E402
from utils import rem_temp_files  # noqa: E402

app = Flask(__name__)

SECTION_KEY_MAP = {
    "做什麼": "what",
    "為什麼變熱門": "why_hot",
    "為什麼爆紅": "why_hot",
    "適合誰": "who_for",
    "風險與門檻": "risks",
    "風險 / 門檻": "risks",
    "風險/門檻": "risks",
}


@app.after_request
def disable_response_caching(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

JOB_LOCK = threading.Lock()
JOB_STATE = {
    "running": False,
    "status": "idle",
    "progress": 0,
    "stage": "Idle",
    "started_at": None,
    "finished_at": None,
    "account_id": None,
    "account_name": None,
    "output_mode": "full_video",
    "subject": "",
    "title": "",
    "description": "",
    "script": "",
    "script_path": None,
    "audio_path": None,
    "subtitle_path": None,
    "image_paths": [],
    "video_path": None,
    "uploaded_url": None,
    "error": None,
    "logs": [],
}
PREVIEW_STATE = {
    "content_mode": "auto",
    "output_mode": "full_video",
    "subject": "",
    "title": "",
    "description": "",
    "script": "",
    "segments": [],
    "scene_plan": [],
    "account_id": "",
    "account_name": "",
    "created_at": "",
}

FLUX_REMOTE_HOST = os.environ.get("FLUX_REMOTE_HOST", "")
FLUX_REMOTE_DIR = os.environ.get("FLUX_REMOTE_DIR", "")
FLUX_SSH_KEY = Path.home() / ".ssh" / "id_ed25519_ollama_tunnel"
FLUX_HEALTH_URL = "http://127.0.0.1:18081/health"
NORMALIZER_PRIMARY_MODEL = "qwen3:14b"
NORMALIZER_FALLBACK_MODEL = "qwen3.5:35b-a3b"
VOICEOVER_PRIMARY_MODEL = "qwen3:14b"
VOICEOVER_FALLBACK_MODEL = "qwen3.5:35b-a3b"
GITHUB_VOICEOVER_PRIMARY_MODEL = "qwen3.5:35b-a3b"
GITHUB_VOICEOVER_FALLBACK_MODEL = "qwen3:14b"
TECHNEWS_VOICEOVER_PRIMARY_MODEL = "qwen3.5:35b-a3b"
TECHNEWS_VOICEOVER_FALLBACK_MODEL = "qwen3:14b"
MARKET_VOICEOVER_PRIMARY_MODEL = "qwen3.5:35b-a3b"
MARKET_VOICEOVER_FALLBACK_MODEL = "qwen3:14b"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(message: str) -> None:
    JOB_STATE["logs"].append(f"[{now_text()}] {message}")
    JOB_STATE["logs"] = JOB_STATE["logs"][-300:]


def flux_health_ready() -> bool:
    try:
        with urllib_request.urlopen(FLUX_HEALTH_URL, timeout=5) as response:
            return response.status == 200
    except (urllib_error.URLError, TimeoutError, OSError):
        return False


def run_flux_remote(command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    if not FLUX_SSH_KEY.exists():
        raise RuntimeError(f"FLUX SSH key not found: {FLUX_SSH_KEY}")
    ssh_cmd = [
        "ssh",
        "-i",
        str(FLUX_SSH_KEY),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        FLUX_REMOTE_HOST,
        command,
    ]
    return subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def ensure_flux_service_running() -> None:
    if flux_health_ready():
        append_log("FLUX image service already reachable.")
        return

    append_log("Starting FLUX image service on remote host ...")
    run_flux_remote(f"cd {FLUX_REMOTE_DIR} && docker compose up -d flux-image-service", timeout=180)

    deadline = time.time() + 180
    while time.time() < deadline:
        if flux_health_ready():
            append_log("FLUX image service is ready.")
            return
        time.sleep(3)
    raise RuntimeError("FLUX image service did not become ready in time.")


def stop_flux_service() -> None:
    append_log("Stopping FLUX image service to release GPU memory ...")
    try:
        run_flux_remote(f"cd {FLUX_REMOTE_DIR} && docker compose stop flux-image-service", timeout=180)
        deadline = time.time() + 60
        while time.time() < deadline:
            if not flux_health_ready():
                append_log("FLUX image service stopped.")
                return
            time.sleep(2)
        append_log("FLUX stop requested, but health endpoint still responds.")
    except Exception as exc:
        append_log(f"Failed to stop FLUX image service cleanly: {exc}")


def set_stage(stage: str, progress: int) -> None:
    JOB_STATE["stage"] = stage
    JOB_STATE["progress"] = progress
    append_log(f"{progress}% - {stage}")


def read_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def write_config(data: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_accounts() -> list[dict]:
    return get_accounts("youtube")


def write_accounts(accounts: list[dict]) -> None:
    cache_path = Path(get_youtube_cache_path())
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump({"accounts": accounts}, file, indent=4, ensure_ascii=False)


def find_account(account_id: str) -> dict | None:
    for account in read_accounts():
        if account["id"] == account_id:
            return account
    return None


def save_result_state(payload: dict) -> None:
    MP_DIR.mkdir(exist_ok=True)
    with open(RESULT_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
#  File registry – maps UUID filenames to human-friendly download names
# ---------------------------------------------------------------------------

def _load_file_registry() -> dict:
    if FILE_REGISTRY_PATH.exists():
        try:
            return json.load(open(FILE_REGISTRY_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_file_registry(registry: dict) -> None:
    MP_DIR.mkdir(exist_ok=True)
    with open(FILE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _subject_to_english(subject: str) -> str:
    for keyword, english in CONTENT_ENGLISH_MAP:
        if keyword.lower() in subject.lower():
            return english
    return subject[:30] if subject else "Output"


def register_artifacts(subject: str, created_at: str, **paths: str) -> None:
    """Register generated files with human-friendly download names.

    Example::

        register_artifacts(
            "國際情勢報告", "2026-04-08 22:59",
            video="/path/to/uuid.mp4",
            audio="/path/to/uuid.wav",
            subtitle="/path/to/uuid.srt",
        )
    """
    registry = _load_file_registry()
    date_part = ""
    if created_at:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", created_at)
        if m:
            date_part = m.group(1)[2:] + m.group(2) + m.group(3)

    english = _subject_to_english(subject)
    friendly_prefix = f"{english} {date_part}".strip()

    type_labels = {"video": "", "audio": " Audio", "subtitle": " Subtitle"}
    for label, filepath in paths.items():
        if not filepath or not os.path.exists(str(filepath)):
            continue
        basename = os.path.basename(filepath)
        _, ext = os.path.splitext(basename)
        suffix = type_labels.get(label, f" {label.title()}")
        registry[basename] = {
            "friendly": f"{friendly_prefix}{suffix}{ext}",
            "subject": subject,
            "created": created_at,
            "type": label,
        }
    _save_file_registry(registry)


def get_friendly_download_name(basename: str) -> str:
    registry = _load_file_registry()
    entry = registry.get(basename)
    if entry and entry.get("friendly"):
        return entry["friendly"]
    return basename


def cleanup_old_artifacts(keep_latest: int = 3) -> dict:
    """Remove old artifacts, keeping only files from the *keep_latest* most
    recent generation runs (identified by their .mp4 modification time).

    Returns a summary dict with counts and freed bytes.
    """
    if not MP_DIR.exists():
        return {"removed": 0, "freed_kb": 0}

    # Identify the keep_latest most recent mp4 timestamps
    mp4s = sorted(MP_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(mp4s) <= keep_latest:
        return {"removed": 0, "freed_kb": 0, "message": "Nothing to clean up."}

    # Keep files newer than the oldest kept mp4
    cutoff = mp4s[keep_latest - 1].stat().st_mtime if keep_latest > 0 else float("inf")

    protected = {RESULT_STATE_PATH.name, FILE_REGISTRY_PATH.name, "youtube.json"}
    removed = 0
    freed = 0
    registry = _load_file_registry()
    new_registry = {}

    for path in MP_DIR.iterdir():
        if path.name in protected:
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".wav", ".mp3", ".srt", ".mp4", ".json", ".txt"}:
            continue
        if path.name.endswith(".timing.json"):
            # Remove timing files older than cutoff
            if path.stat().st_mtime < cutoff:
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
                removed += 1
            continue
        try:
            if path.stat().st_mtime < cutoff:
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
                removed += 1
            else:
                if path.name in registry:
                    new_registry[path.name] = registry[path.name]
        except Exception:
            pass

    _save_file_registry(new_registry)
    return {"removed": removed, "freed_kb": round(freed / 1024, 1)}


def compose_video_with_subtitles(youtube: YouTube, subtitles_path: str) -> str:
    from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
    from moviepy.video.fx.all import crop

    if not getattr(youtube, "tts_path", None) or not os.path.exists(youtube.tts_path):
        raise RuntimeError("TTS audio is missing.")
    if not getattr(youtube, "images", None):
        raise RuntimeError("No generated images available for composition.")

    output_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
    temp_audio_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.m4a")

    tts_clip = AudioFileClip(youtube.tts_path)
    max_duration = max(float(tts_clip.duration or 0), 0.1)
    req_dur = max_duration / max(len(youtube.images), 1)

    clips = []
    total_duration = 0.0
    while total_duration < max_duration:
        for image_path in youtube.images:
            clip = ImageClip(image_path).set_duration(req_dur).set_fps(30)
            if round((clip.w / clip.h), 4) < 0.5625:
                clip = crop(
                    clip,
                    width=clip.w,
                    height=round(clip.w / 0.5625),
                    x_center=clip.w / 2,
                    y_center=clip.h / 2,
                )
            else:
                clip = crop(
                    clip,
                    width=round(0.5625 * clip.h),
                    height=clip.h,
                    x_center=clip.w / 2,
                    y_center=clip.h / 2,
                )
            clip = clip.resize((1080, 1920))
            clips.append(clip)
            total_duration += float(clip.duration or 0)
            if total_duration >= max_duration:
                break

    final_clip = concatenate_videoclips(clips).set_fps(30)
    final_clip = final_clip.set_audio(tts_clip.set_fps(44100)).set_duration(max_duration)

    subtitle_clips = []
    if subtitles_path and os.path.exists(subtitles_path):
        subtitle_clips = youtube._build_subtitle_overlays(subtitles_path, video_size=(1080, 1920))
    if subtitle_clips:
        final_clip = CompositeVideoClip([final_clip, *subtitle_clips])

    final_clip.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=temp_audio_path,
        remove_temp=True,
        ffmpeg_params=["-movflags", "+faststart"],
    )

    try:
        final_clip.close()
    except Exception:
        pass
    try:
        tts_clip.close()
    except Exception:
        pass
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass
    for clip in subtitle_clips:
        try:
            clip.close()
        except Exception:
            pass

    return output_path


def write_text_artifact(prefix: str, suffix: str, content: str) -> str:
    MP_DIR.mkdir(exist_ok=True)
    path = MP_DIR / f"{prefix}-{uuid4()}{suffix}"
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def load_result_state() -> dict | None:
    if not RESULT_STATE_PATH.exists():
        return None
    with open(RESULT_STATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def snapshot_job_state() -> dict:
    with JOB_LOCK:
        job = dict(JOB_STATE)
    return {
        "running": bool(job.get("running")),
        "status": job.get("status") or "idle",
        "account_name": job.get("account_name") or "-",
        "stage": job.get("stage") or "-",
        "progress": int(job.get("progress") or 0),
        "output_mode": job.get("output_mode") or "",
        "started_at": job.get("started_at") or "-",
        "finished_at": job.get("finished_at") or "-",
        "uploaded_url": job.get("uploaded_url") or "-",
        "error": job.get("error") or "",
        "logs": list(job.get("logs") or []),
    }


def list_artifacts() -> list[dict]:
    artifacts = []
    if not MP_DIR.exists():
        return artifacts

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return -1

    for path in sorted(MP_DIR.iterdir(), key=_mtime, reverse=True):
        if path.name in {RESULT_STATE_PATH.name, FILE_REGISTRY_PATH.name}:
            continue
        if path.name.endswith(".timing.json"):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".wav", ".mp3", ".srt", ".mp4", ".json", ".txt"}:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        artifacts.append(
            {
                "name": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "is_image": path.suffix.lower() in {".png", ".jpg", ".jpeg"},
                "is_video": path.suffix.lower() == ".mp4",
            }
        )
    return artifacts


def _ollama_base_url() -> str:
    try:
        config_data = read_config()
    except Exception:
        config_data = {}
    return (config_data.get("ollama_base_url") or get_ollama_base_url() or "http://127.0.0.1:11434").rstrip("/")


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _call_ollama_json(models: tuple[str, ...], prompt: str, timeout: int = 240, temperature: float = 0.1) -> dict | None:
    base_url = _ollama_base_url()
    for model in models:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            req = urllib_request.Request(
                f"{base_url}/api/generate",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
            parsed = _extract_json_object(body.get("response", ""))
            if parsed:
                return parsed
        except Exception:
            continue
    return None


def _normalize_item(item: dict, default_rank: int) -> dict:
    rank = item.get("rank", default_rank)
    try:
        rank = int(rank)
    except Exception:
        rank = default_rank
    return {
        "rank": rank,
        "name": normalize_script(str(item.get("name", "") or "")),
        "what": normalize_script(str(item.get("what", "") or item.get("summary", "") or "")),
        "why_hot": normalize_script(str(item.get("why_hot", "") or item.get("why_it_matters", "") or "")),
        "who_for": normalize_script(str(item.get("who_for", "") or "")),
        "risks": normalize_script(str(item.get("risk", "") or item.get("risks", "") or "")),
    }


def _convert_normalized_payload(normalized: dict, source_text: str = "") -> dict | None:
    items = normalized.get("items")
    if not isinstance(items, list) or not items:
        return None

    content_type = str(normalized.get("content_type", "") or "").strip().lower()
    title = normalize_script(str(normalized.get("title", "") or ""))
    summary = normalize_script(str(normalized.get("summary", "") or ""))
    normalized_items = [_normalize_item(item or {}, idx) for idx, item in enumerate(items, start=1)]

    if "market" in content_type:
        themes = []
        for item in normalized_items[:6]:
            themes.append(
                {
                    "region": normalize_script(str((items[item["rank"] - 1] or {}).get("region", "") or "Market")),
                    "slot": f"Theme {item['rank']}",
                    "title": item["name"] or f"Theme {item['rank']}",
                    "summary": item["what"] or item["why_hot"],
                    "bullets": [value for value in [item["what"], item["why_hot"], item["risks"]] if value],
                }
            )
        return {
            "content_type": "market_report",
            "header": title or "Market report",
            "summary": summary,
            "themes": themes,
        }

    if "news" in content_type or "tech" in content_type:
        events = []
        for item in normalized_items[:6]:
            events.append(
                {
                    "rank": item["rank"],
                    "title": item["name"] or f"Headline {item['rank']}",
                    "importance": "",
                    "summary": item["what"] or item["why_hot"],
                    "highlights": [value for value in [item["what"], item["why_hot"], item["risks"]] if value],
                }
            )
        return {
            "content_type": "daily_tech_news",
            "header": title or "Tech news",
            "summary": summary,
            "events": events,
        }

    canonical_projects = {}
    if source_text:
        try:
            for project in parse_ranked_projects(source_text).get("projects", []):
                canonical_projects[int(project.get("rank", 0))] = project
        except Exception:
            canonical_projects = {}

    projects = []
    raw_items = items[: len(normalized_items)]
    for item, raw_item in zip(normalized_items, raw_items):
        canonical = canonical_projects.get(item["rank"], {})
        projects.append(
            {
                "rank": item["rank"],
                "name": canonical.get("name") or item["name"] or f"Item {item['rank']}",
                "link": normalize_script(str(canonical.get("link", "") or (raw_item or {}).get("link", "") or "")),
                "language": normalize_script(str(canonical.get("language", "") or (raw_item or {}).get("language", "") or "")),
                "stats": normalize_script(str(canonical.get("stats", "") or (raw_item or {}).get("stats", "") or "")),
                "what": item["what"] or normalize_script(str(canonical.get("what", "") or "")),
                "why_hot": item["why_hot"] or normalize_script(str(canonical.get("why_hot", "") or "")),
                "who_for": item["who_for"] or normalize_script(str(canonical.get("who_for", "") or "")),
                "risks": item["risks"] or normalize_script(str(canonical.get("risks", "") or "")),
            }
        )
    return {
        "content_type": "github_weekly",
        "header": title or "GitHub weekly",
        "summary": summary,
        "projects": projects,
    }


def _try_llm_normalize(source_text: str) -> dict | None:
    cleaned = source_text.replace("\ufeff", "").strip()
    if len(cleaned) < 80:
        return None

    prompt = f"""
You are a content normalizer for short-form video generation.

Read the source content and convert it into a single JSON object.
Rules:
- Output JSON only. No markdown fences. No explanation.
- Write title, summary, what, why_hot, who_for, risks, and region in Traditional Chinese by default.
- Only keep English for proper nouns, repo names, company names, product names, stock tickers, or quoted headlines that must stay in English.
- Preserve project names, company names, casing, punctuation, parentheses, hyphens, and symbols exactly when they appear in the source.
- Preserve numeric stats and weekly gain formatting when available.
- Never invent, estimate, extrapolate, or “fill in” facts, numbers, growth rates, timeframes, rankings, or financial metrics.
- If a detail is not explicitly stated in the source, leave it empty or summarize only what is clearly present.
- Do not turn a vague statement into a precise metric.
- Do not add outside knowledge.
- Choose content_type from exactly: github_weekly, daily_tech_news, market_report, generic
- Classify by the real document intent, not by incidental numbering.
- If the document is a stock, macro, market, sector, Taiwan market, US market, oil, rates, AI supply chain, TSMC, or watch-list report, choose market_report.
- If the document is a technology news digest with multiple events or headlines, choose daily_tech_news.
- Only choose github_weekly when the document is mainly about GitHub/open-source projects, repos, stars, forks, maintainers, or weekly project rankings.
- Do not misclassify market reports as github_weekly just because the text contains numbered lists like 1. 2. 3.
- For market_report, each item name should be a short Traditional Chinese market theme headline, not an English sentence.
- For market_report, region should be one of: US Markets, Taiwan Markets, Global.
- For daily_tech_news, each item name should be the news headline in Traditional Chinese, unless the headline is primarily a product or project name.
- Return 3 to 6 items when possible.
- Each item should include:
  - rank: integer
  - name: short item title
  - what: what it is / what happened
  - why_hot: why it matters / why it is notable
  - who_for: optional
  - risks: optional
  - stats: optional
  - link: optional
  - language: optional
  - region: optional

JSON schema:
{{
  "content_type": "github_weekly|daily_tech_news|market_report|generic",
  "title": "short title",
  "summary": "short summary",
  "items": [
    {{
      "rank": 1,
      "name": "item name",
      "what": "what it is",
      "why_hot": "why it matters",
      "who_for": "",
      "risks": "",
      "stats": "",
      "link": "",
      "language": "",
      "region": ""
    }}
  ]
}}

Source:
{cleaned}
""".strip()

    normalized = _call_ollama_json(
        (NORMALIZER_PRIMARY_MODEL, NORMALIZER_FALLBACK_MODEL),
        prompt,
        timeout=240,
        temperature=0.1,
    )
    parsed = _convert_normalized_payload(normalized or {}, cleaned)
    if parsed:
        return parsed
    return None


def _voiceover_prompt_type(content_type: str) -> str | None:
    if content_type == "github_weekly":
        return "GitHub 開源項目週報"
    if content_type == "daily_tech_news":
        return "每日科技新聞"
    if content_type == "market_report":
        return "市場主題報告"
    return None


def _voiceover_has_required_refs(script: str, parsed: dict) -> bool:
    content_type = parsed.get("content_type", "generic")
    if content_type == "github_weekly":
        required = [project.get("name", "") for project in (parsed.get("projects") or [])[:2]]
        return all(name and name in script for name in required)
    if content_type == "daily_tech_news":
        return len(script) >= 60 and ("新聞" in script or "今天" in script or "重點" in script)
    if content_type == "market_report":
        return len(script) >= 60 and any(token in script for token in ("市場", "美股", "台股", "資金", "科技股"))
    return len(script) >= 60


def _merge_script_group(lines: list[str]) -> str:
    cleaned = [clean_field_text(line).rstrip("。！？!?") for line in lines if clean_field_text(line)]
    if not cleaned:
        return ""
    merged = "，".join(cleaned)
    if merged and not re.search(r"[。！？!?]$", merged):
        merged += "。"
    return merged


def _normalize_voiceover_script(script: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in str(script or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    normalized_lines = []
    for line in lines:
        cleaned_line = clean_field_text(line)
        if cleaned_line and not re.search(r"[。！？!?]$", cleaned_line):
            cleaned_line += "。"
        if cleaned_line:
            normalized_lines.append(cleaned_line)
    return "\n".join(normalized_lines), normalized_lines


def _try_llm_github_weekly_voiceover_v1(parsed: dict, custom_subject: str = "") -> tuple[str, str] | None:
    projects = parsed.get("projects") or []
    if not projects:
        return None

    compact_projects = []
    for project in projects[:4]:
        compact_projects.append(
            {
                "rank": project.get("rank"),
                "name": clean_field_text(project.get("name", "")),
                "stats": clean_field_text(project.get("stats", "")),
                "what": clean_field_text(project.get("what", "")),
                "why_hot": clean_field_text(project.get("why_hot", "")),
                "who_for": clean_field_text(project.get("who_for", "")),
                "risks": clean_field_text(project.get("risks", "")),
            }
        )

    payload = {
        "header": clean_field_text(parsed.get("header", "")),
        "summary": clean_field_text(parsed.get("summary", "")),
        "projects": compact_projects,
    }
    fallback_subject = resolve_subject(custom_subject, "GitHub 熱門項目週報")
    prompt = f"""
你是一位擅長做繁體中文短影音口播稿的科技創作者。
請把下面的 GitHub 週報資料，改寫成更像短影音快報的旁白稿。

規則：
- 只輸出 JSON。
- 只能使用輸入資料裡已經出現的資訊，不能自行補新事實。
- 一定要保留 repo 名稱原樣，例如 oh-my-codex、claude-howto、VibeVoice（Microsoft）。
- 可以保留星數與週增星數，但不要把整份報告念成表格。
- 開頭先講本週趨勢，接著聚焦最值得看的 3 個專案，最後一句帶到後面還值得追的項目。
- 口氣要像短影音科技博主在做每週精選，不要像文件摘要，不要像投影片條列。
- script 拆成 9 到 12 行，每行一句完整口播，適合 TTS 逐句念。
- script 總長目標 360 到 520 個中文字。
- 每行結尾都要有中文標點。
- 不要使用 markdown、項目符號、編號。

輸出格式：
{{
  "subject": "簡短主題",
  "script": "逐行口播稿，用 \\n 分隔"
}}

預設主題：
{fallback_subject}

結構化資料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    result = _call_ollama_json(
        (GITHUB_VOICEOVER_PRIMARY_MODEL, GITHUB_VOICEOVER_FALLBACK_MODEL),
        prompt,
        timeout=300,
        temperature=0.35,
    )
    if not isinstance(result, dict):
        return None

    subject = resolve_subject(str(result.get("subject", "") or ""), fallback_subject)
    script, normalized_lines = _normalize_voiceover_script(result.get("script", ""))
    if not script or looks_garbled(script):
        return None
    if len(normalized_lines) < 8:
        return None
    required = [project.get("name", "") for project in projects[:3]]
    if not all(name and name in script for name in required[:2]):
        return None
    return subject, script


def _try_llm_daily_news_voiceover_v1(parsed: dict, custom_subject: str = "") -> tuple[str, str] | None:
    events = parsed.get("events") or []
    if not events:
        return None

    compact_events = []
    for event in events[:4]:
        compact_events.append(
            {
                "rank": event.get("rank"),
                "title": clean_field_text(event.get("title", "")),
                "importance": clean_field_text(event.get("importance", "")),
                "summary": clean_field_text(event.get("summary", "")),
                "highlights": [clean_field_text(item) for item in (event.get("highlights") or [])[:3] if clean_field_text(item)],
            }
        )

    payload = {
        "header": clean_field_text(parsed.get("header", "")),
        "summary": clean_field_text(parsed.get("summary", "")),
        "events": compact_events,
    }
    fallback_subject = resolve_subject(custom_subject, "今日科技新聞重點")
    prompt = f"""
你是一位擅長做繁體中文短影音科技快報的主播。
請把下面的科技情報報告，改寫成更像「今天科技圈發生什麼」的短影音旁白稿。

規則：
- 只輸出 JSON。
- 只能使用輸入資料裡已經出現的資訊，不能補充外部事實。
- 保留公司名、產品名、型號、縮寫與英文專有名詞原樣。
- 先用一句話講今天科技圈最值得注意的方向，再帶 3 到 4 則最重要新聞。
- 每則新聞都要有兩件事：發生了什麼、為什麼值得關注。
- 口氣要像短影音新聞主播，節奏快、重點明確，但不要變成條列摘要。
- script 拆成 9 到 12 行，每行一句完整口播，適合 TTS 逐句念。
- script 總長目標 360 到 520 個中文字。
- 每行結尾都要有中文標點。
- 不要使用 markdown、項目符號、編號。

輸出格式：
{{
  "subject": "簡短主題",
  "script": "逐行口播稿，用 \\n 分隔"
}}

預設主題：
{fallback_subject}

結構化資料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    result = _call_ollama_json(
        (TECHNEWS_VOICEOVER_PRIMARY_MODEL, TECHNEWS_VOICEOVER_FALLBACK_MODEL),
        prompt,
        timeout=300,
        temperature=0.35,
    )
    if not isinstance(result, dict):
        return None

    subject = resolve_subject(str(result.get("subject", "") or ""), fallback_subject)
    script, normalized_lines = _normalize_voiceover_script(result.get("script", ""))
    if not script or looks_garbled(script):
        return None
    if len(normalized_lines) < 8:
        return None
    required = [event.get("title", "") for event in events[:2]]
    if not all(title and any(token in script for token in title.split()[:1]) for title in required):
        return None
    return subject, script


def polish_market_report_script(script: str) -> str:
    lines = [line.strip() for line in script.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) <= 7:
        return "\n".join(lines)

    target_group_count = 6
    groups: list[list[str]] = []
    if lines:
        groups.append([lines[0]])
    cursor = 1
    while cursor < len(lines):
        remaining = len(lines) - cursor
        slots_left = target_group_count - len(groups)
        if slots_left <= 1:
            groups.append(lines[cursor:])
            break
        take = 2 if remaining > slots_left else 1
        groups.append(lines[cursor : cursor + take])
        cursor += take

    polished = []
    for group in groups[:target_group_count]:
        merged = _merge_script_group(group)
        if merged:
            polished.append(short_text(merged, 78))
    return "\n".join(polished)


def enrich_market_report_script(script: str, parsed: dict) -> str:
    if len(script) >= 420:
        return script

    themes = parsed.get("themes", [])[:4]
    if not themes:
        return script

    enriched_lines = [line.strip() for line in script.split("\n") if line.strip()]
    if not enriched_lines:
        enriched_lines = ["今天市場，先看資金到底在交易什麼。"]

    for theme in themes:
        region = theme.get("region", "")
        prefix = "美股這邊" if "US" in region else "台股這邊" if "Taiwan" in region else "市場這邊"
        title = clean_field_text(theme.get("title", ""))
        summary = clean_field_text(theme.get("summary", ""))
        bullets = [clean_field_text(item) for item in theme.get("bullets", []) if clean_field_text(item)]

        opener = f"{prefix}，市場目前在交易 {title}。"
        if opener not in enriched_lines:
            enriched_lines.append(opener)

        if summary:
            summary_line = summary if re.search(r"[。！？!?]$", summary) else f"{summary}。"
            if summary_line not in enriched_lines:
                enriched_lines.append(summary_line)

        if bullets:
            bullet_slice = bullets[:2]
            bullet_line = "，".join(item.rstrip("。！？!?") for item in bullet_slice if item)
            if bullet_line:
                bullet_line = bullet_line if re.search(r"[。！？!?]$", bullet_line) else f"{bullet_line}。"
                if bullet_line not in enriched_lines:
                    enriched_lines.append(bullet_line)

    if not any("接下來" in line or "接著" in line or "觀察" in line for line in enriched_lines):
        enriched_lines.append("接下來要觀察的，就是資金輪動能不能延續，還有科技股修復力道會不會再往上。")

    polished = []
    for line in enriched_lines:
        cleaned = clean_field_text(line)
        if cleaned and not re.search(r"[。！？!?]$", cleaned):
            cleaned += "。"
        if cleaned:
            polished.append(cleaned)

    polished_script = polish_market_report_script("\n".join(polished))
    return polished_script


def _try_llm_voiceover(parsed: dict, custom_subject: str = "") -> tuple[str, str] | None:
    content_type = parsed.get("content_type", "generic")
    items = (
        parsed.get("projects")
        or parsed.get("events")
        or parsed.get("themes")
        or []
    )
    if not items:
        return None

    payload = {
        "content_type": content_type,
        "header": parsed.get("header", ""),
        "summary": parsed.get("summary", ""),
        "items": items[:5],
    }
    fallback_subject = resolve_subject(custom_subject, _voiceover_prompt_type(content_type) or "短影音重點整理")
    prompt = f"""
你是一位擅長做繁體中文短影音口播稿的編輯。

請把下面的結構化資料，改寫成「自然、好講、像主持人口播」的短影音講稿。

規則：
- 只輸出 JSON。
- 不要補充資料來源外的新事實。
- 不要自行補數字、百分比、成長率、營收、價格、時間點或名次。
- 如果輸入資料沒有寫，就不要自己加。
- 保留原本的專案名、公司名、產品名、大小寫、括號、連字號、數字和星數格式。
- 口氣要像在跟觀眾快速講重點，不要像報告朗讀。
- script 請拆成 8 到 14 行，每行一句或半句，適合 TTS 逐句念。
- 每行盡量短，避免太書面。
- 不要使用 markdown、項目符號、編號。
- 如果是 GitHub 週報，聚焦前三名，再一句帶到後面值得看的項目。
- 如果是科技新聞，聚焦前三則新聞，講「發生什麼」和「為什麼值得看」。
- 如果是市場報告，聚焦最重要的兩到三個主題，講市場在交易什麼。

輸出格式：
{{
  "subject": "簡短主題",
  "script": "逐行口播稿，每行用 \\n 分隔"
}}

建議主題：
{fallback_subject}

資料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    result = _call_ollama_json(
        (VOICEOVER_PRIMARY_MODEL, VOICEOVER_FALLBACK_MODEL),
        prompt,
        timeout=240,
        temperature=0.35,
    )
    if not isinstance(result, dict):
        return None

    subject = resolve_subject(str(result.get("subject", "") or ""), fallback_subject)
    script = str(result.get("script", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in script.split("\n") if line.strip()]
    script = "\n".join(lines)
    if not script or looks_garbled(script):
        return None
    if len(lines) < 6:
        return None
    if not _voiceover_has_required_refs(script, parsed):
        return None
    return subject, script


def normalize_script(text: str) -> str:
    cleaned = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    lines = [line.strip(" -•\t") for line in cleaned.split("\n")]
    lines = [line for line in lines if line]
    return " ".join(lines).strip()


def extract_report_date(source_text: str) -> str:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{int(year):04d}年{int(month)}月{int(day)}日"


def parsed_report_date(parsed: dict) -> str:
    return clean_field_text(str(parsed.get("report_date", "") or ""))


def default_title_from_subject(subject: str, parsed: dict | None = None) -> str:
    base = (subject or "AI short").strip()
    report_date = parsed_report_date(parsed or {})
    content_type = str((parsed or {}).get("content_type", "") or "")
    if report_date and content_type in {"daily_tech_news", "market_report"} and report_date not in base:
        base = f"{base}｜{report_date}"
    if len(base) > 82:
        base = base[:79].rstrip() + "..."
    if "#shorts" not in base.lower():
        base = f"{base} #shorts"
    return base


def looks_garbled(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    bad = value.count("?") + value.count("�")
    return bad >= 2 and bad / max(len(value), 1) >= 0.08


def fallback_subject(parsed: dict, script: str) -> str:
    projects = parsed.get("projects") or []
    if projects:
        return f"GitHub 熱門開源項目週報：{projects[0].get('name', 'Top Repo')}"
    cleaned = normalize_script(script)
    return cleaned[:32].rstrip("，。,. ") or "GitHub 熱門開源項目"


def resolve_subject(custom_subject: str, fallback: str) -> str:
    candidate = (custom_subject or "").strip()
    if candidate and not looks_garbled(candidate):
        return candidate
    return fallback


def resolve_metadata(custom_title: str, custom_description: str, subject: str, script: str, parsed: dict) -> tuple[str, str]:
    safe_subject = subject if not looks_garbled(subject) else fallback_subject(parsed, script)
    title_candidate = (custom_title or "").strip()
    desc_candidate = (custom_description or "").strip()
    title = title_candidate if title_candidate and not looks_garbled(title_candidate) else default_title_from_subject(safe_subject, parsed)
    description = desc_candidate if desc_candidate and not looks_garbled(desc_candidate) else normalize_script(script)
    return title, description


def apply_report_date_to_script(script: str, parsed: dict) -> str:
    content_type = str(parsed.get("content_type", "") or "")
    report_date = parsed_report_date(parsed)
    cleaned = str(script or "").strip()
    if not cleaned or not report_date:
        return cleaned
    if content_type not in {"daily_tech_news", "market_report"}:
        return cleaned
    if report_date in cleaned:
        return cleaned
    if content_type == "daily_tech_news":
        return f"今天是{report_date}，{cleaned}"
    return f"今天是{report_date}，{cleaned}"


def clean_field_text(text: str) -> str:
    text = normalize_script(text)
    text = re.sub(r"^[：:]+", "", text).strip()
    return text


def short_text(text: str, limit: int = 60) -> str:
    text = clean_field_text(text)
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 1].rstrip("，。、；：,. ")
    return trimmed + "…"


def ensure_cn_punctuation(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", clean_field_text(text)).strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip("。！？!?；;，、：,:… ")
    if not cleaned:
        return ""
    return cleaned + "。"


def natural_clip(text: str, limit: int = 72) -> str:
    cleaned = re.sub(r"\s+", " ", clean_field_text(text)).strip()
    if len(cleaned) <= limit:
        return cleaned.rstrip("。！？!?；;，、：,:… ")

    for sep in ["。", "；", "：", "，", "（", "("]:
        idx = cleaned.find(sep)
        if 10 <= idx <= limit:
            return cleaned[:idx].rstrip("。！？!?；;，、：,:… ")

    candidates = [cleaned.rfind(sep, 0, limit + 1) for sep in ["，", "、", "；", "：", " "]]
    cut_at = max(candidates)
    if cut_at >= max(14, limit // 2):
        return cleaned[:cut_at].rstrip("。！？!?；;，、：,:… ")
    return cleaned[:limit].rstrip("。！？!?；;，、：,:… ")


def tech_spoken_title(title: str) -> str:
    raw = clean_field_text(title)
    raw_parts = [part.strip() for part in re.split(r"　+", raw) if part.strip()]
    if len(raw_parts) > 1 and 8 <= len(raw_parts[0]) <= 40:
        raw = raw_parts[0]
    cleaned = re.sub(r"\s+", " ", raw).strip()
    if "——" in cleaned:
        cleaned = cleaned.split("——", 1)[0].strip()
    if "—" in cleaned:
        left, right = [part.strip() for part in cleaned.split("—", 1)]
        if left and len(left) <= 30:
            cleaned = left
        elif right:
            cleaned = f"{left}，{right}"
    return natural_clip(cleaned, 38)


def tech_summary_line(event: dict) -> str:
    highlights = [natural_clip(item, 54) for item in (event.get("highlights") or []) if natural_clip(item, 54)]
    if highlights:
        first = highlights[0]
        if len(first) >= 14:
            return first
    summary = natural_clip(event.get("summary", ""), 66)
    if summary:
        summary = re.sub(r"^在 AI [^，。]+時代，", "", summary).strip()
        summary = re.sub(r"^最新商業數據顯示，", "", summary).strip()
        summary = re.sub(r"^Google 於 [^，。]+，", "", summary).strip()
        if len(summary) >= 14:
            return summary
    return ""


def tech_impact_line(event: dict) -> str:
    highlights = [natural_clip(item, 58) for item in (event.get("highlights") or []) if natural_clip(item, 58)]
    if len(highlights) >= 2:
        return highlights[1]
    summary = natural_clip(event.get("summary", ""), 84)
    if summary:
        return summary
    return ""


def tech_extra_line(event: dict) -> str:
    highlights = [natural_clip(item, 62) for item in (event.get("highlights") or []) if natural_clip(item, 62)]
    if len(highlights) >= 3:
        return highlights[2]
    if len(highlights) >= 2:
        return highlights[-1]
    summary = natural_clip(event.get("summary", ""), 96)
    if summary:
        summary = re.sub(r"^Anthropic 年化營收已突破 300 億美元[^，。]*，", "", summary).strip()
        summary = re.sub(r"^在 AI 需求爆發的時代，", "", summary).strip()
        summary = re.sub(r"^最新商業數據顯示，", "", summary).strip()
        summary = re.sub(r"^Google 於 3 月 31 日釋出的 Chrome 瀏覽器更新，", "", summary).strip()
        if len(summary) >= 16:
            return summary
    return ""


def polish_daily_news_line(text: str) -> str:
    cleaned = ensure_cn_punctuation(text)
    if not cleaned:
        return ""

    replacements = [
        ("這也等於，代表", "這也代表"),
        ("這也表示，", "這也表示"),
        ("影響不只是單一漏洞，而是 影響", "影響不只是單一漏洞，而是"),
        ("而且影響不只是單一漏洞，還包括 影響", "而且影響不只是單一漏洞，還包括"),
        ("所以最直接的動作，就是 建議", "所以最直接的做法，就是"),
        ("這條線現在被注意到，是因為 ", "這條線現在被注意到，是因為"),
        ("這條線現在會被放大看，是因為 ", "這條線現在會被放大看，是因為"),
        ("這件事最先要看的，就是 ", "這件事最先要看的，就是"),
        ("這條消息真正有感的地方，是 ", "這條消息真正有感的地方，是"),
        ("這組數字最有感的地方，是 ", "這組數字最有感的地方，是"),
        ("這件事麻煩的地方，是 ", "這件事麻煩的地方，是"),
        ("這也在提醒市場，反映", "這也在提醒市場，"),
        ("這代表，代表", "這代表"),
        ("，而是 而是", "，而是"),
        ("就是 建議", "就是建議"),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"([，。！？；：])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([，。！？；：])", r"\1", cleaned)
    cleaned = re.sub(r"^這也代表，代表", "這也代表", cleaned)
    cleaned = re.sub(r"^所以最直接的做法，就是建議", "所以最直接的做法，就是", cleaned)
    cleaned = cleaned.rstrip("。！？!?；;，、：,:… ")
    return cleaned + "。"


def detect_content_type(source_text: str) -> str:
    text = source_text.replace("﻿", "")
    # --- keyword-based checks first (more specific) ---
    if (
        "每日國際新聞報告" in text
        or "國際情勢報告" in text
        or "國際新聞報告" in text
    ):
        return "international_brief"
    if (
        "科技情報報告" in text
        or "Tech Intelligence Report" in text
        or "每日科技新聞" in text
        or "重大科技事件" in text
    ):
        return "daily_tech_news"
    if (
        "Daily Market Theme Report" in text
        or "Market Theme Report" in text
        or "【美股市場" in text
        or "【美股" in text
        or "【台股市場" in text
        or "【台股" in text
        or "美股市場 · US Markets" in text
        or "台股市場 · Taiwan Markets" in text
        or "市場主題報告" in text
    ):
        return "market_report"
    if (
        "1️⃣" in text
        or "GitHub 熱門項目週報" in text
        or "GitHub 熱門開源項目" in text
    ):
        return "github_weekly"
    if (
        "每日趨勢報告" in text
        or "Daily Trend" in text
        or ("每日摘要" in text and ("KPOP" in text or "韓流" in text or "動漫" in text or "動畫" in text))
    ):
        return "daily_trend"
    # --- structural pattern fallbacks ---
    if "對台灣影響" in text or re.search(r"^\s*【事件[一二三四五六七八九十0-9]+】", text, re.M):
        return "international_brief"
    if re.search(r"^[─-]{20,}\s*$", text, re.M) and re.search(r"^【\s*\d+\s*】", text, re.M):
        return "daily_tech_news"
    if re.search(r"^\s*\d+[.)]\s+", text, re.M) or re.search(r"^\s*(\d+)(?:️?⃣)\s+", text, re.M):
        return "github_weekly"
    return "generic"


def safe_weekly_gain(stats: str) -> str:
    match = re.search(r"\+\s*([0-9,]+)\s*[\u2605*]?", stats)
    if match:
        return f"+{match.group(1)} ★"
    return ""


def safe_total_stars(stats: str) -> str:
    match = re.search(r"(?:總|total)\s*[:：]?\s*([0-9,]+)\s*[\u2605*]?", stats, re.I)
    if not match:
        match = re.search(r"（\s*總\s*([0-9,]+)\s*[\u2605*]?\s*）", stats, re.I)
    if not match:
        match = re.search(r"([0-9,]+)\s*[\u2605*]", stats)
    if match:
        return f"{match.group(1)} ★"
    return ""


def parse_ranked_projects(source_text: str) -> dict:
    lines = [line.rstrip() for line in source_text.replace("\ufeff", "").replace("\r\n", "\n").split("\n")]
    header_lines = []
    summary_lines = []
    projects = []
    current = None
    current_field = None
    in_summary = False

    def add_to_field(field_name: str, content: str) -> None:
        if current is None or not content:
            return
        current[field_name] = clean_field_text((current.get(field_name, "") + " " + content).strip())

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[=\-_.\s]+", line):
            continue
        if line == "---":
            current_field = None
            continue

        match = re.match(r"^【第\s*(\d+)\s*名】\s*(.+)$", line)
        if not match:
            match = re.match(r"^(\d+)\s*[.)]\s*(.+)$", line)
        if not match:
            match = re.match(r"^(\d+)(?:\ufe0f?\u20e3)\s*(.+)$", line)
        if match:
            current = {
                "rank": int(match.group(1)),
                "name": match.group(2).strip(),
                "link": "",
                "language": "",
                "stats": "",
                "what": "",
                "why_hot": "",
                "who_for": "",
                "risks": "",
            }
            projects.append(current)
            current_field = None
            in_summary = False
            continue

        if current is None:
            if "本週趨勢摘要" in line:
                in_summary = True
                continue
            if "本週精選專案" in line:
                in_summary = False
                continue
            if "GitHub 熱門" in line or "週報" in line:
                header_lines.append(line)
                continue
            if in_summary and len(line) >= 10:
                summary_lines.append(line)
                continue
            if len(line) >= 10 and "生成時間" not in line:
                summary_lines.append(line)
            continue

        if line.startswith("🔗"):
            current["link"] = line.lstrip("🔗").strip()
            current_field = None
            continue
        if line.startswith(("📌", "⭐", "★")) or "本週 +" in line:
            current["stats"] = line
            current_field = None
            continue
        if line.startswith("語言："):
            current["language"] = line.split("：", 1)[1].strip()
            current_field = None
            continue

        section_match = re.match(r"^(做什麼|為什麼變熱門|為什麼爆紅|適合誰|風險與門檻|風險 / 門檻)\s*[:：]\s*(.*)$", line)
        if section_match:
            title = section_match.group(1).strip()
            body = section_match.group(2).strip()
            key = SECTION_KEY_MAP.get(title)
            if key:
                current_field = key
                if body:
                    add_to_field(key, body)
            continue

        if current_field:
            add_to_field(current_field, line)

    return {
        "content_type": "github_weekly",
        "header": " ".join(header_lines).strip(),
        "summary": " ".join(summary_lines[:3]).strip(),
        "projects": projects,
    }

def parse_daily_news_events(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n")
    events = []

    def append_event(rank: int, title: str, body: str) -> None:
        importance = ""
        importance_match = re.search(r"重要性[:：]\s*([★☆]+)", body)
        if not importance_match:
            importance_match = re.search(r"重要性[:：]\s*[█░]+\s*(\S+)", body)
        if importance_match:
            importance = importance_match.group(1)

        # --- Phase 1: join word-wrapped lines into complete sentences ---
        joined_lines: list[str] = []
        section_label = ""
        for raw in body.splitlines():
            stripped = raw.strip()
            if not stripped:
                # Empty line → force start of new paragraph
                if joined_lines and joined_lines[-1] != "":
                    joined_lines.append("")
                continue
            # Skip separator lines (===, ━━━, ───, etc.)
            if re.fullmatch(r"[━─═=\-_\s]+", stripped):
                continue
            # Skip 🔷 lines entirely (metadata: title echo, date, source)
            if stripped.startswith("🔷"):
                continue
            # Skip metadata lines
            if stripped.startswith(("來源：", "來源:", "重點：", "主要看點", "Sources:", "資料來源", "標籤：", "標籤:")):
                continue
            # Handle section headers like "事件詳情：", "市場反應：", "戰略意義："
            if re.fullmatch(r"(事件詳情|市場反應|戰略意義|影響分析|技術細節|背景資訊|後續觀察|延伸分析)\s*[：:]", stripped):
                section_label = stripped.rstrip("：: ")
                joined_lines.append(f"__SECTION__:{section_label}")
                continue
            # Skip importance lines
            if stripped.startswith("重要性") and ("★" in stripped or "☆" in stripped or "█" in stripped):
                continue
            # Bullet items get their own line
            if stripped.startswith(("•", "▸", "◆")) or re.match(r"^\d+\.\s+", stripped):
                joined_lines.append(stripped)
                continue
            # Regular text: merge with previous line if it didn't end with terminal punctuation
            if joined_lines and joined_lines[-1] and joined_lines[-1] != "" and not joined_lines[-1].startswith("__SECTION__"):
                prev = joined_lines[-1]
                if not re.search(r"[。！？!?；]$", prev) and not prev.startswith(("•", "▸", "◆")) and not re.match(r"^\d+\.\s+", prev):
                    joined_lines[-1] = prev + stripped
                    continue
            joined_lines.append(stripped)

        # --- Phase 2: classify joined lines into paragraphs vs highlights ---
        paragraphs = []
        highlights = []
        current_section = ""
        for jline in joined_lines:
            if not jline:
                continue
            if jline.startswith("__SECTION__:"):
                current_section = jline.split(":", 1)[1]
                continue
            # Strip bullet prefixes
            normalized = re.sub(r"^[•▸◆]\s*", "", jline).strip()
            if not normalized:
                continue
            # Numbered items (1. xxx) → highlights
            if re.match(r"^\d+\.\s+", normalized):
                bullet_text = re.sub(r"^\d+\.\s+", "", normalized).strip()
                if bullet_text and len(bullet_text) >= 6:
                    highlights.append(clean_field_text(bullet_text))
                continue
            if jline.startswith(("•", "▸", "◆")):
                highlights.append(clean_field_text(normalized))
                continue
            # Non-detail sections → highlights; detail section → paragraphs
            if current_section and current_section != "事件詳情":
                highlights.append(clean_field_text(normalized))
            else:
                paragraphs.append(clean_field_text(normalized))

        # If no explicit highlights, promote later paragraphs as highlights
        if not highlights and len(paragraphs) > 1:
            for extra_p in paragraphs[1:4]:
                if extra_p and len(extra_p) >= 10:
                    highlights.append(extra_p)

        summary = paragraphs[0] if paragraphs else ""
        events.append(
            {
                "rank": rank,
                "title": clean_field_text(title),
                "importance": importance,
                "summary": summary,
                "highlights": highlights[:3],
            }
        )

    event_pattern = re.compile(
        r"━━━\s*事件[一二三四五六七八九十0-9]+\s*[:：]\s*(.*?)\s*━━━(.*?)(?=━━━\s*事件|={10,}|$)",
        re.S,
    )
    for idx, match in enumerate(event_pattern.finditer(text), start=1):
        append_event(idx, match.group(1), match.group(2))

    # Fallback: header-based parsing for 【事件N】 format (handles centered/indented lines)
    if not events:
        header_pat = re.compile(r"^\s*【\s*事件\s*[一二三四五六七八九十\d]+\s*】", re.M)
        # Find end-of-events marker (趨勢觀察, 報告結束, 報告產生時間, etc.)
        end_marker = re.search(
            r"^\s*(趨勢觀察|報告結束|報告產生時間|製作系統|本報告由)",
            text, re.M,
        )
        text_end = end_marker.start() if end_marker else len(text)
        headers = list(header_pat.finditer(text))
        for h_idx, header in enumerate(headers):
            start = header.end()
            end = headers[h_idx + 1].start() if h_idx + 1 < len(headers) else text_end
            body = text[start:end]
            # Try to extract title from first non-empty content line
            title = ""
            for bline in body.splitlines():
                bline = bline.strip()
                if not bline or re.fullmatch(r"[━─═=\-_\s]+", bline):
                    continue
                # Title line often starts with ◆ or 🔷
                if bline.startswith("◆"):
                    title = re.sub(r"^◆\s*", "", bline).strip()
                    break
                if bline.startswith("🔷"):
                    title = re.sub(r"^🔷\s*", "", bline).strip()
                    break
                if len(bline) >= 6:
                    title = bline
                    break
            if title:
                append_event(h_idx + 1, title, body)

    if not events:
        lines = text.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not re.fullmatch(r"[─-]{20,}", line):
                idx += 1
                continue
            if idx + 2 >= len(lines):
                break
            title_line = lines[idx + 1].strip()
            if not re.match(r"^【\s*\d+\s*】", title_line):
                idx += 1
                continue
            if not re.fullmatch(r"[─-]{20,}", lines[idx + 2].strip()):
                idx += 1
                continue

            title_match = re.match(r"^【\s*(\d+)\s*】\s*(.+)$", title_line)
            rank = int(title_match.group(1))
            title = title_match.group(2)

            body_lines = []
            cursor = idx + 3
            while cursor < len(lines):
                current = lines[cursor].strip()
                if re.fullmatch(r"[─-]{20,}", current):
                    next_line = lines[cursor + 1].strip() if cursor + 1 < len(lines) else ""
                    if re.match(r"^【\s*\d+\s*】", next_line):
                        break
                if re.fullmatch(r"[=]{10,}", current) or current.startswith("報告結束") or current.startswith("製作系統："):
                    break
                body_lines.append(lines[cursor])
                cursor += 1

            append_event(rank, title, "\n".join(body_lines))
            idx = cursor

    # === Fallback 3: 事件N：Title format (no brackets, ─── separators) ===
    if not events:
        _cn_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        event_title_pat = re.compile(
            r"^\s*事件([一二三四五六七八九十\d]+)\s*[：:]\s*(.+)$", re.M
        )
        end_marker = re.search(
            r"^\s*(趨勢觀察|報告結束|報告產生時間|製作系統|本報告由|報告總結|={10,})",
            text, re.M,
        )
        text_end = end_marker.start() if end_marker else len(text)
        headers = list(event_title_pat.finditer(text))
        for h_idx, header in enumerate(headers):
            cn_rank = header.group(1)
            rank = _cn_num_map.get(cn_rank, int(cn_rank) if cn_rank.isdigit() else h_idx + 1)
            title = header.group(2).strip()
            start = header.end()
            end = headers[h_idx + 1].start() if h_idx + 1 < len(headers) else text_end
            body = text[start:end]
            # Strip leading 摘要： prefix from body if present
            body = re.sub(r"^\s*摘要\s*[：:]\s*", "", body.lstrip("\n"), count=1)
            if title:
                append_event(rank, title, body)

    return {
        "content_type": "daily_tech_news",
        "header": "科技情報報告",
        "report_date": extract_report_date(source_text),
        "summary": short_text(" ".join(event["title"] for event in events[:3]), 80),
        "events": events,
    }


def parse_international_brief(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    events = []

    event_pattern = re.compile(
        r"【事件([一二三四五六七八九十0-9]+)】\s*(.*?)\n(.*?)(?=\n【事件[一二三四五六七八九十0-9]+】|\n[═=]{10,}|\Z)",
        re.S,
    )

    def _clean_lines(block: str) -> list[str]:
        lines = []
        for raw in block.splitlines():
            line = clean_field_text(raw.strip().lstrip("-•▸"))
            if not line:
                continue
            if re.fullmatch(r"[─-]{10,}", line):
                continue
            lines.append(line)
        return lines

    for idx, match in enumerate(event_pattern.finditer(text), start=1):
        title = clean_field_text(match.group(2))
        body = match.group(3)
        impact = ""
        summary = ""
        details: list[str] = []

        summary_match = re.search(r"摘要[:：]\s*(.*?)(?=\n(?:關鍵細節|對台灣影響)[:：]|\Z)", body, re.S)
        if summary_match:
            summary_lines = _clean_lines(summary_match.group(1))
            summary = " ".join(summary_lines[:3]).strip()

        detail_match = re.search(r"關鍵細節[:：]\s*(.*?)(?=\n對台灣影響[:：]|\Z)", body, re.S)
        if detail_match:
            details = _clean_lines(detail_match.group(1))[:4]

        impact_match = re.search(r"對台灣影響[:：]\s*([^\n]*)(.*?)(?=\n(?:【事件|[═=]{10,}|報告結束)|\Z)", body, re.S)
        if impact_match:
            impact_label = clean_field_text(impact_match.group(1))
            impact_lines = _clean_lines(impact_match.group(2))
            impact = " ".join(([impact_label] if impact_label else []) + impact_lines[:3]).strip()

        events.append(
            {
                "rank": idx,
                "title": title,
                "summary": short_text(summary, 160),
                "details": details,
                "taiwan_impact": short_text(impact, 160),
            }
        )

    return {
        "content_type": "international_brief",
        "header": "國際情勢報告",
        "report_date": extract_report_date(source_text),
        "summary": short_text(" ".join(event["title"] for event in events[:3]), 90),
        "events": events,
    }


def parse_market_report(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    region = ""
    themes = []
    current = None

    def flush_current():
        nonlocal current
        if current:
            current["summary"] = short_text(" ".join(current["summary_lines"][:4]), 100)
            themes.append(current)
            current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("【美股市場"):
            flush_current()
            region = "US Markets"
            continue
        if line.startswith("【台股市場"):
            flush_current()
            region = "Taiwan Markets"
            continue
        m = re.match(r"^■\s*(今日主導主題[一二三四五六七八九十0-9]+)\s*[:：]\s*(.+)$", line)
        if m:
            flush_current()
            current = {
                "region": region or "Market",
                "slot": m.group(1),
                "title": clean_field_text(m.group(2)),
                "summary_lines": [],
                "bullets": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("•"):
            current["bullets"].append(clean_field_text(line))
            continue
        if line.startswith("【") and line.endswith("】"):
            continue
        if line.startswith("股票代碼") or line.startswith("個股代碼") or line.startswith("股價突破"):
            current["summary_lines"].append(clean_field_text(line))
            continue
        if len(line) >= 8:
            current["summary_lines"].append(clean_field_text(line))
    flush_current()
    return {
        "content_type": "market_report",
        "header": "Daily Market Theme Report",
        "report_date": extract_report_date(source_text),
        "summary": short_text(" ".join(theme["title"] for theme in themes[:3]), 90),
        "themes": themes,
    }


def parse_market_report_v2(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    region = ""
    themes = []
    current = None
    last_kind = ""

    def flush_current() -> None:
        nonlocal current, last_kind
        if not current:
            return
        summary_lines = current.pop("summary_lines", [])
        summary_text = " ".join(summary_lines[:5]).strip()
        current["summary"] = short_text(summary_text, 180)
        current["bullets"] = current.get("bullets", [])[:6]
        themes.append(current)
        current = None
        last_kind = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[=\-─━_│\s]+", line):
            continue
        if line.startswith("Report Time") or line.startswith("Sources:"):
            continue

        if line.startswith("【美股") or re.match(r"^【.*US (Markets?|Stock).*】", line):
            flush_current()
            region = "US Markets"
            continue
        if line.startswith("【台股") or re.match(r"^【.*Taiwan (Markets?|Stock).*】", line):
            flush_current()
            region = "Taiwan Markets"
            continue
        if (
            line.startswith("【關鍵數據")
            or line.startswith("【本週觀察")
            or line.startswith("報告結束")
        ):
            flush_current()
            break

        theme_match = None
        if line.startswith("【主導主題") and "：" in line and line.endswith("】"):
            title = line.split("：", 1)[1].rsplit("】", 1)[0].strip()
            theme_match = title
        else:
            m = re.match(r"^主導主題(?:[一二三四五六七八九十0-9]+)?\s*[：:]\s*(.+)$", line)
            if m:
                theme_match = m.group(1).strip()
            else:
                m = re.match(r"^Theme(?:\s*\d+)?\s*[：:]\s*(.+)$", line, re.IGNORECASE)
                if m and current is None:
                    theme_match = m.group(1).strip()

        if theme_match:
            flush_current()
            current = {
                "region": region or "Market",
                "slot": f"Theme {len(themes) + 1}",
                "title": clean_field_text(theme_match),
                "summary_lines": [],
                "bullets": [],
            }
            continue

        if current is not None:
            m = re.match(r"^Theme(?:\s*\d+)?\s*[：:]\s*(.+)$", line, re.IGNORECASE)
            if m:
                current["english_title"] = clean_field_text(m.group(1).strip())
                continue

        if current is None:
            continue

        if line.startswith("Top Theme"):
            continue
        if line.startswith("【") and line.endswith("】"):
            continue
        if line.endswith(("：", ":")):
            continue

        if raw[:1] in {" ", "\t", "　"} and last_kind:
            extra = clean_field_text(line)
            if extra:
                if last_kind == "bullet" and current.get("bullets"):
                    current["bullets"][-1] = clean_field_text(current["bullets"][-1] + " " + extra)
                    continue
                if last_kind == "summary" and current.get("summary_lines"):
                    current["summary_lines"][-1] = clean_field_text(current["summary_lines"][-1] + " " + extra)
                    continue

        if re.match(r"^[•\-☐]\s*", line):
            bullet = re.sub(r"^[•\-☐]\s*", "", line).strip()
            if bullet:
                current["bullets"].append(clean_field_text(bullet))
                last_kind = "bullet"
            continue

        if re.match(r"^\d+\.\s*", line):
            bullet = re.sub(r"^\d+\.\s*", "", line).strip()
            if bullet:
                current["bullets"].append(clean_field_text(bullet))
                last_kind = "bullet"
            continue

        if len(line) >= 8:
            current["summary_lines"].append(clean_field_text(line))
            last_kind = "summary"

    flush_current()
    return {
        "content_type": "market_report",
        "header": "Market Theme Report",
        "report_date": extract_report_date(source_text),
        "summary": short_text(" ".join(theme["title"] for theme in themes[:3]), 90),
        "themes": themes,
    }

def build_ranked_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    projects = parsed["projects"]
    header = parsed.get("header", "")
    summary = parsed.get("summary", "")

    if not projects:
        subject = resolve_subject(custom_subject, "GitHub 熱門項目週報")
        script = normalize_script(summary or header or "本週 GitHub 開源圈的焦點，仍然集中在 AI Agent、語音 AI 和開發工具。")
        return subject, script

    subject = resolve_subject(custom_subject, "GitHub 熱門項目週報")
    featured = projects[:5]
    beats = ["這週 GitHub 最值得看的，先抓五個專案。"]
    trend_line = natural_clip(summary, 90) or "AI Agent、語音 AI，還有開發者工具，還是這週最熱的三條線。"
    beats.append(f"如果先看整體趨勢，{trend_line.rstrip('。！？!?')}。")

    for idx, project in enumerate(featured, start=1):
        name = project["name"]
        weekly_gain = safe_weekly_gain(project.get("stats", ""))
        total_stars = safe_total_stars(project.get("stats", ""))
        what = natural_clip(project.get("what", ""), 88)
        why_hot = natural_clip(project.get("why_hot", ""), 92)
        who_for = natural_clip(project.get("who_for", ""), 62)
        risks = natural_clip(project.get("risks", ""), 62)

        if weekly_gain and total_stars:
            intro = f"先看第{idx}個，{name}，這週新增 {weekly_gain}，總星數來到 {total_stars}。"
        elif weekly_gain:
            intro = f"先看第{idx}個，{name}，這週新增 {weekly_gain}。"
        else:
            intro = f"先看第{idx}個，{name}。"

        beats.append(intro)
        if what:
            if idx == 1:
                beats.append(f"它在做的事情其實很直接，{ensure_cn_punctuation(what)}")
            elif idx == 2:
                beats.append(f"這個專案的核心用途是，{ensure_cn_punctuation(what)}")
            elif idx in (3, 4):
                beats.append(f"它主要在處理的是，{ensure_cn_punctuation(what)}")
            else:
                beats.append(f"這個專案在做的事情也很清楚，{ensure_cn_punctuation(what)}")
        if why_hot:
            if idx == 1:
                beats.append(f"這次會衝上來，關鍵就在於{ensure_cn_punctuation(why_hot).rstrip('。')}。")
            elif idx == 2:
                beats.append(f"它會被大量討論，主要是因為，{ensure_cn_punctuation(why_hot).rstrip('。')}。")
            elif idx in (3, 4):
                beats.append(f"這波熱度拉起來，原因大概就是，{ensure_cn_punctuation(why_hot).rstrip('。')}。")
            else:
                beats.append(f"這次能往上衝，背後原因是，{ensure_cn_punctuation(why_hot).rstrip('。')}。")
        if who_for:
            if idx == 1:
                beats.append(f"這個專案比較適合{ensure_cn_punctuation(who_for).rstrip('。')}。")
            elif idx == 2:
                beats.append(f"如果你現在正想補這塊，這個專案很適合{ensure_cn_punctuation(who_for).rstrip('。')}。")
            elif idx in (3, 4):
                beats.append(f"如果你本來就在看這條線，這個專案很適合{ensure_cn_punctuation(who_for).rstrip('。')}。")
            else:
                beats.append(f"如果你想往這方向深入，這個專案也很適合{ensure_cn_punctuation(who_for).rstrip('。')}。")
        if risks:
            if idx == 1:
                beats.append(f"不過先提醒一下，{ensure_cn_punctuation(risks).rstrip('。')}。")
            elif idx == 2:
                beats.append(f"實際上手前也要注意，{ensure_cn_punctuation(risks).rstrip('。')}。")
            elif idx in (3, 4):
                beats.append(f"另外一個要先知道的點是，{ensure_cn_punctuation(risks).rstrip('。')}。")
            else:
                beats.append(f"先講風險，{ensure_cn_punctuation(risks).rstrip('。')}。")

    follow_up = [project["name"] for project in projects[5:9] if project.get("name")]
    if follow_up:
        beats.append(f"除了前面這五個，後面像 {'、'.join(follow_up)} 也都有熱度。")
    beats.append("整體來看，這週最熱的還是 AI Agent、生產力工具，還有語音 AI 這幾條線。")

    normalized = [polish_daily_news_line(beat) for beat in beats if beat.strip()]
    script = "\n".join(line for line in normalized if line)
    return subject, script

def build_daily_news_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    events = parsed.get("events", [])
    subject = resolve_subject(custom_subject, "今日科技新聞重點")
    report_date = parsed_report_date(parsed)
    if not events:
        if report_date:
            return subject, f"今天是{report_date}，科技圈有幾個值得關注的大事件。"
        return subject, "今天科技圈有幾個值得關注的大事件。"

    count_map = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
    ordinal_map = {1: "第一", 2: "第二", 3: "第三", 4: "第四", 5: "第五", 6: "第六", 7: "第七", 8: "第八"}

    visible_events = events[:5]
    count_label = count_map.get(len(visible_events), str(len(visible_events)))

    if report_date:
        sections = [f"今天是{report_date}，科技圈有{count_label}則新聞值得看。"]
    else:
        sections = [f"今天科技圈有{count_label}則新聞值得看。"]

    for idx, event in enumerate(visible_events, start=1):
        title = clean_field_text(event.get("title", "")) or f"第{idx}則科技新聞"
        spoken_title = tech_spoken_title(title)
        summary = ensure_cn_punctuation(clean_field_text(event.get("summary", "")))
        highlights = [
            ensure_cn_punctuation(clean_field_text(item))
            for item in event.get("highlights", [])
            if clean_field_text(item)
        ]

        lines = [f"{ordinal_map.get(idx, f'第{idx}')}則，{spoken_title}。"]
        if summary:
            lines.append(summary)
        if highlights:
            lines.append(f"這裡最需要注意的是，{highlights[0].rstrip('。')}。")
        if len(highlights) > 1:
            lines.append(f"另外，{highlights[1].rstrip('。')}。")
        if len(highlights) > 2:
            lines.append(f"再來，{highlights[2].rstrip('。')}。")

        polished = [polish_daily_news_line(line) for line in lines if line.strip()]
        sections.append(" ".join(line for line in polished if line))

    if report_date:
        sections.append(f"以上是{report_date}整理的{count_label}則科技新聞重點。")
    else:
        sections.append(f"以上是今天整理的{count_label}則科技新聞重點。")
    return subject, "\n\n".join(section for section in sections if section.strip())


def build_international_brief_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    events = parsed.get("events", [])
    subject = resolve_subject(custom_subject, "國際情勢報告")
    report_date = parsed_report_date(parsed)
    if not events:
        if report_date:
            return subject, f"今天是{report_date}，國際線先看幾件最重要的事。"
        return subject, "今天國際線先看幾件最重要的事。"

    count_map = {
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    visible_events = events[:5]
    count_label = count_map.get(len(visible_events), str(len(visible_events)))
    intro = f"今天是{report_date}，國際線先看{count_label}件事。" if report_date else f"今天國際線先看{count_label}件事。"
    sections = [intro]

    for idx, event in enumerate(visible_events, start=1):
        spoken_title = clean_field_text(event.get("title", "")) or f"第{idx}則國際消息"
        summary_line = clean_field_text(event.get("summary", ""))
        detail_lines = [clean_field_text(item) for item in event.get("details", []) if clean_field_text(item)]
        impact_line = clean_field_text(event.get("taiwan_impact", ""))
        event_lines: list[str] = []

        if idx == 1:
            event_lines.append(f"第一，美伊和中東局勢先看，{spoken_title}。")
        elif idx == 2:
            event_lines.append(f"第二，拉回兩岸政治線，{spoken_title}。")
        elif idx == 3:
            event_lines.append(f"第三，再看中東戰事本身，{spoken_title}。")
        elif idx == 4:
            event_lines.append(f"第四，國際組織這邊要看，{spoken_title}。")
        else:
            event_lines.append(f"第五，美歐關係這邊要注意，{spoken_title}。")

        if summary_line:
            event_lines.append(f"{summary_line.rstrip('。')}。")

        if detail_lines:
            lead = detail_lines[0].rstrip("。")
            event_lines.append(f"這裡最需要注意的是，{lead}。")
        if len(detail_lines) > 1:
            second = detail_lines[1].rstrip("。")
            event_lines.append(f"另外，{second}。")

        if impact_line:
            event_lines.append(f"對台灣來說，{impact_line.rstrip('。')}。")

        sections.append(" ".join(polish_daily_news_line(line) for line in event_lines if line.strip()))

    script = "\n\n".join(section for section in sections if section.strip())
    return subject, script


def parse_international_brief_v2(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    def clean_block_lines(block: str) -> list[str]:
        items: list[str] = []
        for raw in block.splitlines():
            line = re.sub(r"^[\s•\-*]+", "", raw).strip()
            if not line or re.fullmatch(r"[=\-─━═\s]+", line):
                continue
            items.append(clean_field_text(line))
        return items

    def infer_report_date() -> str:
        match = re.search(r"(20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)", text)
        if match:
            return clean_field_text(match.group(1))
        return extract_report_date(source_text)

    def normalize_impact(label: str, lines: list[str]) -> str:
        level = ""
        match = re.search(r"（([^）]+)）", label)
        if match:
            level = clean_field_text(match.group(1))
        elif label:
            level = clean_field_text(label)

        parts: list[str] = []
        if level:
            parts.append(f"影響層級{level}")
        if lines:
            parts.append(lines[0])
        if len(lines) > 1:
            parts.append(lines[1])
        return "；".join(part for part in parts if part)

    header_pattern = re.compile(r"^【事件([一二三四五六七八九十0-9]+)】\s*(.+?)\s*$", re.M)
    headers = list(header_pattern.finditer(text))
    events: list[dict] = []

    for idx, header in enumerate(headers, start=1):
        start = header.end()
        end = headers[idx].start() if idx < len(headers) else len(text)
        body = text[start:end]

        title = clean_field_text(header.group(2))
        summary_match = re.search(r"摘要[:：]\s*(.*?)(?=\n\s*關鍵細節[:：]|\n\s*對台灣影響[:：]|\Z)", body, re.S)
        details_match = re.search(r"關鍵細節[:：]\s*(.*?)(?=\n\s*對台灣影響[:：]|\Z)", body, re.S)
        impact_match = re.search(r"對台灣影響[:：]\s*([^\n]*)(.*?)(?=\n\s*(?:【事件[一二三四五六七八九十0-9]+】|報告結束|下次報告)|\Z)", body, re.S)

        summary = clean_field_text(summary_match.group(1)) if summary_match else ""
        details = clean_block_lines(details_match.group(1))[:3] if details_match else []

        impact = ""
        if impact_match:
            impact_label = clean_field_text(impact_match.group(1))
            impact_lines = clean_block_lines(impact_match.group(2))[:3]
            impact = normalize_impact(impact_label, impact_lines)

        events.append(
            {
                "rank": idx,
                "title": title,
                "summary": summary,
                "details": details,
                "taiwan_impact": impact,
            }
        )

    return {
        "content_type": "international_brief",
        "header": "國際情勢報告",
        "report_date": infer_report_date(),
        "summary": short_text(" ".join(event["title"] for event in events[:3]), 120),
        "events": events,
    }


def build_international_brief_script_v2(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    events = parsed.get("events", [])
    subject = resolve_subject(custom_subject, "國際情勢報告")
    report_date = parsed_report_date(parsed)

    if not events:
        if report_date:
            return subject, f"今天是{report_date}，國際線先看幾件最重要的事。"
        return subject, "今天國際線先看幾件最重要的事。"

    count_map = {
        1: "一",
        2: "兩",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    ordinal_map = {
        1: "第一",
        2: "第二",
        3: "第三",
        4: "第四",
        5: "第五",
        6: "第六",
        7: "第七",
        8: "第八",
        9: "第九",
        10: "第十",
    }

    visible_events = events[:5]
    count_label = count_map.get(len(visible_events), str(len(visible_events)))
    intro = f"今天是{report_date}，國際線先看{count_label}件事。" if report_date else f"今天國際線先看{count_label}件事。"
    sections = [intro]

    for idx, event in enumerate(visible_events, start=1):
        title = clean_field_text(event.get("title", "")) or f"第{idx}則國際事件"
        summary = ensure_cn_punctuation(clean_field_text(event.get("summary", "")))
        details = [
            ensure_cn_punctuation(clean_field_text(item))
            for item in event.get("details", [])
            if clean_field_text(item)
        ]
        impact = ensure_cn_punctuation(clean_field_text(event.get("taiwan_impact", "")))

        lines = [f"{ordinal_map.get(idx, f'第{idx}')}則，{title}。"]
        if summary:
            lines.append(summary)
        if details:
            lines.append(f"這裡最需要注意的是，{details[0].rstrip('。')}。")
        if len(details) > 1:
            lines.append(f"另外，{details[1].rstrip('。')}。")
        if impact:
            lines.append(f"對台灣來說，{impact.rstrip('。')}。")

        polished = [polish_daily_news_line(line) for line in lines if line.strip()]
        sections.append(" ".join(line for line in polished if line))

    return subject, "\n\n".join(section for section in sections if section.strip())


def build_scene_plan_from_international_brief_v2(parsed: dict, title: str, script: str) -> list[dict]:
    events = parsed.get("events", [])
    if not events:
        return []

    plans = [
        {
            "index": 1,
            "scene_type": "Global opener",
            "voiceover": split_script_segments(script, 1)[0] if script else title,
            "on_screen_text": title or "國際情勢報告",
            "primary_stat": "",
            "secondary_stat": "",
            "summary": "World map, newsroom lighting, global headlines wall, subtle crisis markers.",
            "reason": "Open with a premium international-news mood and clear geopolitical tension.",
            "tags": ["International", "Geopolitics", "Taiwan impact"],
            "image_direction": "Cinematic global-news opener, premium newsroom look, dramatic but realistic, vertical 9:16, no visible logos or captions.",
        }
    ]

    for idx, event in enumerate(events[:4], start=2):
        title_text = clean_field_text(event.get("title", ""))
        summary_text = clean_field_text(event.get("summary", ""))
        impact_text = clean_field_text(event.get("taiwan_impact", ""))
        plans.append(
            {
                "index": idx,
                "scene_type": f"International event #{event.get('rank', idx - 1)}",
                "voiceover": f"第{event.get('rank', idx - 1)}則，{title_text}。",
                "on_screen_text": title_text,
                "primary_stat": "國際",
                "secondary_stat": "",
                "summary": summary_text,
                "reason": impact_text or "Highlight the event itself and why it matters to Taiwan.",
                "tags": ["International", "Geopolitics", "Taiwan impact"],
                "image_direction": f"Cinematic geopolitical editorial image inspired by this event: {title_text}. Show real-world atmosphere, diplomatic tension or conflict, premium magazine style, vertical 9:16. No visible text, logos, UI screenshots, or infographic boxes.",
            }
        )

    return plans


# ---------------------------------------------------------------------------
# Daily Trend (KPOP / Anime / etc.) – parser, script builder, scene planner
# ---------------------------------------------------------------------------
def parse_daily_trend_digest(source_text: str) -> dict:
    """Parse a daily trend digest (from daily_trend.py MiniMax output) into structured dict."""
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    # Try to extract topic name and date from report header
    topic_name = ""
    report_date = ""
    topic_match = re.search(r"([\w\s]+?)每日趨勢報告", text)
    if topic_match:
        topic_name = clean_field_text(topic_match.group(1))
    date_match = re.search(r"日期[：:]\s*(\d{4}-\d{1,2}-\d{1,2})", text)
    if date_match:
        parts = date_match.group(1).split("-")
        report_date = f"{parts[0]}年{int(parts[1])}月{int(parts[2])}日"
    if not report_date:
        report_date = extract_report_date(source_text)

    # Extract individual items: 【標題】followed by summary text
    items = []
    # Pattern: 【title】 followed by text until next 【 or 今日重點 or end
    item_pattern = re.compile(
        r"【([^】]+)】\s*\n(.*?)(?=\n\s*【|\n\s*今日重點|\n\s*={3,}|\Z)",
        re.S,
    )
    for idx, m in enumerate(item_pattern.finditer(text), start=1):
        title = clean_field_text(m.group(1))
        summary = clean_field_text(m.group(2).strip())
        if title:
            items.append({"rank": idx, "title": title, "summary": summary})

    # Extract 今日重點 summary
    overall_summary = ""
    summary_match = re.search(r"今日重點[：:]\s*(.+?)(?:\n\s*={3,}|\Z)", text, re.S)
    if summary_match:
        overall_summary = clean_field_text(summary_match.group(1).strip())

    if not topic_name:
        # Guess from content
        lower = text.lower()
        if any(kw in lower for kw in ["kpop", "韓流", "韓團", "韓國"]):
            topic_name = "KPOP 韓流"
        elif any(kw in lower for kw in ["anime", "動漫", "動畫"]):
            topic_name = "日本動漫"
        else:
            topic_name = "每日趨勢"

    return {
        "content_type": "daily_trend",
        "header": f"{topic_name} 每日趨勢報告",
        "report_date": report_date,
        "topic_name": topic_name,
        "summary": overall_summary or short_text(" ".join(item["title"] for item in items[:3]), 120),
        "items": items,
    }


def build_daily_trend_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    """Build a narration script from a daily trend digest."""
    items = parsed.get("items", [])
    topic_name = parsed.get("topic_name", "每日趨勢")
    subject = resolve_subject(custom_subject, f"{topic_name}每日趨勢")
    report_date = parsed_report_date(parsed)

    if not items:
        if report_date:
            return subject, f"今天是{report_date}，來看看{topic_name}有什麼新消息。"
        return subject, f"今天來看看{topic_name}有什麼新消息。"

    count_map = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}
    ordinal_map = {1: "第一", 2: "第二", 3: "第三", 4: "第四", 5: "第五", 6: "第六", 7: "第七", 8: "第八"}

    visible_items = items[:6]
    count_label = count_map.get(len(visible_items), str(len(visible_items)))
    intro = f"今天是{report_date}，{topic_name}快報，帶你看{count_label}則最新消息。" if report_date else f"{topic_name}快報，帶你看{count_label}則最新消息。"
    sections = [intro]

    for idx, item in enumerate(visible_items, start=1):
        title = clean_field_text(item.get("title", "")) or f"第{idx}則消息"
        summary = ensure_cn_punctuation(clean_field_text(item.get("summary", "")))

        lines = [f"{ordinal_map.get(idx, f'第{idx}')}則，{title}。"]
        if summary:
            # Split long summaries into manageable sentences
            sentences = re.split(r"(?<=[。！？])", summary)
            for sent in sentences[:3]:
                sent = sent.strip()
                if sent:
                    lines.append(sent)

        polished = [polish_daily_news_line(line) for line in lines if line.strip()]
        sections.append(" ".join(line for line in polished if line))

    # Add closing with overall summary if available
    overall = parsed.get("summary", "")
    if overall:
        sections.append(f"總結一下，{ensure_cn_punctuation(clean_field_text(overall))}")
    else:
        sections.append(f"以上就是今天的{topic_name}快報，我們明天見。")

    return subject, "\n\n".join(section for section in sections if section.strip())


def build_scene_plan_from_daily_trend(parsed: dict, title: str, script: str) -> list[dict]:
    """Build scene plan for a daily trend digest video."""
    items = parsed.get("items", [])
    topic_name = parsed.get("topic_name", "每日趨勢")

    if not items:
        return []

    # Determine visual style based on topic
    lower_topic = topic_name.lower()
    if any(kw in lower_topic for kw in ["kpop", "韓流", "韓團"]):
        opener_style = "Vibrant K-POP stage lighting, neon colors, concert atmosphere, idol silhouettes, premium entertainment news feel"
        item_style = "K-POP idol glamour shot, stage lighting, neon aesthetic, magazine cover quality"
        tags = ["K-POP", "Entertainment", "Trending"]
    elif any(kw in lower_topic for kw in ["anime", "動漫", "動畫"]):
        opener_style = "Dramatic anime-inspired artwork, vibrant colors, manga aesthetic, Japanese pop culture energy, premium editorial look"
        item_style = "Anime-inspired cinematic artwork, vibrant Japanese pop culture aesthetic, dramatic lighting"
        tags = ["Anime", "Entertainment", "Japanese Culture"]
    else:
        opener_style = "Modern trending news opener, social media energy, dynamic collage, premium editorial look"
        item_style = "Modern editorial photography, trending topic, premium social media aesthetic"
        tags = ["Trending", "Daily", "News"]

    plans = [
        {
            "index": 1,
            "scene_type": f"{topic_name} opener",
            "voiceover": split_script_segments(script, 1)[0] if script else title,
            "on_screen_text": title or f"{topic_name} 每日趨勢",
            "primary_stat": "",
            "secondary_stat": "",
            "summary": f"Dynamic opening for {topic_name} daily trend report.",
            "reason": f"Set the mood for today's {topic_name} trending news.",
            "tags": tags,
            "image_direction": f"Cinematic vertical 9:16 opener. {opener_style}. No visible text, logos, or UI elements.",
        }
    ]

    for idx, item in enumerate(items[:5], start=2):
        title_text = clean_field_text(item.get("title", ""))
        summary_text = clean_field_text(item.get("summary", ""))
        plans.append(
            {
                "index": idx,
                "scene_type": f"{topic_name} trend #{item.get('rank', idx - 1)}",
                "voiceover": f"第{item.get('rank', idx - 1)}則，{title_text}。",
                "on_screen_text": title_text,
                "primary_stat": topic_name,
                "secondary_stat": "",
                "summary": summary_text or title_text,
                "reason": f"Highlight this trending {topic_name} news item.",
                "tags": tags,
                "image_direction": f"Cinematic editorial image inspired by: {title_text}. {item_style}. Vertical 9:16, premium quality. No visible text, logos, UI screenshots, or infographic boxes.",
            }
        )

    return plans


def build_market_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    themes = parsed.get("themes", [])
    subject = resolve_subject(custom_subject, "今日市場主題")
    if not themes:
        return subject, "今天市場主要圍繞風險情緒、科技股與資金輪動。"

    beats = ["今天市場，先看最強主題。"]
    for idx, theme in enumerate(themes[:4], start=1):
        prefix = "美股" if "US" in theme.get("region", "") else "台股"
        beats.append(f"{prefix}第{idx}個重點，{short_text(theme['title'], 28)}。")
        beats.append(short_text(theme.get("summary", ""), 32) or "資金正在往這個方向集中。")
    beats.append("後面就看資金有沒有延續。")
    return subject, "\n".join(short_text(beat, 36) for beat in beats if beat.strip())


def _try_llm_voiceover_v2(parsed: dict, custom_subject: str = "") -> tuple[str, str] | None:
    content_type = parsed.get("content_type", "generic")
    items = (
        parsed.get("projects")
        or parsed.get("events")
        or parsed.get("themes")
        or []
    )
    if not items:
        return None

    payload = {
        "content_type": content_type,
        "header": parsed.get("header", ""),
        "summary": parsed.get("summary", ""),
        "items": items[:5],
    }
    fallback_subject = resolve_subject(custom_subject, _voiceover_prompt_type(content_type) or "短影音重點整理")
    style_rules = {
        "github_weekly": [
            "聚焦前三個專案，先講本週趨勢，再點出各專案為什麼值得看。",
            "一定要保留 repo 名稱原樣，不要改寫。",
            "結尾一句帶到後面還值得追的專案即可。",
        ],
        "daily_tech_news": [
            "像科技新聞主播，先講今天最值得看的幾則新聞，再講影響。",
            "每一則用一句發生什麼，一句為什麼重要。",
        ],
        "market_report": [
            "像市場快報主播，先講今天市場主軸，再講兩到三個最重要的交易焦點。",
            "優先使用美股、台股、能源、殖利率、台積電、AI供應鏈這類市場語彙。",
            "不要做專案排行口吻，不要說第1名、第2名。",
            "不要自己補不存在的數字、成長率、營收或價格。",
            "每一句都要有承接感，像主播在往下帶，不要寫成投影片條列。",
            "至少要有一個明確轉折，例如先講美股，再帶到台股，最後收觀察重點。",
            "寧可多交代一點脈絡，也不要只剩下標題式摘要。",
            "結尾只留一句接下來觀察什麼。",
        ],
        "generic": [
            "像短影音主持人口播，先講主題，再講兩到三個重點。",
        ],
    }
    rule_text = "\n".join(f"- {line}" for line in style_rules.get(content_type, style_rules["generic"]))
    prompt = f"""
你是一位擅長做繁體中文短影音口播稿的編輯。

請把下面的結構化資料，改寫成自然、好講、像主持人口播的短影音講稿。

規則：
- 只輸出 JSON。
- 口播稿要是繁體中文。
- 只能使用輸入資料裡已經出現的資訊，不能補充新事實、數字、推論或背景知識。
- 如果輸入資料沒有寫，就不要自己加。
- 保留原本的專案名、公司名、產品名、股票代號、大小寫、括號、連字號、數字和星數格式。
- script 請拆成 6 到 9 行，每行一句完整但口語的短句，適合 TTS 逐句念。
- script 總長目標大約 420 到 620 個中文字，適合約 2.5 到 3 分鐘的口播。
- 每行大約 24 到 60 個中文字，寧可自然，也不要硬切成碎句。
- 每一行結尾都請加上中文標點。
- 行與行之間要有連接感，讓觀眾聽起來像一段完整口播，而不是一條一條摘要。
- 不要使用 markdown、項目符號、編號。

內容型態附加要求：
{rule_text}

輸出格式：
{{
  "subject": "簡短主題",
  "script": "逐行口播稿，用 \\n 分隔"
}}

預設主題：
{fallback_subject}

結構化資料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    result = _call_ollama_json(
        (VOICEOVER_PRIMARY_MODEL, VOICEOVER_FALLBACK_MODEL),
        prompt,
        timeout=240,
        temperature=0.35,
    )
    if not isinstance(result, dict):
        return None

    subject = resolve_subject(str(result.get("subject", "") or ""), fallback_subject)
    script = str(result.get("script", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in script.split("\n") if line.strip()]
    normalized_lines = []
    for line in lines:
        cleaned_line = clean_field_text(line)
        if cleaned_line and not re.search(r"[。！？!?]$", cleaned_line):
            cleaned_line += "。"
        normalized_lines.append(cleaned_line)
    script = "\n".join(normalized_lines)
    if content_type == "market_report":
        script = polish_market_report_script(script)
        script = enrich_market_report_script(script, parsed)
    if not script or looks_garbled(script):
        return None
    if len(normalized_lines) < 5:
        return None
    if not _voiceover_has_required_refs(script, parsed):
        return None
    return subject, script


def _try_llm_market_voiceover_v1(parsed: dict, custom_subject: str = "") -> tuple[str, str] | None:
    themes = parsed.get("themes") or []
    if not themes:
        return None

    compact_themes = []
    for theme in themes[:4]:
        compact_themes.append(
            {
                "region": clean_field_text(str(theme.get("region", "") or "")),
                "title": clean_field_text(str(theme.get("title", "") or "")),
                "summary": clean_field_text(str(theme.get("summary", "") or "")),
                "bullets": [
                    clean_field_text(str(bullet))
                    for bullet in (theme.get("bullets") or [])[:4]
                    if clean_field_text(str(bullet))
                ],
            }
        )

    payload = {
        "header": clean_field_text(str(parsed.get("header", "") or "")),
        "summary": clean_field_text(str(parsed.get("summary", "") or "")),
        "themes": compact_themes,
    }
    fallback_subject = resolve_subject(custom_subject, "市場主題報告")
    prompt = f"""
你是一位擅長寫繁體中文財經短影音口播稿的編輯。

你的工作不是做摘要條列，而是把已經整理好的市場重點，改寫成「短影音快報」口氣。

重要規則：
- 只輸出 JSON。
- 只能使用輸入資料裡已經明確出現的資訊，不能新增任何數字、日期、價格、公司、成長率、結論或背景知識。
- 不要腦補，不要推測，不要替原文補齊缺漏。
- 保留專有名詞、公司名、產品名、股票代號與英文名稱。
- 口氣要像短影音財經主播，不要像研究報告，也不要像條列摘要。
- 開頭第一句要直接告訴觀眾「今天市場在交易什麼」。
- 內容要有承接感，常用「先看…」「再看…」「最後看…」「總結一句」這種主播轉場。
- script 請拆成 10 到 14 行，每行一句完整口語短句，用 \\n 分隔。
- 每行結尾都要有中文標點。
- script 總長目標約 480 到 680 個中文字。
- 優先講主線與資金焦點，不要把所有細節都塞進去。
- 如果某個 bullet 很像表格或資料欄位，請改寫成自然口語，但不能改變事實。
- 不要使用 markdown、項目符號、編號。

輸出格式：
{{
  "subject": "簡短主題",
  "script": "逐行口播稿，用 \\n 分隔"
}}

預設主題：
{fallback_subject}

結構化資料：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    result = _call_ollama_json(
        (MARKET_VOICEOVER_PRIMARY_MODEL, MARKET_VOICEOVER_FALLBACK_MODEL),
        prompt,
        timeout=300,
        temperature=0.35,
    )
    if not isinstance(result, dict):
        return None

    subject = resolve_subject(str(result.get("subject", "") or ""), fallback_subject)
    script = str(result.get("script", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in script.split("\n") if line.strip()]
    normalized_lines = []
    for line in lines:
        cleaned_line = clean_field_text(line)
        if cleaned_line and not re.search(r"[。！？!?]$", cleaned_line):
            cleaned_line += "。"
        normalized_lines.append(cleaned_line)
    script = "\n".join(normalized_lines)
    if len(normalized_lines) < 8:
        return None
    return subject, script


def build_market_script_v2(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    themes = parsed.get("themes", [])
    subject = resolve_subject(custom_subject, "市場主題報告")
    if not themes:
        return subject, "今天市場先看資金在交易什麼，再看後面主線有沒有延續。"

    us_theme = next((theme for theme in themes if "US" in theme.get("region", "")), None)
    tw_theme = next((theme for theme in themes if "Taiwan" in theme.get("region", "")), None)
    other_themes = [theme for theme in themes if theme not in {us_theme, tw_theme}]

    beats = ["今天市場的主軸，還是資金在能源和科技之間換線。"]
    if us_theme:
        us_title = short_text(us_theme.get("title", ""), 26)
        us_summary = short_text(us_theme.get("summary", ""), 42)
        beats.append(f"先看美股，市場在交易 {us_title}。")
        if us_summary:
            beats.append(us_summary if re.search(r"[。！？!?]$", us_summary) else f"{us_summary}。")
    if tw_theme:
        tw_title = short_text(tw_theme.get("title", ""), 26)
        tw_summary = short_text(tw_theme.get("summary", ""), 42)
        beats.append(f"再看台股，焦點落在 {tw_title}。")
        if tw_summary:
            beats.append(tw_summary if re.search(r"[。！？!?]$", tw_summary) else f"{tw_summary}。")
    if other_themes:
        extra = short_text(other_themes[0].get("title", ""), 26)
        if extra:
            beats.append(f"另外，市場也開始關注 {extra} 這條線。")
    beats.append("接下來要看的，就是資金輪動能不能延續。")

    normalized = []
    for beat in beats:
        cleaned = short_text(beat, 44)
        if cleaned and not re.search(r"[。！？!?]$", cleaned):
            cleaned += "。"
        if cleaned:
            normalized.append(cleaned)
    return subject, "\n".join(normalized)


def build_market_script_v3(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    themes = parsed.get("themes", [])
    subject = resolve_subject(custom_subject, "市場主題報告")
    report_date = parsed_report_date(parsed)
    if not themes:
        if report_date:
            return subject, f"今天是{report_date}，市場最重要的，不是單一題材，而是資金、風險和科技主線正在同時拉扯。"
        return subject, "今天市場最重要的，不是單一題材，而是資金、風險和科技主線正在同時拉扯。"

    us_themes = [theme for theme in themes if "US" in str(theme.get("region", ""))]
    tw_themes = [theme for theme in themes if "Taiwan" in str(theme.get("region", ""))]
    # If region detection missed, distribute themes evenly
    if not us_themes and not tw_themes and themes:
        mid = max(1, len(themes) // 2)
        us_themes = themes[:mid]
        tw_themes = themes[mid:]

    ordinal_map = {1: "第一", 2: "第二", 3: "第三", 4: "第四", 5: "第五", 6: "第六"}

    def _clean_bullet(text: str) -> str:
        """Clean a bullet/summary line: strip labels, table chars, junk prefixes."""
        cleaned = re.sub(r"\s+", " ", clean_field_text(text)).strip()
        if not cleaned:
            return ""
        # Remove table drawing chars
        cleaned = re.sub(r"[┌┐└┘│├┤┬┴─═╞╡╤╧╪]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Remove common label prefixes
        cleaned = re.sub(
            r"^(核心邏輯|分析師觀點|相關個股表現|產能狀況|財務展望|重要利多|潛在風險|事件概要|地緣分析|對台灣科技供應鏈的影響|總體經濟觀察|個股重點|上週大盤表現|總結|注意)\s*[：:]\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        # Skip junk lines
        if "資料來源" in cleaned or "Report Time" in cleaned:
            return ""
        if any(c in cleaned for c in "┌┐└┘│├┤┬┴←"):
            return ""
        if re.search(r"\([0-9]{4}\)\s*\+[0-9.]+%|\+[0-9.]+%\+", cleaned):
            return ""
        if cleaned.startswith(("📈", "☐", "■")):
            return ""
        return cleaned

    def _get_theme_sentences(theme: dict) -> list[str]:
        """Extract full, clean sentences from a theme's summary and bullets."""
        sentences: list[str] = []
        seen = set()
        summary = _clean_bullet(theme.get("summary", ""))
        if summary and len(summary) >= 8:
            sentences.append(ensure_cn_punctuation(summary))
            seen.add(summary)
        for bullet in theme.get("bullets", []):
            cleaned = _clean_bullet(str(bullet))
            if not cleaned or len(cleaned) < 8 or cleaned in seen:
                continue
            seen.add(cleaned)
            sentences.append(ensure_cn_punctuation(cleaned))
        return sentences

    visible_themes = (us_themes[:2] + tw_themes[:2])[:5]
    count_label = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五"}.get(len(visible_themes), str(len(visible_themes)))

    sections: list[str] = []

    # --- Intro ---
    has_us = bool(us_themes)
    has_tw = bool(tw_themes)
    if report_date:
        if has_us and has_tw:
            sections.append(f"今天是{report_date}，市場有{count_label}條主線值得注意，先看美股，再看台股。")
        else:
            sections.append(f"今天是{report_date}，市場有{count_label}條主線值得注意。")
    else:
        if has_us and has_tw:
            sections.append(f"今天市場有{count_label}條主線值得注意，先看美股，再看台股。")
        else:
            sections.append(f"今天市場有{count_label}條主線值得注意。")

    # --- Per-theme sections ---
    theme_idx = 0
    region_transitions = {
        "us_first": "先看美股，",
        "us_more": "美股另一條主線，",
        "tw_first": "拉回台股，",
        "tw_more": "台股還有一條線值得注意，",
    }

    for region_label, region_themes in [("us", us_themes[:2]), ("tw", tw_themes[:2])]:
        for r_idx, theme in enumerate(region_themes):
            theme_idx += 1
            title = _clean_bullet(theme.get("title", "")) or "市場主線"
            sentences = _get_theme_sentences(theme)

            ordinal = ordinal_map.get(theme_idx, f"第{theme_idx}")
            if region_label == "us":
                transition = region_transitions["us_first"] if r_idx == 0 else region_transitions["us_more"]
            else:
                transition = region_transitions["tw_first"] if r_idx == 0 else region_transitions["tw_more"]

            lines: list[str] = []
            lines.append(f"{ordinal}條，{transition}{title}。")

            if sentences:
                lines.append(sentences[0])
            if len(sentences) > 1:
                lines.append(f"這裡最需要注意的是，{sentences[1].rstrip('。')}。")
            if len(sentences) > 2:
                lines.append(f"另外，{sentences[2].rstrip('。')}。")
            if len(sentences) > 3:
                lines.append(f"再來，{sentences[3].rstrip('。')}。")

            polished = [polish_daily_news_line(line) for line in lines if line.strip()]
            sections.append(" ".join(line for line in polished if line))

    # --- Closing ---
    if has_us and has_tw:
        sections.append("以上就是今天市場最重要的幾條主線，接下來繼續追蹤資金輪動的方向。")
    elif has_us:
        sections.append("以上就是今天美股最重要的主線，後續看資金會不會延續。")
    else:
        sections.append("以上就是今天台股最重要的主線，後續繼續觀察。")

    return subject, "\n\n".join(section for section in sections if section.strip())


def build_generic_script(source_text: str, custom_subject: str = "") -> tuple[str, str]:
    lines = []
    for raw in source_text.replace("\ufeff", "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[=\-─_│\s]+", line):
            continue
        if line.startswith(("🔗", "📦", "▸", "⚠️")):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            continue
        if "Fork" in line and "⭐" in line:
            continue
        lines.append(line)

    intro = lines[0] if lines else "本週有幾個很值得關注的開源專案。"
    supporting = [line for line in lines[1:] if len(line) >= 12][:3]
    subject = resolve_subject(custom_subject, short_text(intro, 42))
    script = normalize_script(" ".join([intro] + supporting))
    return subject, short_text(script, 620)


# ─── MiniMax LLM-based normalizer ────────────────────────────────────────────
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_URL = "https://api.minimax.io/v1/chat/completions"
MINIMAX_MODEL = "MiniMax-M2"

_MINIMAX_SYSTEM_PROMPT = """你是一個報告結構化工具。你的任務是將原始新聞/市場報告轉換為固定格式的 JSON。

根據報告內容，判斷它屬於以下哪一種類型，然後輸出對應的 JSON 結構：

## 類型一：國際新聞報告（international_brief）
特徵：有「國際新聞」、「國際情勢」等字眼，事件包含「對台灣影響」
```json
{
  "content_type": "international_brief",
  "report_date": "2026年4月8日",
  "events": [
    {
      "rank": 1,
      "title": "事件標題（完整中文）",
      "summary": "事件摘要，2-3句話概述核心內容",
      "details": ["關鍵細節1（完整句子）", "關鍵細節2", "關鍵細節3"],
      "taiwan_impact": "對台灣的影響分析（完整句子）"
    }
  ]
}
```

## 類型二：科技情報報告（daily_tech_news）
特徵：有「科技情報」、「Tech Intelligence」等字眼
```json
{
  "content_type": "daily_tech_news",
  "report_date": "2026年4月8日",
  "events": [
    {
      "rank": 1,
      "title": "事件標題",
      "summary": "事件核心內容摘要，2-3句話",
      "highlights": ["重點1（完整句子）", "重點2", "重點3"]
    }
  ]
}
```

## 類型三：市場主題報告（market_report）
特徵：有「市場」、「台股」、「美股」、「Market」等字眼
```json
{
  "content_type": "market_report",
  "report_date": "2026年4月8日",
  "themes": [
    {
      "region": "Taiwan Markets 或 US Markets",
      "title": "主題標題",
      "summary": "主題摘要，2-3句話",
      "bullets": ["要點1（完整句子）", "要點2", "要點3"]
    }
  ]
}
```

## 類型四：GitHub 週報（github_weekly）
特徵：有「GitHub」、星數統計、專案介紹
```json
{
  "content_type": "github_weekly",
  "report_date": "2026年4月6日",
  "summary": "本週整體趨勢摘要",
  "projects": [
    {
      "name": "專案名稱",
      "stats": "本週 +13476 ★（總 16748 ★）",
      "what": "這個專案做什麼（完整句子）",
      "why_hot": "為什麼變熱門（完整句子）",
      "who_for": "適合什麼人（完整句子）",
      "risks": "風險或門檻（完整句子）"
    }
  ]
}
```

## 類型五：每日趨勢報告（daily_trend）
特徵：有「每日趨勢報告」、「每日摘要」、「KPOP」、「韓流」、「動漫」、「動畫」等字眼，內容是娛樂/流行文化新聞摘要
```json
{
  "content_type": "daily_trend",
  "report_date": "2026年4月9日",
  "topic_name": "KPOP 韓流 或 日本動漫",
  "summary": "今日整體趨勢總結",
  "items": [
    {
      "rank": 1,
      "title": "新聞標題（完整中文）",
      "summary": "新聞摘要，2-3句話概述核心內容"
    }
  ]
}
```

## 重要規則：
1. **只輸出 JSON**，不要加任何說明文字、markdown 標記或程式碼區塊標記
2. 每個欄位都要是**完整句子**，不要有斷句或片段
3. 保留原文的數據和專有名詞（股票代號、公司名稱、金額等）
4. events/themes/projects/items 取前 5 個最重要的
5. summary 和每個 bullet/detail/highlight 都要是獨立可讀的完整敘述
6. report_date 從報告中提取，格式為 "YYYY年M月D日"
"""


def _call_minimax_normalize(source_text: str) -> dict | None:
    """Call MiniMax API to normalize raw report into structured JSON."""
    if not MINIMAX_API_KEY or not source_text or len(source_text) < 50:
        return None
    try:
        payload = {
            "model": MINIMAX_MODEL,
            "messages": [
                {"role": "system", "content": _MINIMAX_SYSTEM_PROMPT},
                {"role": "user", "content": source_text[:12000]},  # limit input size
            ],
            "temperature": 0.1,
            "max_completion_tokens": 8192,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            MINIMAX_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        # Strip <think>...</think> reasoning tags (MiniMax-M2 is a reasoning model)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        # Strip markdown code block markers if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
        parsed = json.loads(content)
        ct = parsed.get("content_type", "")
        if ct not in {"international_brief", "daily_tech_news", "market_report", "github_weekly", "daily_trend"}:
            return None
        # Inject report_date into standard location
        if "report_date" not in parsed:
            parsed["report_date"] = extract_report_date(source_text)
        print(f"[MiniMax] Successfully normalized as {ct}, "
              f"items={len(parsed.get('events', parsed.get('themes', parsed.get('projects', []))))}")
        return parsed
    except Exception as e:
        print(f"[MiniMax] API call failed: {e}")
        return None


def build_script_from_source(source_text: str, custom_subject: str = "", content_mode: str = "auto") -> tuple[str, str, dict]:
    forced_type = (content_mode or "auto").strip()

    # ── Step 1: Try MiniMax LLM normalization first ──
    parsed = _call_minimax_normalize(source_text)

    # ── Step 2: Fallback to regex parsers if MiniMax failed ──
    if parsed is None:
        heuristic_type = forced_type if forced_type in {"github_weekly", "daily_tech_news", "market_report", "international_brief", "daily_trend"} else detect_content_type(source_text)
        if heuristic_type == "market_report":
            parsed = parse_market_report_v2(source_text)
        elif heuristic_type == "github_weekly":
            parsed = parse_ranked_projects(source_text)
        elif heuristic_type == "daily_tech_news":
            parsed = parse_daily_news_events(source_text)
        elif heuristic_type == "international_brief":
            parsed = parse_international_brief_v2(source_text)
        elif heuristic_type == "daily_trend":
            parsed = parse_daily_trend_digest(source_text)
        else:
            parsed = _try_llm_normalize(source_text)
        if parsed is None:
            parsed = {"content_type": "generic", "projects": []}

    # ── Step 3: Build script from parsed data ──
    content_type = parsed.get("content_type", "generic")
    if content_type in {"github_weekly", "daily_tech_news", "market_report", "international_brief", "daily_trend"}:
        llm_voiceover = None
    else:
        llm_voiceover = _try_llm_voiceover_v2(parsed, custom_subject)
    if llm_voiceover is not None:
        subject, script = llm_voiceover
    elif content_type == "github_weekly":
        subject, script = build_ranked_script(parsed, custom_subject)
    elif content_type == "daily_tech_news":
        subject, script = build_daily_news_script(parsed, custom_subject)
    elif content_type == "market_report":
        subject, script = build_market_script_v3(parsed, custom_subject)
    elif content_type == "international_brief":
        subject, script = build_international_brief_script_v2(parsed, custom_subject)
    elif content_type == "daily_trend":
        subject, script = build_daily_trend_script(parsed, custom_subject)
    else:
        subject, script = build_generic_script(source_text, custom_subject)
    script = apply_report_date_to_script(script, parsed)
    return subject, script, parsed


def split_script_segments(script: str, max_segments: int = 8) -> list[str]:
    normalized = script.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    line_chunks = [line.strip() for line in normalized.split("\n") if line.strip()]
    if len(line_chunks) > 1:
        return line_chunks[:max_segments]

    chunks = re.split(r"(?<=[。！？!?])\s*", normalized)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks[:max_segments]

def infer_project_tags(project: dict) -> list[str]:
    corpus = " ".join([
        project.get("name", ""),
        project.get("what", ""),
        project.get("why_hot", ""),
    ]).lower()
    candidates = [
        ("AI Agent", ["agent", "ai agent"]),
        ("AI Coding", ["claude code", "cursor", "codex", "coding"]),
        ("Workflow", ["workflow", "automation", "productivity"]),
        ("Finance", ["trading", "finance", "??", "??"]),
        ("Offline", ["offline", "??", "??"]),
        ("Voice", ["voice", "speech", "tts", "asr"]),
        ("Research", ["research", "memory", "wiki"]),
    ]
    tags = []
    for label, keywords in candidates:
        if any(keyword in corpus for keyword in keywords):
            tags.append(label)
    if not tags:
        tags.append("Open Source")
    return tags[:4]


def compact_reason(text: str, fallback: str) -> str:
    cleaned = short_text(text, 42)
    return cleaned or fallback



def build_scene_plan_from_ranked_projects(parsed: dict, title: str, script: str) -> list[dict]:
    projects = parsed["projects"]
    plans = []
    if not projects:
        return plans

    top_three = projects[:3]
    first_segment = split_script_segments(script, 1)
    plans.append(
        {
            "index": 1,
            "scene_type": "Weekly cover",
            "voiceover": first_segment[0] if first_segment else title,
            "on_screen_text": "GitHub Weekly Top 3",
            "primary_stat": parsed.get("header", "GitHub Weekly Open Source"),
            "secondary_stat": " / ".join(project["name"] for project in top_three),
            "summary": short_text(parsed.get("summary", "This week focuses on AI Agent, AI Coding, and practical open source tools."), 46),
            "reason": "Start with the top 3 repos and then scan the rest.",
            "tags": ["GitHub Weekly", "Top 3", "AI Agent"],
            "image_direction": "Cinematic tech-news opener. Futuristic newsroom mood. Holographic screens, glowing code panels, energetic but clean composition. No visible text or logos.",
        }
    )

    for index, project in enumerate(projects[:3], start=2):
        total_stars = safe_total_stars(project.get("stats", ""))
        weekly_gain = safe_weekly_gain(project.get("stats", ""))
        short_voice = f"第{project['rank']}名，{project['name']}。"
        what = short_text(project.get("what", ""), 24)
        if what:
            short_voice += f"{what}。"
        plans.append(
            {
                "index": index,
                "scene_type": f"Repo spotlight #{project['rank']}",
                "voiceover": short_voice,
                "on_screen_text": project["name"],
                "primary_stat": total_stars or "Stars",
                "secondary_stat": weekly_gain or "Weekly momentum",
                "summary": compact_reason(project.get("what", ""), "Open-source project in the weekly spotlight."),
                "reason": compact_reason(project.get("why_hot", ""), "Rapid weekly growth and strong developer attention."),
                "tags": infer_project_tags(project),
                "image_direction": "Cinematic product-style hero image for a software project. Show developers, glowing terminals, floating agent workflows, code holograms, modern studio lighting, shallow depth of field. No text, no dashboard screenshots, no watermark.",
            }
        )

    follow_up = projects[3:6]
    if follow_up:
        plans.append(
            {
                "index": len(plans) + 1,
                "scene_type": "Fast movers",
                "voiceover": f"另外，{'、'.join(project['name'] for project in follow_up)} 也在升溫。",
                "on_screen_text": "Fast movers",
                "primary_stat": " / ".join(project["name"] for project in follow_up[:2]),
                "secondary_stat": follow_up[2]["name"] if len(follow_up) > 2 else "",
                "summary": "Several more repos are rapidly climbing this week.",
                "reason": "Show momentum, trend energy, and multiple projects rising together.",
                "tags": [tag for project in follow_up for tag in infer_project_tags(project)][:4],
                "image_direction": "Dynamic collage of multiple AI and developer-tool concepts, fast-moving motion streaks, layered interfaces, startup energy, editorial lighting. No text, no logos, no infographic boxes.",
            }
        )

    plans.append(
        {
            "index": len(plans) + 1,
            "scene_type": "Closing CTA",
            "voiceover": f"如果你只想先追三個，就先看 {'、'.join(project['name'] for project in projects[:3])}。",
            "on_screen_text": "Top 3 to watch",
            "primary_stat": " / ".join(project["name"] for project in projects[:2]),
            "secondary_stat": projects[2]["name"] if len(projects) > 2 else "",
            "summary": "End on the strongest three names from this week's ranking.",
            "reason": "Confident closing shot with a premium editorial look.",
            "tags": ["Top 3", "Open Source", "AI"],
            "image_direction": "Confident closing scene, premium tech editorial mood, strong depth, dramatic rim light, creator energy, polished composition. No text, no logos, no watermark.",
        }
    )

    return plans


def build_scene_plan_generic(subject: str, title: str, segments: list[str]) -> list[dict]:
    scene_types = ["Hook cover", "Key idea", "Example", "Second example", "Takeaway", "Closing CTA"]
    plans = []
    for idx, segment in enumerate(segments, start=1):
        plans.append(
            {
                "index": idx,
                "scene_type": scene_types[min(idx - 1, len(scene_types) - 1)],
                "voiceover": segment,
                "on_screen_text": short_text(segment, 38),
                "primary_stat": "",
                "secondary_stat": "",
                "summary": short_text(segment, 48),
                "reason": "",
                "tags": ["Shorts"],
                "image_direction": "Vertical cinematic illustration for a modern short video. Focus on one clear visual idea, strong lighting, rich atmosphere, no visible text.",
            }
        )
    if not plans:
        plans.append(
            {
                "index": 1,
                "scene_type": "Hook cover",
                "voiceover": subject,
                "on_screen_text": title,
                "primary_stat": "",
                "secondary_stat": "",
                "summary": subject,
                "reason": "",
                "tags": ["Shorts"],
                "image_direction": "Vertical cinematic hero shot, premium social-video style, strong focal subject, no visible text.",
            }
        )
    return plans


def build_scene_plan_from_daily_news(parsed: dict, title: str, script: str) -> list[dict]:
    events = parsed.get("events", [])
    plans = []
    if not events:
        return plans
    plans.append(
        {
            "index": 1,
            "scene_type": "News opener",
            "voiceover": split_script_segments(script, 1)[0] if script else title,
            "on_screen_text": "Tech News Today",
            "primary_stat": "",
            "secondary_stat": "",
            "summary": "A fast-moving daily tech news rundown.",
            "reason": "Open with urgency and newsroom energy.",
            "tags": ["Tech News", "Breaking", "Daily Brief"],
            "image_direction": "Cinematic global tech newsroom, glowing world map, control room screens, satellite feeds, dramatic lighting, premium editorial look. No visible text or logos.",
        }
    )
    for idx, event in enumerate(events[:3], start=2):
        plans.append(
            {
                "index": idx,
                "scene_type": f"Headline spotlight #{event['rank']}",
                "voiceover": f"第{event['rank']}則，{event['title']}。",
                "on_screen_text": event["title"],
                "primary_stat": event.get("importance", ""),
                "secondary_stat": "",
                "summary": event.get("summary", ""),
                "reason": "Make it feel like a major global headline with one dominant visual idea.",
                "tags": ["Headline", "Technology", "Breaking"],
                "image_direction": f"Cinematic editorial image inspired by this headline: {event['title']}. Show the real-world domain and atmosphere implied by the story, dramatic but believable, premium news-magazine style. No text, no logos, no UI screenshots.",
            }
        )
    return plans


def build_scene_plan_from_international_brief(parsed: dict, title: str, script: str) -> list[dict]:
    events = parsed.get("events", [])
    plans = []
    if not events:
        return plans
    plans.append(
        {
            "index": 1,
            "scene_type": "Global news opener",
            "voiceover": split_script_segments(script, 1)[0] if script else title,
            "on_screen_text": "Global Brief",
            "primary_stat": "",
            "secondary_stat": "",
            "summary": "A fast global affairs briefing with cross-strait, energy, diplomacy, and security themes.",
            "reason": "Open with a high-stakes international news atmosphere.",
            "tags": ["Global", "Geopolitics", "Daily Brief"],
            "image_direction": "Cinematic international news opener, dramatic newsroom, global map lighting, diplomatic tension, premium editorial look. No visible text or logos.",
        }
    )
    for idx, event in enumerate(events[:4], start=2):
        plans.append(
            {
                "index": idx,
                "scene_type": f"International event #{event['rank']}",
                "voiceover": f"第{event['rank']}則，{event['title']}。",
                "on_screen_text": event["title"],
                "primary_stat": "國際",
                "secondary_stat": "",
                "summary": event.get("summary", ""),
                "reason": event.get("taiwan_impact", "") or "Highlight the event and why it matters to Taiwan.",
                "tags": ["International", "Geopolitics", "Taiwan impact"],
                "image_direction": f"Cinematic geopolitical editorial image inspired by this event: {event['title']}. Show real-world atmosphere, high stakes diplomacy or conflict, premium magazine style. No text, no logos, no UI screenshots.",
            }
        )
    return plans


def build_scene_plan_from_market_report(parsed: dict, title: str, script: str) -> list[dict]:
    themes = parsed.get("themes", [])
    plans = []
    if not themes:
        return plans
    plans.append(
        {
            "index": 1,
            "scene_type": "Market opener",
            "voiceover": split_script_segments(script, 1)[0] if script else title,
            "on_screen_text": "Market Theme Report",
            "primary_stat": "",
            "secondary_stat": "",
            "summary": "Global market sentiment, AI capital rotation, and the day's strongest sectors.",
            "reason": "Start like a premium macro market recap.",
            "tags": ["Markets", "Macro", "Trading"],
            "image_direction": "Cinematic financial-news opener, glowing exchange floor, abstract candlesticks, large data walls, dramatic market atmosphere, premium business editorial style. No visible text or logos.",
        }
    )
    for idx, theme in enumerate(themes[:4], start=2):
        market_region = theme.get("region", "Market")
        plans.append(
            {
                "index": idx,
                "scene_type": f"{market_region} theme",
                "voiceover": f"{market_region}，{theme['title']}。",
                "on_screen_text": theme["title"],
                "primary_stat": market_region,
                "secondary_stat": "",
                "summary": theme.get("summary", ""),
                "reason": "Show sector leadership and capital rotation, not literal tables.",
                "tags": ["Markets", market_region, "AI Trade"],
                "image_direction": f"Cinematic market visual for {market_region}. Theme: {theme['title']}. Show traders, semiconductors, data screens, capital rotation, macro tension, premium Bloomberg-style atmosphere. No visible text, no tickers, no logos.",
            }
        )
    return plans


def build_scene_plan(subject: str, title: str, script: str, parsed: dict) -> list[dict]:
    content_type = parsed.get("content_type", "generic")
    if content_type == "github_weekly" and parsed.get("projects"):
        return build_scene_plan_from_ranked_projects(parsed, title, script)
    if content_type == "daily_tech_news":
        return build_scene_plan_from_daily_news(parsed, title, script)
    if content_type == "international_brief":
        return build_scene_plan_from_international_brief_v2(parsed, title, script)
    if content_type == "market_report":
        return build_scene_plan_from_market_report(parsed, title, script)
    if content_type == "daily_trend":
        return build_scene_plan_from_daily_trend(parsed, title, script)
    return build_scene_plan_generic(subject, title, split_script_segments(script))


def build_image_prompts(scene_plan: list[dict]) -> list[str]:
    prompts = []
    for scene in scene_plan:
        visual_anchor = scene.get("summary") or scene.get("on_screen_text") or scene["scene_type"]
        tag_line = ", ".join(scene.get("tags", []))
        prompts.append(
            "Create a vertical 9:16 cinematic image for a short-form video. "
            f"Scene: {scene['scene_type']}. "
            f"Main idea: {visual_anchor}. "
            f"Context tags: {tag_line}. "
            f"Creative direction: {scene['image_direction']} "
            "Look premium, modern, high-contrast, visually striking, social-media friendly. "
            "Do not render readable text, letters, captions, logos, UI screenshots, watermarks, infographic boxes, or dashboard layouts. "
            "Focus on atmosphere, subject, depth, lighting, and a clean cinematic composition."
        )
    return prompts


def build_preview_payload(
    account: dict,
    custom_subject: str,
    custom_title: str,
    custom_description: str,
    custom_script: str,
    source_text: str,
    content_mode: str = "auto",
    output_mode: str = "full_video",
) -> dict:
    parsed = {"projects": []}
    if custom_script:
        subject = resolve_subject(custom_subject, short_text(custom_script, 40))
        script = normalize_script(custom_script)
    else:
        subject, script, parsed = build_script_from_source(source_text, custom_subject, content_mode)

    if looks_garbled(subject):
        subject = fallback_subject(parsed, script)
    title, description = resolve_metadata(custom_title, custom_description, subject, script, parsed)
    segments = split_script_segments(script)
    scene_plan = build_scene_plan(subject, title, script, parsed)

    return {
        "content_mode": content_mode or "auto",
        "output_mode": output_mode or "full_video",
        "subject": subject,
        "title": title,
        "description": description,
        "script": script,
        "custom_script": custom_script,
        "source_text": source_text,
        "segments": segments,
        "scene_plan": scene_plan,
        "account_id": account["id"],
        "account_name": account["nickname"],
        "created_at": now_text(),
    }


def begin_job(account: dict, action: str) -> tuple[bool, str]:
    with JOB_LOCK:
        if JOB_STATE["running"]:
            return False, "Another job is already running."

        JOB_STATE.update(
            {
                "running": True,
                "status": "running",
                "progress": 0,
                "stage": action,
                "started_at": now_text(),
                "finished_at": None,
                "account_id": account["id"],
                "account_name": account["nickname"],
                "content_mode": "auto",
                "output_mode": "full_video",
                "subject": "",
                "title": "",
                "description": "",
                "script": "",
                "script_path": None,
                "audio_path": None,
                "subtitle_path": None,
                "image_paths": [],
                "video_path": None,
                "uploaded_url": None,
                "error": None,
                "logs": [],
            }
        )
    RESULT_STATE_PATH.unlink(missing_ok=True)
    append_log(f"Job started for account: {account['nickname']}")
    return True, "Job started."


def finish_job(status: str, error: str | None = None) -> None:
    JOB_STATE["running"] = False
    JOB_STATE["status"] = status
    JOB_STATE["finished_at"] = now_text()
    JOB_STATE["error"] = error


def generate_video_worker(account: dict, overrides: dict | None = None) -> None:
    youtube = None
    needs_flux = False
    try:
        overrides = overrides or {}
        output_mode = (overrides.get("output_mode") or "full_video").strip() or "full_video"
        content_mode = (overrides.get("content_mode") or "auto").strip() or "auto"
        needs_flux = output_mode in {"image_cards", "full_video"}
        JOB_STATE["content_mode"] = content_mode
        JOB_STATE["output_mode"] = output_mode
        set_stage("Preparing workspace", 5)
        MP_DIR.mkdir(exist_ok=True)

        set_stage("Launching Firefox", 10)
        youtube = YouTube(
            account["id"],
            account["nickname"],
            account["firefox_profile"],
            account["niche"],
            account["language"],
        )

        custom_subject = overrides.get("custom_subject", "").strip()
        custom_title = overrides.get("custom_title", "").strip()
        custom_description = overrides.get("custom_description", "").strip()
        custom_script = overrides.get("custom_script", "").strip()
        source_text = overrides.get("source_text", "").strip()

        if custom_script or source_text:
            set_stage("Preparing custom content", 20)
            if custom_script:
                youtube.subject = resolve_subject(custom_subject, short_text(custom_script, 40))
                youtube.script = normalize_script(custom_script)
                parsed = {"projects": []}
            else:
                youtube.subject, youtube.script, parsed = build_script_from_source(source_text, custom_subject, content_mode)

            if looks_garbled(youtube.subject):
                youtube.subject = fallback_subject(parsed, youtube.script)
            safe_title, safe_description = resolve_metadata(
                custom_title,
                custom_description,
                youtube.subject,
                youtube.script,
                parsed,
            )
            youtube.metadata = {
                "title": safe_title,
                "description": safe_description,
            }
            youtube.image_prompts = build_image_prompts(build_scene_plan(youtube.subject, youtube.metadata["title"], youtube.script, parsed))
            append_log("Using custom content instead of auto-generated topic/script.")
        else:
            set_stage("Generating topic", 20)
            youtube.generate_topic()

            set_stage("Generating script", 35)
            youtube.generate_script()

            set_stage("Generating title and description", 45)
            youtube.generate_metadata()

        JOB_STATE["subject"] = getattr(youtube, "subject", "") or ""
        JOB_STATE["title"] = getattr(youtube, "metadata", {}).get("title", "") or ""
        JOB_STATE["description"] = getattr(youtube, "metadata", {}).get("description", "") or ""
        JOB_STATE["script"] = getattr(youtube, "script", "") or ""

        if output_mode == "script_only":
            set_stage("Writing script output", 90)
            script_path = write_text_artifact("script", ".txt", getattr(youtube, "script", ""))
            metadata_path = write_text_artifact(
                "metadata",
                ".json",
                json.dumps(
                    {
                        "subject": getattr(youtube, "subject", ""),
                        "metadata": getattr(youtube, "metadata", {}),
                        "created_at": now_text(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            JOB_STATE["script_path"] = script_path
            save_result_state(
                {
                    "account_id": account["id"],
                    "account_name": account["nickname"],
                    "output_mode": output_mode,
                    "script_path": script_path,
                    "metadata_path": metadata_path,
                    "metadata": getattr(youtube, "metadata", {}),
                    "subject": getattr(youtube, "subject", ""),
                    "script": getattr(youtube, "script", ""),
                    "created_at": now_text(),
                }
            )
            set_stage("Completed", 100)
            append_log(f"Script exported: {script_path}")
            finish_job("completed")
            return

        if not getattr(youtube, "image_prompts", None):
            set_stage("Generating image prompts", 55)
            try:
                select_model(get_ollama_model())
            except Exception:
                append_log("Warning: failed to preselect Ollama model; continuing with existing provider state.")
            youtube.generate_prompts()

        set_stage("Generating speech", 78)
        tts = TTS()
        youtube.generate_script_to_speech(tts)

        if output_mode == "audio_subtitles":
            set_stage("Building subtitles", 90)
            subtitles_path = youtube.generate_subtitles(youtube.tts_path)
            audio_path = os.path.abspath(youtube.tts_path)
            subtitle_path = os.path.abspath(subtitles_path)
            JOB_STATE["audio_path"] = audio_path
            JOB_STATE["subtitle_path"] = subtitle_path
            _created = now_text()
            save_result_state(
                {
                    "account_id": account["id"],
                    "account_name": account["nickname"],
                    "output_mode": output_mode,
                    "audio_path": audio_path,
                    "subtitle_path": subtitle_path,
                    "metadata": getattr(youtube, "metadata", {}),
                    "subject": getattr(youtube, "subject", ""),
                    "script": getattr(youtube, "script", ""),
                    "created_at": _created,
                }
            )
            register_artifacts(getattr(youtube, "subject", ""), _created, audio=audio_path, subtitle=subtitle_path)
            set_stage("Completed", 100)
            append_log(f"Audio exported: {audio_path}")
            append_log(f"Subtitles exported: {subtitle_path}")
            finish_job("completed")
            return

        if needs_flux:
            set_stage("Starting image service", 55)
            ensure_flux_service_running()

        if output_mode == "image_cards":
            set_stage("Rendering image cards", 90)
            total_prompts = max(len(youtube.image_prompts), 1)
            image_paths = []
            youtube.images = []
            for idx, prompt in enumerate(youtube.image_prompts, start=1):
                pct = 60 + int(35 * idx / total_prompts)
                set_stage(f"Rendering image {idx}/{total_prompts}", pct)
                image_paths.append(os.path.abspath(youtube.generate_image(prompt)))
            JOB_STATE["image_paths"] = image_paths
            save_result_state(
                {
                    "account_id": account["id"],
                    "account_name": account["nickname"],
                    "output_mode": output_mode,
                    "image_paths": image_paths,
                    "metadata": getattr(youtube, "metadata", {}),
                    "subject": getattr(youtube, "subject", ""),
                    "script": getattr(youtube, "script", ""),
                    "created_at": now_text(),
                }
            )
            set_stage("Completed", 100)
            append_log(f"Image cards exported: {len(image_paths)} file(s)")
            finish_job("completed")
            return

        total_prompts = max(len(youtube.image_prompts), 1)
        for idx, prompt in enumerate(youtube.image_prompts, start=1):
            pct = 55 + int(20 * idx / total_prompts)
            set_stage(f"Rendering image {idx}/{total_prompts}", pct)
            youtube.generate_image(prompt)

        set_stage("Building subtitles", 86)
        subtitles_path = youtube.generate_subtitles(youtube.tts_path)
        audio_path = os.path.abspath(youtube.tts_path)
        subtitle_path = os.path.abspath(subtitles_path)
        JOB_STATE["audio_path"] = audio_path
        JOB_STATE["subtitle_path"] = subtitle_path

        set_stage("Combining video", 90)
        path = compose_video_with_subtitles(youtube, subtitles_path)
        abs_video_path = os.path.abspath(path)
        youtube.video_path = abs_video_path
        JOB_STATE["video_path"] = abs_video_path

        _created = now_text()
        save_result_state(
            {
                "account_id": account["id"],
                "account_name": account["nickname"],
                "output_mode": output_mode,
                "audio_path": audio_path,
                "subtitle_path": subtitle_path,
                "video_path": abs_video_path,
                "metadata": getattr(youtube, "metadata", {}),
                "subject": getattr(youtube, "subject", ""),
                "script": getattr(youtube, "script", ""),
                "created_at": _created,
            }
        )
        register_artifacts(getattr(youtube, "subject", ""), _created, video=abs_video_path, audio=audio_path, subtitle=subtitle_path)

        set_stage("Completed", 100)
        append_log(f"Video created: {abs_video_path}")
        finish_job("completed")
    except Exception as exc:
        append_log(f"Generation failed: {exc}")
        append_log(traceback.format_exc())
        finish_job("failed", str(exc))
    finally:
        if youtube is not None:
            try:
                youtube.browser.quit()
                append_log("Firefox closed.")
            except Exception:
                pass
        if needs_flux:
            stop_flux_service()


def upload_video_worker(account: dict) -> None:
    youtube = None
    try:
        result_state = load_result_state()
        if not result_state or result_state.get("account_id") != account["id"]:
            raise RuntimeError("No generated video found for this account yet.")

        video_path = result_state.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise RuntimeError("Generated video file is missing.")

        set_stage("Launching Firefox", 10)
        youtube = YouTube(
            account["id"],
            account["nickname"],
            account["firefox_profile"],
            account["niche"],
            account["language"],
        )

        youtube.video_path = video_path
        youtube.metadata = result_state.get("metadata", {}) or {"title": "MoneyPrinter video", "description": ""}

        set_stage("Uploading to YouTube", 55)
        success = youtube.upload_video()
        if not success:
            raise RuntimeError("YouTube upload failed.")

        JOB_STATE["uploaded_url"] = getattr(youtube, "uploaded_video_url", None)
        set_stage("Upload completed", 100)
        if JOB_STATE["uploaded_url"]:
            append_log(f"Uploaded URL: {JOB_STATE['uploaded_url']}")
        finish_job("completed")
    except Exception as exc:
        append_log(f"Upload failed: {exc}")
        append_log(traceback.format_exc())
        finish_job("failed", str(exc))
    finally:
        if youtube is not None:
            try:
                youtube.browser.quit()
            except Exception:
                pass


def launch_thread(target, account: dict) -> None:
    thread = threading.Thread(target=target, args=(account,), daemon=True)
    thread.start()


def launch_generation(account: dict, overrides: dict | None = None) -> None:
    thread = threading.Thread(target=generate_video_worker, args=(account, overrides), daemon=True)
    thread.start()


TEMPLATE = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MoneyPrinter GUI</title>
  <style>
    :root { --bg:#f4f7fb; --card:#ffffff; --ink:#132238; --muted:#5b6b80; --line:#dbe4ef; --accent:#0f6fff; --accent2:#0da37f; --danger:#cf3d4f; --warn:#d77b11; --soft:#eef4fb; --soft2:#f8fbff; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Segoe UI","Noto Sans TC",sans-serif; color:var(--ink); background:radial-gradient(circle at top left, rgba(15,111,255,0.12), transparent 32%), linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%); }
    .wrap { max-width:1260px; margin:0 auto; padding:24px; }
    h1 { margin:0 0 8px; font-size:34px; }
    .sub { color:var(--muted); margin-bottom:18px; max-width:900px; line-height:1.6; }
    .hero { background:linear-gradient(135deg, rgba(15,111,255,0.12), rgba(13,163,127,0.1)); border:1px solid #d4e4ff; border-radius:24px; padding:22px; margin-bottom:18px; box-shadow:0 12px 32px rgba(18,34,56,0.06); }
    .hero-grid { display:grid; grid-template-columns:1.3fr 0.7fr; gap:18px; align-items:start; }
    .steps { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:18px; }
    .step { background:#fff; border:1px solid #dbe4ef; border-radius:16px; padding:14px; }
    .step-num { display:inline-flex; width:28px; height:28px; border-radius:999px; align-items:center; justify-content:center; background:#0f6fff; color:#fff; font-size:13px; font-weight:800; margin-bottom:8px; }
    .step strong { display:block; margin-bottom:4px; }
    .hero-side { background:rgba(255,255,255,0.72); border:1px solid #dbe4ef; border-radius:18px; padding:16px; }
    .hint-list { margin:0; padding-left:18px; color:var(--muted); line-height:1.7; }
    .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .tag, .pill { display:inline-block; padding:6px 11px; border-radius:999px; font-size:12px; font-weight:700; }
    .tag { background:#edf3fb; color:#38506a; }
    .grid { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(340px,0.85fr); gap:18px; align-items:start; }
    .stack-grid { display:grid; gap:18px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 12px 32px rgba(18,34,56,0.06); }
    .section-title { margin:0 0 12px; font-size:20px; }
    .section-sub { margin:0 0 14px; color:var(--muted); line-height:1.55; }
    .msg { margin:0 0 14px; padding:12px 14px; border-radius:12px; background:#e8f0ff; color:#154190; border:1px solid #cddcff; }
    .warning { color:var(--warn); font-size:13px; line-height:1.5; }
    .meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
    .kv { background:var(--soft2); border:1px solid var(--line); border-radius:14px; padding:12px; }
    .pill { background:#e8eef8; color:var(--ink); }
    .running { background:#e7f5ee; color:#0b7a5c; } .failed { background:#fdebef; color:#a32638; } .completed { background:#e7efff; color:#1342a4; } .idle { background:#eef2f6; color:#586678; }
    .progress { width:100%; height:14px; background:#e7edf5; border-radius:999px; overflow:hidden; margin:12px 0 8px; }
    .progress > div { height:100%; background:linear-gradient(90deg,#0f6fff,#0da37f); }
    .settings-grid, .field-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .field-grid.wide { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .accounts { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
    label { display:block; font-size:13px; font-weight:700; margin:0 0 6px; color:#304255; }
    input, textarea, select { width:100%; padding:10px 12px; border-radius:12px; border:1px solid var(--line); margin-bottom:10px; font-size:14px; background:#fff; font-family:inherit; }
    textarea { min-height:140px; resize:vertical; }
    textarea.short { min-height:110px; }
    .account { border:1px solid var(--line); border-radius:16px; padding:14px; background:#fcfdff; }
    .account-head { display:flex; justify-content:space-between; gap:10px; margin-bottom:10px; align-items:center; }
    .account-name { font-size:16px; font-weight:800; }
    .muted { color:var(--muted); font-size:13px; line-height:1.6; }
    button { cursor:pointer; border:0; border-radius:12px; padding:11px 16px; font-weight:700; color:white; background:var(--accent); }
    button.secondary { background:var(--accent2); } button.danger { background:var(--danger); }
    .stack { display:flex; gap:8px; flex-wrap:wrap; }
    .actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; }
    .actions button { min-width:150px; }
    .details-group { display:grid; gap:12px; }
    details { border:1px solid var(--line); background:#fbfdff; border-radius:16px; padding:14px; }
    summary { cursor:pointer; font-weight:800; }
    summary span { color:var(--muted); font-weight:500; margin-left:6px; }
    pre { background:#0f1724; color:#dbe6f3; padding:14px; border-radius:14px; max-height:360px; overflow:auto; font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; margin:0; }
    .script-box { margin-bottom:12px; }
    .artifacts { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }
    .artifact { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fcfdff; }
    .artifact img, .artifact video { width:100%; display:block; background:#dfe7f2; aspect-ratio:9 / 16; object-fit:cover; }
    .artifact-body { padding:10px; font-size:13px; }
.quick-links { display:grid; gap:8px; }
.quick-link { background:var(--soft2); border:1px solid var(--line); border-radius:14px; padding:12px; }
.link-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.link-row a { font-size:13px; }
    .empty { padding:14px; background:#fffaf0; border:1px dashed #e7c788; border-radius:14px; color:#8b5d10; }
    @media (max-width:1000px) { .grid, .hero-grid, .steps, .settings-grid, .field-grid, .field-grid.wide, .meta { grid-template-columns:1fr; } .actions button { width:100%; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MoneyPrinter GUI</h1>
    <div class="sub">把文案貼進來後，先預覽腳本，再決定要輸出腳本、語音字幕、AI 生圖或完整影片。現在這版把操作流程集中在同一區，少一點跳來跳去。</div>
    {% if message %}<div class="msg">{{ message }}</div>{% endif %}
    <div class="hero">
      <div class="hero-grid">
        <div>
          <strong style="font-size:18px;">快速開始</strong>
          <div class="steps">
            <div class="step"><div class="step-num">1</div><strong>選帳號</strong><div class="muted">先選要用哪個 YouTube / Firefox profile。</div></div>
            <div class="step"><div class="step-num">2</div><strong>貼內容</strong><div class="muted">可直接貼完整原文，或貼已經整理好的旁白稿。</div></div>
            <div class="step"><div class="step-num">3</div><strong>選輸出</strong><div class="muted">先 Preview，再決定要出腳本、語音、圖片或完整影片。</div></div>
            <div class="step"><div class="step-num">4</div><strong>看結果</strong><div class="muted">右邊可直接看目前進度、最新檔案與影片。</div></div>
          </div>
        </div>
        <div class="hero-side">
          <strong>支援的內容型態</strong>
          <div class="pill-row">
            <span class="tag">GitHub 排行週報</span>
            <span class="tag">每日科技新聞</span>
            <span class="tag">Market report</span>
          </div>
          <ul class="hint-list">
            <li>先按「預覽腳本」確認內容方向對不對。</li>
            <li>如果只是要素材，先用「語音 + 字幕」或「AI 生圖」。</li>
            <li>完整影片通常最慢，建議最後再跑。</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="grid">
      <div class="stack-grid">
        <div class="card">
          <h2 class="section-title">1. 建立內容</h2>
          <div class="section-sub">主要操作都放在這裡。先選帳號與輸出模式，再貼文案，先 Preview 沒問題再 Generate。</div>
          <form method="post" action="/generate-custom" accept-charset="utf-8" id="content-form">
      <div class="field-grid">
        <div>
          <label>帳號</label>
          <select name="account_id" id="account-id" required>
            <option value="">請選擇帳號</option>
                  {% for account in accounts %}
                  <option value="{{ account.id }}" {% if selected_account_id == account.id %}selected{% endif %}>{{ account.nickname }} ｜ {{ account.niche }}</option>
                  {% endfor %}
                </select>
              </div>
        <div>
          <label>內容類型</label>
          <select name="content_mode" id="content-mode">
            {% for value, label in content_mode_labels.items() %}
            <option value="{{ value }}" {% if value == selected_content_mode %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>輸出模式</label>
          <select name="output_mode" id="output-mode">
            {% for value, label in output_mode_labels.items() %}
            <option value="{{ value }}" {% if value == selected_output_mode %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
                </select>
              </div>
            </div>
            <div class="field-grid wide">
              <div><label>主題</label><input id="custom-subject" name="custom_subject" placeholder="可留空，系統會自己整理" value="{{ form_subject }}"></div>
              <div><label>標題</label><input id="custom-title" name="custom_title" placeholder="可留空，系統會自動生成" value="{{ form_title }}"></div>
              <div><label>描述</label><input id="custom-description" name="custom_description" placeholder="可留空" value="{{ form_description }}"></div>
            </div>
            <label>已經整理好的旁白稿</label>
            <textarea id="custom-script" class="short" name="custom_script" placeholder="如果你已經有最終旁白，就貼這裡。留空的話會優先使用下面的原始內容自動整理。">{{ form_custom_script }}</textarea>
            <label>原始內容</label>
            <textarea id="source-text" name="source_text" placeholder="支援 GitHub 排行週報、每日科技新聞、market report 等長文內容。建議直接貼完整原文。">{{ form_source_text }}</textarea>
            <div id="preview-stale-note" class="warning" style="display:none; margin-top:10px;">內容已變更，請重新按一次「先預覽腳本」，避免看到上一輪的結果。</div>
            <div class="actions">
              <button class="secondary" formaction="/preview-custom" type="submit">先預覽腳本</button>
              <button type="submit">直接生成</button>
            </div>
            {% if draft_script %}
            <div class="script-box" style="margin-top:16px;">
              <strong>系統整理後的旁白稿</strong>
              <pre>{{ draft_script }}</pre>
            </div>
            <div class="script-box" style="margin-top:12px;">
              <strong>系統建議主題 / 標題 / 描述</strong>
              <pre>主題：{{ draft_subject or "-" }}
標題：{{ draft_title or "-" }}
描述：{{ draft_description or "-" }}</pre>
            </div>
            {% endif %}
          </form>
        </div>
        <div class="card">
          <h2 class="section-title">2. 目前旁白稿</h2>
          {% if draft_script %}
            <div class="meta">
              <div class="kv"><strong>主題</strong><br>{{ draft_subject or "-" }}</div>
              <div class="kv"><strong>標題</strong><br>{{ draft_title or "-" }}</div>
              <div class="kv"><strong>輸出模式</strong><br>{{ output_mode_labels.get(selected_output_mode, selected_output_mode or "-") }}</div>
              <div class="kv"><strong>更新時間</strong><br>{{ draft_created_at or "-" }}</div>
            </div>
            {% if draft_description %}
            <div class="script-box"><strong>描述</strong><div class="muted" style="margin-top:8px;">{{ draft_description }}</div></div>
            {% endif %}
            <div class="script-box"><strong>整理後旁白稿</strong><pre>{{ draft_script }}</pre></div>
          {% else %}
            <div class="empty">先按一次「先預覽腳本」，或直接開始生成，這裡就會顯示整理好的旁白稿。</div>
          {% endif %}
        </div>
        <div class="card">
          <h2 class="section-title">2. 腳本與分鏡預覽</h2>
          {% if preview.script %}
            <div class="meta">
              <div class="kv"><strong>帳號</strong><br>{{ preview.account_name }}</div>
              <div class="kv"><strong>建立時間</strong><br>{{ preview.created_at }}</div>
              <div class="kv"><strong>主題</strong><br>{{ preview.subject }}</div>
              <div class="kv"><strong>標題</strong><br>{{ preview.title }}</div>
            </div>
            <div class="script-box"><strong>描述</strong><div class="muted" style="margin-top:8px;">{{ preview.description }}</div></div>
            <div class="script-box"><strong>旁白稿</strong><pre>{{ preview.script }}</pre></div>
            <div class="accounts">
              <div class="account">
                <strong>段落切分</strong>
                <pre>{% for segment in preview.segments %}{{ loop.index }}. {{ segment }}
{% endfor %}</pre>
              </div>
              <div class="account">
                <strong>Scene plan</strong>
                <pre>{% for scene in preview.scene_plan %}{{ scene.index }}. {{ scene.scene_type }}
On-screen: {{ scene.on_screen_text }}
Visuals: {{ scene.visual_elements }}
Direction: {{ scene.image_direction }}

{% endfor %}</pre>
              </div>
            </div>
          {% else %}
            <div class="empty">還沒有 preview。先把原始內容貼上去，按一次「先預覽腳本」會比較安全。</div>
          {% endif %}
        </div>
        <div class="card">
          <h2 class="section-title">3. 帳號與設定</h2>
          <div class="details-group">
            <details open>
              <summary>帳號管理<span>編輯現有帳號、直接生成或上傳到 YouTube</span></summary>
              <div class="accounts" style="margin-top:14px;">
                {% for account in accounts %}
                <div class="account">
                  <div class="account-head">
                    <div>
                      <div class="account-name">{{ account.nickname }}</div>
                      <div class="muted">{{ account.niche }} ｜ {{ account.language }}</div>
                    </div>
                    <span class="tag">ID: {{ account.id[:8] }}</span>
                  </div>
                  <form method="post" action="/accounts/update/{{ account.id }}" accept-charset="utf-8">
                    <input name="nickname" value="{{ account.nickname }}" placeholder="Nickname" required>
                    <input name="niche" value="{{ account.niche }}" placeholder="Niche" required>
                    <input name="language" value="{{ account.language }}" placeholder="Language" required>
                    <input name="firefox_profile" value="{{ account.firefox_profile }}" placeholder="Firefox profile" required>
                    <div class="stack"><button class="secondary" type="submit">儲存帳號</button></div>
                  </form>
                  <div class="stack" style="margin-top:8px;">
                    <form method="post" action="/generate/{{ account.id }}"><button type="submit">直接生成</button></form>
                    <form method="post" action="/upload/{{ account.id }}"><button class="secondary" type="submit">上傳 YouTube</button></form>
                    <form method="post" action="/delete/{{ account.id }}"><button class="danger" type="submit">刪除</button></form>
                  </div>
                </div>
                {% endfor %}
              </div>
            </details>
            <details>
              <summary>新增帳號<span>第一次使用新 Firefox profile 時再開</span></summary>
              <form method="post" action="/accounts/add" accept-charset="utf-8" style="margin-top:14px;">
                <div class="field-grid">
                  <div><label>Nickname</label><input name="nickname" placeholder="Nickname" required></div>
                  <div><label>Niche</label><input name="niche" placeholder="AI tools" required></div>
                  <div><label>Language</label><input name="language" placeholder="Traditional Chinese" required></div>
                  <div><label>Firefox profile</label><input name="firefox_profile" placeholder="~/.mozilla/firefox/xxxx.moneyprinter" required></div>
                </div>
                <button class="secondary" type="submit">新增帳號</button>
              </form>
            </details>
            <details>
              <summary>進階設定<span>模型與腳本參數</span></summary>
              <form method="post" action="/settings/save" accept-charset="utf-8" style="margin-top:14px;">
                <div class="settings-grid">
                  <div><label>Ollama URL</label><input name="ollama_base_url" value="{{ config.ollama_base_url }}"></div>
                  <div><label>Ollama model</label><input name="ollama_model" value="{{ config.ollama_model }}"></div>
                  <div><label>Sentence count</label><input name="script_sentence_length" value="{{ config.script_sentence_length }}"></div>
                  <div><label>Threads</label><input name="threads" value="{{ config.threads }}"></div>
                </div>
                <button class="secondary" type="submit">儲存設定</button>
              </form>
            </details>
          </div>
        </div>
      </div>
      <div class="stack-grid">
        <div class="card">
          <h2 class="section-title">目前狀態</h2>
          <div class="meta">
            <div class="kv"><strong>狀態</strong><br><span id="job-status-pill" class="pill {{ job.status }}">{{ job.status }}</span></div>
            <div class="kv"><strong>帳號</strong><br><span id="job-account-name">{{ job.account_name or "-" }}</span></div>
            <div class="kv"><strong>階段</strong><br><span id="job-stage">{{ job.stage }}</span></div>
            <div class="kv"><strong>進度</strong><br><span id="job-progress-text">{{ job.progress }}%</span></div>
            <div class="kv"><strong>輸出</strong><br><span id="job-output-mode">{{ output_mode_labels.get(job.output_mode, job.output_mode or "-") }}</span></div>
            <div class="kv"><strong>開始</strong><br><span id="job-started-at">{{ job.started_at or "-" }}</span></div>
            <div class="kv"><strong>完成</strong><br><span id="job-finished-at">{{ job.finished_at or "-" }}</span></div>
            <div class="kv"><strong>結果</strong><br><span id="job-result-url">{{ job.uploaded_url or "-" }}</span></div>
          </div>
          <div class="progress"><div id="job-progress-bar" style="width: {{ job.progress }}%"></div></div>
          <div id="job-error-box" class="warning" {% if not job.error %}style="display:none;"{% endif %}><strong>錯誤：</strong><span id="job-error-text">{{ job.error }}</span></div>
          <details style="margin-top:12px;" open>
            <summary>執行記錄<span>有卡住時再看這裡</span></summary>
            <div style="margin-top:12px;"><pre id="job-log-text">{{ log_text }}</pre></div>
          </details>
        </div>
        <div class="card">
          <h2 class="section-title">最新輸出</h2>
          <div class="quick-links">
            {% if result.script_path %}
              <div class="quick-link"><strong>腳本</strong><br>{{ result.script_path_name }}<div class="link-row"><a href="/artifacts/{{ result.script_path_name }}">開啟</a><a href="/download/{{ result.script_path_name }}">下載</a></div></div>
            {% endif %}
            {% if result.audio_path %}
              <div class="quick-link"><strong>音訊</strong><br><audio controls src="/artifacts/{{ result.audio_path_name }}"></audio><div class="link-row"><a href="/artifacts/{{ result.audio_path_name }}">開啟</a><a href="/download/{{ result.audio_path_name }}">下載</a></div></div>
            {% endif %}
            {% if result.subtitle_path %}
              <div class="quick-link"><strong>字幕</strong><br>{{ result.subtitle_path_name }}<div class="link-row"><a href="/artifacts/{{ result.subtitle_path_name }}">開啟</a><a href="/download/{{ result.subtitle_path_name }}">下載</a></div></div>
            {% endif %}
            {% if result.video_path %}
              <div class="quick-link"><strong>影片</strong><br>{{ result.video_path_name }}<div class="link-row"><a href="/artifacts/{{ result.video_path_name }}">開啟</a><a href="/download/{{ result.video_path_name }}">下載</a></div></div>
            {% endif %}
            {% if result.metadata_path %}
              <div class="quick-link"><strong>Metadata</strong><br>{{ result.metadata_path_name }}<div class="link-row"><a href="/artifacts/{{ result.metadata_path_name }}">開啟</a><a href="/download/{{ result.metadata_path_name }}">下載</a></div></div>
            {% endif %}
          </div>
          {% if result.image_paths %}
            <div style="margin-top:12px;"><strong>最新圖片</strong></div>
            <div class="artifacts" style="margin-top:8px;">
              {% for image_name in result.image_path_names %}
              <div class="artifact">
                <img src="/artifacts/{{ image_name }}" alt="{{ image_name }}">
                <div class="artifact-body">
                  <div><strong>{{ image_name }}</strong></div>
                  <div class="link-row"><a href="/artifacts/{{ image_name }}">開啟</a><a href="/download/{{ image_name }}">下載</a></div>
                </div>
              </div>
              {% endfor %}
            </div>
          {% endif %}
          {% if not result.script_path and not result.audio_path and not result.subtitle_path and not result.image_paths and not result.video_path %}
            <div class="empty">目前還沒有輸出。先做一次 Preview 或 Generate 就會出現在這裡。</div>
          {% endif %}
        </div>
        <div class="card">
          <h2 class="section-title">最新影片預覽</h2>
          {% if latest_video %}
            <video controls src="/artifacts/{{ latest_video.name }}" style="width:100%;border-radius:14px;background:#111;"></video>
            <div class="link-row"><a href="/artifacts/{{ latest_video.name }}">開啟 {{ latest_video.name }}</a><a href="/download/{{ latest_video.name }}">下載影片</a></div>
          {% else %}
            <div class="empty">還沒有 mp4。若只是測流程，可以先跑「語音 + 字幕」。</div>
          {% endif %}
        </div>
        <div class="card">
          <h2 class="section-title">最近檔案</h2>
          <div class="artifacts">
            {% for artifact in artifacts %}
            <div class="artifact">
              {% if artifact.is_image %}<img src="/artifacts/{{ artifact.name }}" alt="{{ artifact.name }}">{% elif artifact.is_video %}<video controls src="/artifacts/{{ artifact.name }}"></video>{% endif %}
              <div class="artifact-body">
                <div><strong>{{ artifact.name }}</strong></div>
                <div>{{ artifact.mtime }}</div>
                <div>{{ artifact.size_kb }} KB</div>
                <div class="link-row"><a href="/artifacts/{{ artifact.name }}">開啟</a><a href="/download/{{ artifact.name }}">下載</a></div>
              </div>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>
    </div>
  </div>
<script>
  (function () {
    const outputModeLabels = {{ output_mode_labels | tojson }};
    const els = {
      form: document.getElementById("content-form"),
      accountId: document.getElementById("account-id"),
      contentMode: document.getElementById("content-mode"),
      outputModeSelect: document.getElementById("output-mode"),
      customSubject: document.getElementById("custom-subject"),
      customTitle: document.getElementById("custom-title"),
      customDescription: document.getElementById("custom-description"),
      customScript: document.getElementById("custom-script"),
      sourceText: document.getElementById("source-text"),
      staleNote: document.getElementById("preview-stale-note"),
      statusPill: document.getElementById("job-status-pill"),
      accountName: document.getElementById("job-account-name"),
      stage: document.getElementById("job-stage"),
      progressText: document.getElementById("job-progress-text"),
      outputMode: document.getElementById("job-output-mode"),
      startedAt: document.getElementById("job-started-at"),
      finishedAt: document.getElementById("job-finished-at"),
      resultUrl: document.getElementById("job-result-url"),
      progressBar: document.getElementById("job-progress-bar"),
      errorBox: document.getElementById("job-error-box"),
      errorText: document.getElementById("job-error-text"),
      logText: document.getElementById("job-log-text"),
    };

    function renderJob(job) {
      if (!job) return;
      const status = job.status || "idle";
      els.statusPill.className = `pill ${status}`;
      els.statusPill.textContent = status;
      els.accountName.textContent = job.account_name || "-";
      els.stage.textContent = job.stage || "-";
      els.progressText.textContent = `${job.progress || 0}%`;
      els.outputMode.textContent = outputModeLabels[job.output_mode] || job.output_mode || "-";
      els.startedAt.textContent = job.started_at || "-";
      els.finishedAt.textContent = job.finished_at || "-";
      els.resultUrl.textContent = job.uploaded_url || "-";
      els.progressBar.style.width = `${job.progress || 0}%`;
      if (job.error) {
        els.errorText.textContent = job.error;
        els.errorBox.style.display = "";
      } else {
        els.errorText.textContent = "";
        els.errorBox.style.display = "none";
      }
      els.logText.textContent = (job.logs && job.logs.length) ? job.logs.join("\\n") : "No logs yet.";
    }

    let pollTimer = null;
    let wasRunning = {{ "true" if job.running else "false" }};

    async function pollJobStatus() {
      try {
        const response = await fetch("/api/job-state", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        renderJob(payload.job);
        const isRunning = !!(payload.job && payload.job.running);
        if (!isRunning && wasRunning) {
          window.location.reload();
          return;
        }
        wasRunning = isRunning;
        if (!isRunning && pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
      } catch (error) {
        console.debug("job poll failed", error);
      }
    }

    if (wasRunning) {
      pollTimer = setInterval(pollJobStatus, 3000);
    }

    const previewInputs = [
      els.accountId,
      els.contentMode,
      els.outputModeSelect,
      els.customSubject,
      els.customTitle,
      els.customDescription,
      els.customScript,
      els.sourceText,
    ].filter(Boolean);

    function markPreviewStale() {
      if (!els.staleNote) return;
      els.staleNote.style.display = "";
    }

    for (const input of previewInputs) {
      input.addEventListener("input", markPreviewStale);
      input.addEventListener("change", markPreviewStale);
    }
  })();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    config_data = read_config()
    accounts = read_accounts()
    artifacts = list_artifacts()
    latest_video = next((artifact for artifact in artifacts if artifact["is_video"]), None)
    result_state = load_result_state() or {}
    job = dict(JOB_STATE)
    if job.get("video_path"):
        job["video_path_name"] = os.path.basename(job["video_path"])
    if job.get("audio_path"):
        job["audio_path_name"] = os.path.basename(job["audio_path"])
    if job.get("subtitle_path"):
        job["subtitle_path_name"] = os.path.basename(job["subtitle_path"])
    if job.get("script_path"):
        job["script_path_name"] = os.path.basename(job["script_path"])
    if job.get("image_paths"):
        job["image_path_names"] = [os.path.basename(path) for path in job["image_paths"]]
    for key in ("video_path", "audio_path", "subtitle_path", "script_path", "metadata_path"):
        value = result_state.get(key)
        if value:
            result_state[f"{key}_name"] = os.path.basename(value)
    if result_state.get("image_paths"):
        result_state["image_path_names"] = [os.path.basename(path) for path in result_state["image_paths"]]
    selected_account_id = (
        (job.get("account_id") if job.get("running") else None)
        or PREVIEW_STATE.get("account_id")
        or result_state.get("account_id")
        or ""
    )
    selected_content_mode = (
        (job.get("content_mode") if job.get("running") else None)
        or PREVIEW_STATE.get("content_mode")
        or result_state.get("content_mode")
        or "auto"
    )
    selected_output_mode = (
        (job.get("output_mode") if job.get("running") else None)
        or PREVIEW_STATE.get("output_mode")
        or result_state.get("output_mode")
        or "full_video"
    )
    draft_subject = (
        (job.get("subject") if job.get("running") else None)
        or PREVIEW_STATE.get("subject")
        or result_state.get("subject")
        or ""
    )
    draft_title = (
        (job.get("title") if job.get("running") else None)
        or PREVIEW_STATE.get("title")
        or result_state.get("metadata", {}).get("title")
        or ""
    )
    draft_description = (
        (job.get("description") if job.get("running") else None)
        or PREVIEW_STATE.get("description")
        or result_state.get("metadata", {}).get("description")
        or ""
    )
    draft_created_at = (
        (job.get("created_at") if job.get("running") else None)
        or PREVIEW_STATE.get("created_at")
        or result_state.get("created_at")
        or ""
    )
    form_subject = ""
    form_title = ""
    form_description = ""
    form_custom_script = (
        (job.get("custom_script") if job.get("running") else None)
        or PREVIEW_STATE.get("custom_script")
        or ""
    )
    form_source_text = (
        (job.get("source_text") if job.get("running") else None)
        or PREVIEW_STATE.get("source_text")
        or ""
    )
    if looks_garbled(form_custom_script):
        form_custom_script = ""
    if looks_garbled(form_source_text):
        form_source_text = ""
    draft_script = (
        (job.get("script") if job.get("running") else None)
        or PREVIEW_STATE.get("script")
        or result_state.get("script")
        or ""
    )
    return render_template_string(
        TEMPLATE,
        config=config_data,
        accounts=accounts,
        artifacts=artifacts,
        latest_video=latest_video,
        result=result_state,
        job=job,
        preview=PREVIEW_STATE,
        selected_account_id=selected_account_id,
        selected_content_mode=selected_content_mode,
        selected_output_mode=selected_output_mode,
        form_subject=form_subject,
        form_title=form_title,
        form_description=form_description,
        form_custom_script=form_custom_script,
        form_source_text=form_source_text,
        draft_subject=draft_subject,
        draft_title=draft_title,
        draft_description=draft_description,
        draft_created_at=draft_created_at,
        draft_script=draft_script,
        output_mode_labels={
            "script_only": "只出腳本",
            "audio_subtitles": "語音 + 字幕",
            "image_cards": "AI 生圖",
            "full_video": "完整影片",
        },
        content_mode_labels={
            "auto": "自動判斷",
            "github_weekly": "GitHub 週報",
            "daily_tech_news": "科技情報報告",
            "market_report": "Market report",
            "international_brief": "國際情勢報告",
        },
        log_text="\n".join(job["logs"]) if job["logs"] else "No logs yet.",
        message=request.args.get("message", ""),
    )


@app.get("/api/job-state")
def api_job_state():
    return jsonify({"job": snapshot_job_state()})


@app.post("/settings/save")
def save_settings():
    config_data = read_config()
    config_data["ollama_base_url"] = request.form.get("ollama_base_url", "").strip() or get_ollama_base_url()
    config_data["ollama_model"] = request.form.get("ollama_model", "").strip() or get_ollama_model()
    config_data["script_sentence_length"] = int(request.form.get("script_sentence_length", "4").strip() or "4")
    config_data["threads"] = int(request.form.get("threads", "2").strip() or "2")
    write_config(config_data)
    return redirect(url_for("index", message="Settings saved."))


@app.post("/accounts/add")
def create_account():
    nickname = request.form.get("nickname", "").strip()
    firefox_profile = request.form.get("firefox_profile", "").strip()
    niche = request.form.get("niche", "").strip()
    language = request.form.get("language", "").strip()
    if not all([nickname, firefox_profile, niche, language]):
        return redirect(url_for("index", message="All fields are required."))
    accounts = read_accounts()
    accounts.append({"id": str(uuid4()), "nickname": nickname, "firefox_profile": firefox_profile, "niche": niche, "language": language, "videos": []})
    write_accounts(accounts)
    return redirect(url_for("index", message=f"Account added: {nickname}"))


@app.post("/accounts/update/<account_id>")
def update_account(account_id: str):
    accounts = read_accounts()
    found = False
    for account in accounts:
        if account["id"] == account_id:
            account["nickname"] = request.form.get("nickname", "").strip()
            account["niche"] = request.form.get("niche", "").strip()
            account["language"] = request.form.get("language", "").strip()
            account["firefox_profile"] = request.form.get("firefox_profile", "").strip()
            found = True
            break
    if not found:
        return redirect(url_for("index", message="Account not found."))
    write_accounts(accounts)
    return redirect(url_for("index", message="Account updated."))


@app.post("/delete/<account_id>")
def delete_account(account_id: str):
    accounts = [account for account in read_accounts() if account["id"] != account_id]
    write_accounts(accounts)
    return redirect(url_for("index", message="Account deleted."))


@app.post("/generate/<account_id>")
def generate(account_id: str):
    account = find_account(account_id)
    if account is None:
        return redirect(url_for("index", message="Account not found."))
    ok, message = begin_job(account, "Generate video")
    if ok:
        launch_generation(account)
    return redirect(url_for("index", message=message))


@app.post("/generate-custom")
def generate_custom():
    account_id = request.form.get("account_id", "").strip()
    account = find_account(account_id)
    if account is None:
        return redirect(url_for("index", message="Account not found."))
    custom_script = request.form.get("custom_script", "").strip()
    source_text = request.form.get("source_text", "").strip()
    if not custom_script and not source_text and PREVIEW_STATE.get("account_id") == account_id:
        custom_script = PREVIEW_STATE.get("custom_script", "").strip()
        source_text = PREVIEW_STATE.get("source_text", "").strip()
        if not custom_script and PREVIEW_STATE.get("script"):
            custom_script = PREVIEW_STATE.get("script", "").strip()
    if not custom_script and not source_text:
        return redirect(url_for("index", message="Please provide either custom script or source material."))
    overrides = {
        "content_mode": request.form.get("content_mode", "auto").strip() or "auto",
        "output_mode": request.form.get("output_mode", "full_video").strip() or "full_video",
        "custom_subject": request.form.get("custom_subject", "").strip(),
        "custom_title": request.form.get("custom_title", "").strip(),
        "custom_description": request.form.get("custom_description", "").strip(),
        "custom_script": custom_script,
        "source_text": source_text,
    }
    ok, message = begin_job(account, "Generate video from custom text")
    if ok:
        launch_generation(account, overrides)
    return redirect(url_for("index", message=message))


@app.post("/preview-custom")
def preview_custom():
    account_id = request.form.get("account_id", "").strip()
    account = find_account(account_id)
    if account is None:
        return redirect(url_for("index", message="Account not found."))
    custom_script = request.form.get("custom_script", "").strip()
    source_text = request.form.get("source_text", "").strip()
    if not custom_script and not source_text:
        return redirect(url_for("index", message="Please provide either custom script or source material."))
    # --- debug: dump source for parser debugging ---
    try:
        _debug_path = MP_DIR / "_debug_last_source.txt"
        _debug_cm = request.form.get("content_mode", "auto").strip() or "auto"
        _debug_det = detect_content_type(source_text) if source_text else "empty"
        with open(_debug_path, "w", encoding="utf-8") as _df:
            _df.write(f"[content_mode={_debug_cm}] [detected={_debug_det}]\n")
            _df.write("=" * 60 + "\n")
            _df.write(source_text or "(no source text)")
    except Exception:
        pass
    # --- end debug ---
    preview = build_preview_payload(
        account=account,
        custom_subject=request.form.get("custom_subject", "").strip(),
        custom_title=request.form.get("custom_title", "").strip(),
        custom_description=request.form.get("custom_description", "").strip(),
        custom_script=custom_script,
        source_text=source_text,
        content_mode=request.form.get("content_mode", "auto").strip() or "auto",
        output_mode=request.form.get("output_mode", "full_video").strip() or "full_video",
    )
    PREVIEW_STATE.update(preview)
    return redirect(url_for("index", message="Preview generated. Review the script before making the video."))


@app.post("/upload/<account_id>")
def upload(account_id: str):
    account = find_account(account_id)
    if account is None:
        return redirect(url_for("index", message="Account not found."))
    ok, message = begin_job(account, "Upload to YouTube")
    if ok:
        launch_thread(upload_video_worker, account)
    return redirect(url_for("index", message=message))


@app.route("/artifacts/<path:name>")
def artifact(name: str):
    return send_from_directory(MP_DIR, os.path.basename(name))


@app.route("/download/<path:name>")
def download_artifact(name: str):
    basename = os.path.basename(name)
    friendly = get_friendly_download_name(basename)
    return send_from_directory(MP_DIR, basename, as_attachment=True, download_name=friendly)


@app.post("/api/cleanup")
def api_cleanup():
    keep = request.args.get("keep", 3, type=int)
    result = cleanup_old_artifacts(keep_latest=max(keep, 1))
    return jsonify(result)


def parse_international_brief_v2(source_text: str) -> dict:
    text = source_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    def clean_block_lines(block: str) -> list[str]:
        items: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            line = re.sub(r"^[\-\u2022\u2023\u25e6\u2043\u2219•●▪▸▶☐\s]+", "", line)
            if not line:
                continue
            if re.fullmatch(r"[=\-─━═\s]{8,}", line):
                continue
            items.append(clean_field_text(line))
        return items

    date_match = re.search(r"(20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)", text)
    report_date = clean_field_text(date_match.group(1)) if date_match else extract_report_date(source_text)

    header_pattern = re.compile(r"^【事件\s*([一二三四五六七八九十\d]+)】\s*(.+)$", re.M)
    headers = list(header_pattern.finditer(text))

    events: list[dict] = []
    for idx, header in enumerate(headers, start=1):
        start = header.end()
        end = headers[idx].start() if idx < len(headers) else len(text)
        body = text[start:end]
        title = clean_field_text(header.group(2))

        summary_match = re.search(
            r"摘要[:：]\s*(.*?)(?=\n\s*關鍵細節[:：]|\n\s*對台灣影響[:：]|\Z)",
            body,
            re.S,
        )
        details_match = re.search(
            r"關鍵細節[:：]\s*(.*?)(?=\n\s*對台灣影響[:：]|\Z)",
            body,
            re.S,
        )
        impact_match = re.search(r"對台灣影響[:：]\s*([^\n]*)(.*)", body, re.S)

        summary = clean_field_text(summary_match.group(1)) if summary_match else ""
        details = clean_block_lines(details_match.group(1))[:4] if details_match else []

        taiwan_impact = ""
        if impact_match:
            impact_header = clean_field_text(impact_match.group(1))
            impact_lines = clean_block_lines(impact_match.group(2))[:4]
            parts = []
            if impact_header:
                parts.append(impact_header)
            parts.extend(impact_lines)
            taiwan_impact = " ".join(part for part in parts if part)

        events.append(
            {
                "rank": idx,
                "title": title,
                "summary": summary,
                "details": details,
                "taiwan_impact": taiwan_impact,
            }
        )

    return {
        "content_type": "international_brief",
        "header": "國際情勢報告",
        "report_date": report_date,
        "summary": short_text(" ".join(event["title"] for event in events[:3]), 120),
        "events": events,
    }


def build_international_brief_script_v2(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    events = parsed.get("events", [])
    subject = resolve_subject(custom_subject, "國際情勢報告")
    report_date = parsed_report_date(parsed)

    if not events:
        if report_date:
            return subject, f"今天是{report_date}，國際線目前還沒有可整理的重點。"
        return subject, "今天國際線目前還沒有可整理的重點。"

    count_map = {
        1: "一",
        2: "兩",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    ordinal_map = {
        1: "第一",
        2: "第二",
        3: "第三",
        4: "第四",
        5: "第五",
        6: "第六",
        7: "第七",
        8: "第八",
        9: "第九",
        10: "第十",
    }

    visible_events = events[:5]
    count_label = count_map.get(len(visible_events), str(len(visible_events)))
    intro = f"今天是{report_date}，國際線先看{count_label}件事。" if report_date else f"今天國際線先看{count_label}件事。"
    sections = [intro]

    for idx, event in enumerate(visible_events, start=1):
        title = clean_field_text(event.get("title", "")) or f"第{idx}則國際新聞"
        summary = ensure_cn_punctuation(clean_field_text(event.get("summary", "")))
        details = [
            ensure_cn_punctuation(clean_field_text(item))
            for item in event.get("details", [])
            if clean_field_text(item)
        ]
        impact = ensure_cn_punctuation(clean_field_text(event.get("taiwan_impact", "")))

        lines = [f"{ordinal_map.get(idx, f'第{idx}')}件事，{title}。"]
        if summary:
            lines.append(summary)
        if details:
            lines.append(f"關鍵細節先看，{details[0].rstrip('。')}。")
        if len(details) > 1:
            lines.append(f"另外還有一點，{details[1].rstrip('。')}。")
        if impact:
            lines.append(f"對台灣比較直接的影響是，{impact.rstrip('。')}。")

        polished = [polish_daily_news_line(line) for line in lines if line.strip()]
        sections.append(" ".join(line for line in polished if line))

    return subject, "\n\n".join(section for section in sections if section.strip())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
