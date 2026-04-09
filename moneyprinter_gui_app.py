import json
import os
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from PIL import Image as PILImage

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_PATH = ROOT_DIR / "config.json"
MP_DIR = ROOT_DIR / ".mp"
RESULT_STATE_PATH = MP_DIR / "gui_last_result.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if not hasattr(PILImage, "ANTIALIAS") and hasattr(PILImage, "Resampling"):
    PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS

from cache import get_accounts, get_youtube_cache_path  # noqa: E402
from classes.Tts import TTS  # noqa: E402
from classes.YouTube import YouTube  # noqa: E402
from config import get_ollama_base_url, get_ollama_model  # noqa: E402
from utils import rem_temp_files  # noqa: E402

app = Flask(__name__)

SECTION_KEY_MAP = {
    "做什麼": "what",
    "為什麼爆紅": "why_hot",
    "適合誰": "who_for",
    "風險 / 門檻": "risks",
    "風險/門檻": "risks",
}

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
    "video_path": None,
    "uploaded_url": None,
    "error": None,
    "logs": [],
}
PREVIEW_STATE = {
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


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(message: str) -> None:
    JOB_STATE["logs"].append(f"[{now_text()}] {message}")
    JOB_STATE["logs"] = JOB_STATE["logs"][-300:]


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


def load_result_state() -> dict | None:
    if not RESULT_STATE_PATH.exists():
        return None
    with open(RESULT_STATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def list_artifacts() -> list[dict]:
    artifacts = []
    if not MP_DIR.exists():
        return artifacts

    for path in sorted(MP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name == RESULT_STATE_PATH.name:
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".wav", ".mp3", ".srt", ".mp4", ".json"}:
            continue
        stat = path.stat()
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


def normalize_script(text: str) -> str:
    cleaned = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    lines = [line.strip(" -•\t") for line in cleaned.split("\n")]
    lines = [line for line in lines if line]
    return " ".join(lines).strip()


def default_title_from_subject(subject: str) -> str:
    base = (subject or "AI short").strip()
    if len(base) > 82:
        base = base[:79].rstrip() + "..."
    if "#shorts" not in base.lower():
        base = f"{base} #shorts"
    return base


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


def safe_weekly_gain(stats: str) -> str:
    match = re.search(r"\+\s*([0-9,]+)\s*⭐", stats)
    if match:
        return f"+{match.group(1)} ⭐"
    return ""


def safe_total_stars(stats: str) -> str:
    match = re.search(r"📦\s*([0-9,]+)\s*⭐", stats)
    if match:
        return f"{match.group(1)} ⭐"
    return ""


def parse_ranked_projects(source_text: str) -> dict:
    lines = [line.rstrip() for line in source_text.replace("\ufeff", "").replace("\r\n", "\n").split("\n")]
    header_lines = []
    summary_lines = []
    projects = []
    current = None
    current_field = None

    def add_to_field(field_name: str, content: str) -> None:
        if current is None or not content:
            return
        current[field_name] = clean_field_text((current.get(field_name, "") + " " + content).strip())

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[=\-─_│\s]+", line):
            continue

        match = re.match(r"^【第\s*(\d+)\s*名】\s*(.+)$", line)
        if match:
            current = {
                "rank": int(match.group(1)),
                "name": match.group(2).strip(),
                "link": "",
                "stats": "",
                "what": "",
                "why_hot": "",
                "who_for": "",
                "risks": "",
            }
            projects.append(current)
            current_field = None
            continue

        if current is None:
            if "每週精選" in line or "熱門開源項目" in line:
                header_lines.append(line)
            elif len(line) >= 10:
                summary_lines.append(line)
            continue

        if line.startswith("🔗"):
            current["link"] = line.lstrip("🔗").strip()
            current_field = None
            continue
        if line.startswith("📦"):
            current["stats"] = line
            current_field = None
            continue

        section_match = re.match(r"^▸\s*([^：:]+)\s*[：:]\s*(.*)$", line)
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
        "header": " ".join(header_lines).strip(),
        "summary": " ".join(summary_lines[:3]).strip(),
        "projects": projects,
    }


def build_ranked_script(parsed: dict, custom_subject: str = "") -> tuple[str, str]:
    projects = parsed["projects"]
    header = parsed.get("header", "")
    summary = parsed.get("summary", "")

    if not projects:
        subject = custom_subject.strip() or "GitHub 熱門開源項目"
        script = normalize_script(summary or header or "本週開源圈持續聚焦 AI Agent、開發工具與可實際落地的自動化應用。")
        return subject, script

    top = projects[0]
    subject = custom_subject.strip() or f"GitHub 熱門開源項目週報：{top['name']}"

    beats = ["本週 GitHub 最熱，還是 AI Agent。"]

    top_three = projects[:3]
    for idx, project in enumerate(top_three, start=1):
        name = project["name"]
        what = short_text(project.get("what", ""), 18)

        beats.append(f"第{idx}名，{name}。")
        if idx == 1:
            beats.append(what or "它像 AI coding 的超大全。")
        elif idx == 2:
            beats.append(what or "它是長時程 Agent 框架。")
        else:
            beats.append(what or "主打離線也能用。")

    next_names = [project["name"] for project in projects[3:5]]
    if next_names:
        beats.append(f"另外，{ '、'.join(next_names) } 也在升溫。")

    closing_names = [project["name"] for project in top_three]
    beats.append("如果你只想先追三個。")
    beats.append("先看 " + "、".join(closing_names) + "。")

    beats = [short_text(beat, 36) for beat in beats if beat.strip()]
    script = "\n".join(beats)
    if len(script) > 420:
        trimmed = []
        total = 0
        for beat in beats:
            if total + len(beat) > 360:
                break
            trimmed.append(beat)
            total += len(beat)
        script = "\n".join(trimmed)

    return subject, script


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
    subject = custom_subject.strip() or short_text(intro, 42)
    script = normalize_script(" ".join([intro] + supporting))
    return subject, short_text(script, 620)


def build_script_from_source(source_text: str, custom_subject: str = "") -> tuple[str, str, dict]:
    parsed = parse_ranked_projects(source_text)
    if parsed["projects"]:
        subject, script = build_ranked_script(parsed, custom_subject)
    else:
        subject, script = build_generic_script(source_text, custom_subject)
    return subject, script, parsed


def split_script_segments(script: str, max_segments: int = 8) -> list[str]:
    chunks = re.split(r"[。！？!?]\s*", script)
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
            "on_screen_text": "GitHub ??????
?? Top 3",
            "primary_stat": parsed.get("header", "GitHub ??????????"),
            "secondary_stat": "?".join(project["name"] for project in top_three),
            "summary": short_text(parsed.get("summary", "?????? AI Agent?AI Coding ????????????"), 46),
            "reason": "??? 3 ????????????????",
            "tags": ["GitHub Weekly", "Top 3", "AI Agent"],
            "image_direction": "????????????????????????????????",
        }
    )

    for index, project in enumerate(projects[:3], start=2):
        total_stars = safe_total_stars(project.get("stats", ""))
        weekly_gain = safe_weekly_gain(project.get("stats", ""))
        short_voice = f"?{project['rank']}??{project['name']}?"
        what = short_text(project.get("what", ""), 24)
        if what:
            short_voice += f" {what}?"
        plans.append(
            {
                "index": index,
                "scene_type": f"Repo spotlight #{project['rank']}",
                "voiceover": short_voice,
                "on_screen_text": project["name"],
                "primary_stat": total_stars or "Stars",
                "secondary_stat": weekly_gain or "??????",
                "summary": compact_reason(project.get("what", ""), "???????????"),
                "reason": compact_reason(project.get("why_hot", ""), "??????????"),
                "tags": infer_project_tags(project),
                "image_direction": "??????????? debug ???????????????",
            }
        )

    follow_up = projects[3:6]
    if follow_up:
        plans.append(
            {
                "index": len(plans) + 1,
                "scene_type": "Fast movers",
                "voiceover": f"????? {'?'.join(project['name'] for project in follow_up)} ???????",
                "on_screen_text": "???????",
                "primary_stat": " / ".join(project["name"] for project in follow_up[:2]),
                "secondary_stat": follow_up[2]["name"] if len(follow_up) > 2 else "",
                "summary": "??????????????????",
                "reason": "?????????????????????",
                "tags": [tag for project in follow_up for tag in infer_project_tags(project)][:4],
                "image_direction": "????????????????????",
            }
        )

    plans.append(
        {
            "index": len(plans) + 1,
            "scene_type": "Closing CTA",
            "voiceover": f"?????????????? {'?'.join(project['name'] for project in projects[:3])} ????",
            "on_screen_text": "??? 3 ?",
            "primary_stat": "?".join(project["name"] for project in projects[:2]),
            "secondary_stat": projects[2]["name"] if len(projects) > 2 else "",
            "summary": "????????????????????",
            "reason": "?????? CTA??????????",
            "tags": ["Top 3", "??", "???"],
            "image_direction": "???? CTA ??????????????",
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
                "image_direction": "9:16 ?????????????????????????",
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
                "image_direction": "??????????????????",
            }
        )
    return plans


def build_scene_plan(subject: str, title: str, script: str, parsed: dict) -> list[dict]:
    if parsed.get("projects"):
        return build_scene_plan_from_ranked_projects(parsed, title, script)
    return build_scene_plan_generic(subject, title, split_script_segments(script))


def build_image_prompts(scene_plan: list[dict]) -> list[str]:
    prompts = []
    for scene in scene_plan:
        prompts.append(
            "Vertical 9:16 editorial infographic card for a short video. "
            f"Scene type: {scene['scene_type']}. "
            f"On-screen text: {scene['on_screen_text']}. "
            f"Primary stat: {scene.get('primary_stat', '')}. "
            f"Secondary stat: {scene.get('secondary_stat', '')}. "
            f"Summary: {scene.get('summary', '')}. "
            f"Reason: {scene.get('reason', '')}. "
            f"Tags: {' | '.join(scene.get('tags', []))}. "
            f"Direction: {scene['image_direction']} "
            "High legibility typography, clean layout, modern tech media style."
        )
    return prompts


def build_preview_payload(account: dict, custom_subject: str, custom_title: str, custom_description: str, custom_script: str, source_text: str) -> dict:
    parsed = {"projects": []}
    if custom_script:
        subject = custom_subject.strip() or short_text(custom_script, 40)
        script = normalize_script(custom_script)
    else:
        subject, script, parsed = build_script_from_source(source_text, custom_subject)

    title = custom_title.strip() or default_title_from_subject(subject)
    description = custom_description.strip() or script
    segments = split_script_segments(script)
    scene_plan = build_scene_plan(subject, title, script, parsed)

    return {
        "subject": subject,
        "title": title,
        "description": description,
        "script": script,
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
                "video_path": None,
                "uploaded_url": None,
                "error": None,
                "logs": [],
            }
        )
    append_log(f"Job started for account: {account['nickname']}")
    return True, "Job started."


def finish_job(status: str, error: str | None = None) -> None:
    JOB_STATE["running"] = False
    JOB_STATE["status"] = status
    JOB_STATE["finished_at"] = now_text()
    JOB_STATE["error"] = error


def generate_video_worker(account: dict, overrides: dict | None = None) -> None:
    youtube = None
    try:
        overrides = overrides or {}
        set_stage("Preparing workspace", 5)
        rem_temp_files()

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
                youtube.subject = custom_subject or short_text(custom_script, 40)
                youtube.script = normalize_script(custom_script)
                parsed = {"projects": []}
            else:
                youtube.subject, youtube.script, parsed = build_script_from_source(source_text, custom_subject)

            youtube.metadata = {
                "title": custom_title or default_title_from_subject(youtube.subject),
                "description": custom_description or youtube.script,
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

        if not getattr(youtube, "image_prompts", None):
            set_stage("Generating image prompts", 55)
            youtube.generate_prompts()

        total_prompts = max(len(youtube.image_prompts), 1)
        for idx, prompt in enumerate(youtube.image_prompts, start=1):
            pct = 55 + int(20 * idx / total_prompts)
            set_stage(f"Rendering image {idx}/{total_prompts}", pct)
            youtube.generate_image(prompt)

        set_stage("Generating speech", 78)
        tts = TTS()
        youtube.generate_script_to_speech(tts)

        set_stage("Combining video", 90)
        path = youtube.combine()
        abs_video_path = os.path.abspath(path)
        youtube.video_path = abs_video_path
        JOB_STATE["video_path"] = abs_video_path

        save_result_state(
            {
                "account_id": account["id"],
                "account_name": account["nickname"],
                "video_path": abs_video_path,
                "metadata": getattr(youtube, "metadata", {}),
                "subject": getattr(youtube, "subject", ""),
                "script": getattr(youtube, "script", ""),
                "created_at": now_text(),
            }
        )

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
    :root { --bg:#f4f7fb; --card:#ffffff; --ink:#132238; --muted:#5b6b80; --line:#dbe4ef; --accent:#0f6fff; --accent2:#0da37f; --danger:#cf3d4f; --warn:#d77b11; }
    body { margin:0; font-family:"Segoe UI","Noto Sans TC",sans-serif; color:var(--ink); background:radial-gradient(circle at top left, rgba(15,111,255,0.12), transparent 32%), linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%); }
    .wrap { max-width:1220px; margin:0 auto; padding:24px; }
    h1 { margin:0 0 8px; font-size:32px; }
    .sub { color:var(--muted); margin-bottom:20px; }
    .grid { display:grid; grid-template-columns:1.1fr 0.9fr; gap:18px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 12px 32px rgba(18,34,56,0.06); }
    .section-title { margin:0 0 12px; font-size:20px; }
    .msg { margin:0 0 14px; padding:12px 14px; border-radius:12px; background:#e8f0ff; color:#154190; border:1px solid #cddcff; }
    .warning { color:var(--warn); font-size:13px; }
    .meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
    .pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:12px; font-weight:700; background:#e8eef8; color:var(--ink); }
    .running { background:#e7f5ee; color:#0b7a5c; } .failed { background:#fdebef; color:#a32638; } .completed { background:#e7efff; color:#1342a4; } .idle { background:#eef2f6; color:#586678; }
    .progress { width:100%; height:14px; background:#e7edf5; border-radius:999px; overflow:hidden; margin:12px 0 8px; }
    .progress > div { height:100%; background:linear-gradient(90deg,#0f6fff,#0da37f); }
    .settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .accounts { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:12px; }
    textarea { width:100%; box-sizing:border-box; min-height:140px; padding:10px 12px; border-radius:12px; border:1px solid var(--line); margin-bottom:10px; font-size:14px; resize:vertical; font-family:inherit; }
    select { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:12px; border:1px solid var(--line); margin-bottom:10px; font-size:14px; background:#fff; }
    .account { border:1px solid var(--line); border-radius:14px; padding:14px; background:#fcfdff; }
    input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:12px; border:1px solid var(--line); margin-bottom:10px; font-size:14px; }
    button { cursor:pointer; border:0; border-radius:12px; padding:10px 14px; font-weight:700; color:white; background:var(--accent); }
    button.secondary { background:var(--accent2); } button.danger { background:var(--danger); }
    .stack { display:flex; gap:8px; flex-wrap:wrap; }
    pre { background:#0f1724; color:#dbe6f3; padding:14px; border-radius:14px; max-height:360px; overflow:auto; font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; }
    .artifacts { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
    .artifact { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fcfdff; }
    .artifact img, .artifact video { width:100%; display:block; background:#dfe7f2; }
    .artifact-body { padding:10px; font-size:13px; }
    @media (max-width:900px) { .grid, .settings-grid, .meta { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MoneyPrinter GUI</h1>
    <div class="sub">先把腳本與分鏡做好，再決定要不要產片。這版已停用自動刷新。</div>
    {% if message %}<div class="msg">{{ message }}</div>{% endif %}
    <div class="grid">
      <div class="card">
        <h2 class="section-title">Settings</h2>
        <form method="post" action="/settings/save" accept-charset="utf-8">
          <div class="settings-grid">
            <div><label>Ollama URL</label><input name="ollama_base_url" value="{{ config.ollama_base_url }}"></div>
            <div><label>Ollama model</label><input name="ollama_model" value="{{ config.ollama_model }}"></div>
            <div><label>Sentence count</label><input name="script_sentence_length" value="{{ config.script_sentence_length }}"></div>
            <div><label>Threads</label><input name="threads" value="{{ config.threads }}"></div>
          </div>
          <div class="stack"><button class="secondary" type="submit">Save settings</button></div>
        </form>
        <div style="height:18px"></div>
        <h2 class="section-title">Accounts</h2>
        <div class="accounts">
          {% for account in accounts %}
          <div class="account">
            <form method="post" action="/accounts/update/{{ account.id }}" accept-charset="utf-8">
              <input name="nickname" value="{{ account.nickname }}" placeholder="Nickname" required>
              <input name="niche" value="{{ account.niche }}" placeholder="Niche" required>
              <input name="language" value="{{ account.language }}" placeholder="Language" required>
              <input name="firefox_profile" value="{{ account.firefox_profile }}" placeholder="Firefox profile" required>
              <div class="stack"><button class="secondary" type="submit">Save</button></div>
            </form>
            <div style="height:8px"></div>
            <div class="stack">
              <form method="post" action="/generate/{{ account.id }}"><button type="submit">Generate</button></form>
              <form method="post" action="/upload/{{ account.id }}"><button class="secondary" type="submit">Upload to YouTube</button></form>
              <form method="post" action="/delete/{{ account.id }}"><button class="danger" type="submit">Delete</button></form>
            </div>
          </div>
          {% endfor %}
        </div>
        <div style="height:18px"></div>
        <h2 class="section-title">Add account</h2>
        <form method="post" action="/accounts/add" accept-charset="utf-8">
          <input name="nickname" placeholder="Nickname" required>
          <input name="firefox_profile" placeholder="~/.mozilla/firefox/xxxx.moneyprinter" required>
          <input name="niche" placeholder="AI tools" required>
          <input name="language" placeholder="Traditional Chinese" required>
          <button class="secondary" type="submit">Add account</button>
        </form>
        <div style="height:18px"></div>
        <h2 class="section-title">Generate from your own content</h2>
        <form method="post" action="/generate-custom" accept-charset="utf-8">
          <label>Account</label>
          <select name="account_id" required>
            <option value="">Choose an account</option>
            {% for account in accounts %}
            <option value="{{ account.id }}" {% if preview.account_id == account.id %}selected{% endif %}>{{ account.nickname }} ({{ account.niche }})</option>
            {% endfor %}
          </select>
          <input name="custom_subject" placeholder="Optional subject / topic" value="{{ preview.subject }}">
          <input name="custom_title" placeholder="Optional YouTube title" value="{{ preview.title }}">
          <input name="custom_description" placeholder="Optional description" value="{{ preview.description }}">
          <label>Custom script</label>
          <textarea name="custom_script" placeholder="Paste the final narration here if you already have a script."></textarea>
          <label>Source material</label>
          <textarea name="source_text" placeholder="Paste long-form source content here. If custom script is empty, the GUI will condense this into a short narration."></textarea>
          <div class="stack">
            <button class="secondary" formaction="/preview-custom" type="submit">Preview script first</button>
            <button type="submit">Generate from custom text</button>
          </div>
        </form>
      </div>
      <div class="card">
        <h2 class="section-title">Job status</h2>
        <div class="meta">
          <div><strong>Status</strong><br><span class="pill {{ job.status }}">{{ job.status }}</span></div>
          <div><strong>Account</strong><br>{{ job.account_name or "-" }}</div>
          <div><strong>Stage</strong><br>{{ job.stage }}</div>
          <div><strong>Progress</strong><br>{{ job.progress }}%</div>
          <div><strong>Started</strong><br>{{ job.started_at or "-" }}</div>
          <div><strong>Finished</strong><br>{{ job.finished_at or "-" }}</div>
        </div>
        <div class="progress"><div style="width: {{ job.progress }}%"></div></div>
        {% if job.video_path %}<div><strong>Latest video:</strong> <a href="/artifacts/{{ job.video_path_name }}">{{ job.video_path_name }}</a></div>{% endif %}
        {% if job.uploaded_url %}<div><strong>Uploaded URL:</strong> <a href="{{ job.uploaded_url }}" target="_blank">{{ job.uploaded_url }}</a></div>{% endif %}
        {% if job.error %}<div class="warning"><strong>Error:</strong> {{ job.error }}</div>{% endif %}
        <pre>{{ log_text }}</pre>
      </div>
    </div>
    <div style="height:18px"></div>
    <div class="card">
      <h2 class="section-title">Script preview</h2>
      {% if preview.script %}
        <div class="meta">
          <div><strong>Account</strong><br>{{ preview.account_name }}</div>
          <div><strong>Created</strong><br>{{ preview.created_at }}</div>
          <div><strong>Subject</strong><br>{{ preview.subject }}</div>
          <div><strong>Title</strong><br>{{ preview.title }}</div>
        </div>
        <div style="margin-bottom:10px;"><strong>Description</strong><br>{{ preview.description }}</div>
        <div style="margin-bottom:10px;"><strong>Voiceover script</strong><pre>{{ preview.script }}</pre></div>
        <div class="accounts">
          <div class="account">
            <strong>Scene segments</strong>
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
        <div class="warning">No preview yet. Paste your content above and click "Preview script first".</div>
      {% endif %}
    </div>
    <div style="height:18px"></div>
    <div class="card">
      <h2 class="section-title">Latest video preview</h2>
      {% if latest_video %}
        <video controls src="/artifacts/{{ latest_video.name }}" style="width:100%;max-width:520px;border-radius:14px;background:#111;"></video>
        <div style="margin-top:8px;"><a href="/artifacts/{{ latest_video.name }}">Open {{ latest_video.name }}</a></div>
      {% else %}
        <div class="warning">No mp4 has been generated yet.</div>
      {% endif %}
    </div>
    <div style="height:18px"></div>
    <div class="card">
      <h2 class="section-title">Latest artifacts</h2>
      <div class="artifacts">
        {% for artifact in artifacts %}
        <div class="artifact">
          {% if artifact.is_image %}<img src="/artifacts/{{ artifact.name }}" alt="{{ artifact.name }}">{% elif artifact.is_video %}<video controls src="/artifacts/{{ artifact.name }}"></video>{% endif %}
          <div class="artifact-body">
            <div><strong>{{ artifact.name }}</strong></div>
            <div>{{ artifact.mtime }}</div>
            <div>{{ artifact.size_kb }} KB</div>
            <div><a href="/artifacts/{{ artifact.name }}">Open</a></div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    config_data = read_config()
    accounts = read_accounts()
    artifacts = list_artifacts()
    latest_video = next((artifact for artifact in artifacts if artifact["is_video"]), None)
    job = dict(JOB_STATE)
    if job.get("video_path"):
        job["video_path_name"] = os.path.basename(job["video_path"])
    return render_template_string(
        TEMPLATE,
        config=config_data,
        accounts=accounts,
        artifacts=artifacts,
        latest_video=latest_video,
        job=job,
        preview=PREVIEW_STATE,
        log_text="\n".join(job["logs"]) if job["logs"] else "No logs yet.",
        message=request.args.get("message", ""),
    )


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
    if not custom_script and not source_text:
        return redirect(url_for("index", message="Please provide either custom script or source material."))
    overrides = {
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
    preview = build_preview_payload(
        account=account,
        custom_subject=request.form.get("custom_subject", "").strip(),
        custom_title=request.form.get("custom_title", "").strip(),
        custom_description=request.form.get("custom_description", "").strip(),
        custom_script=custom_script,
        source_text=source_text,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
