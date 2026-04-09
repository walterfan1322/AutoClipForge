import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import edge_tts
from gtts import gTTS

from config import ROOT_DIR

KITTEN_SAMPLE_RATE = 24000
DEFAULT_ZH_EDGE_VOICE = "zh-TW-HsiaoChenNeural"
DEFAULT_EN_EDGE_VOICE = "en-US-JennyNeural"
DEFAULT_ZH_ESPEAK = "cmn"
DEFAULT_EN_ESPEAK = "en"
DEFAULT_EDGE_RATE = "+0%"
DEFAULT_EDGE_PITCH = "+0Hz"
STRONG_PAUSE_SECONDS = 0.12
MEDIUM_PAUSE_SECONDS = 0.05
PLAYBACK_SPEED = 1.08
MAX_CHUNK_CHARS = 110
MIN_MEDIUM_SPLIT_CHARS = 40
MERGE_TARGET_CHARS = 125
MIN_STANDALONE_CHARS = 58
STRONG_PUNCTUATION = set("。！？!?；;")
MEDIUM_PUNCTUATION = set("，,、：:")


def _ffmpeg_convert(input_file: str, output_file: str) -> str:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ar",
            str(KITTEN_SAMPLE_RATE),
            output_file,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_file


def _trim_chunk_silence(input_wav: str, output_wav: str) -> str:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_wav,
            "-af",
            (
                "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,"
                "areverse,"
                "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,"
                "areverse"
            ),
            "-ar",
            str(KITTEN_SAMPLE_RATE),
            "-ac",
            "1",
            output_wav,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_wav


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _preprocess_zh_subtitle(text: str) -> str:
    text = str(text or "").replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    # Normalize slash-heavy date and list notation so Mandarin TTS does not
    # literally read out "斜線".
    text = re.sub(r"(\d{4})/(\d{1,2})/(\d{1,2})", r"\1年\2月\3日", text)
    text = re.sub(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", r"\1月\2日", text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z\u4e00-\u9fff%％])", "每", text)
    text = re.sub(r"(?<=[A-Za-z\u4e00-\u9fff])\s*/\s*(?=[A-Za-z\u4e00-\u9fff])", "和", text)
    return text.strip()


def _prepare_tts_text(text: str) -> str:
    text = _preprocess_zh_subtitle(text)
    paragraphs = []
    for raw in text.split("\n\n"):
        paragraph = raw.strip()
        if not paragraph:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[。！？])", paragraph) if s.strip()]
        if len(sentences) <= 1:
            paragraphs.append(paragraph)
            continue

        rebuilt = []
        for idx, sentence in enumerate(sentences):
            cleaned = sentence.strip()
            if idx < len(sentences) - 1:
                cleaned = cleaned.rstrip("。！？")
                rebuilt.append(cleaned + "，")
            else:
                rebuilt.append(cleaned)
        paragraphs.append(" ".join(part for part in rebuilt if part).strip())
    return "\n\n".join(paragraphs)


def _chunk_pause(chunk: str) -> float:
    chunk = chunk.strip()
    if not chunk:
        return STRONG_PAUSE_SECONDS
    return MEDIUM_PAUSE_SECONDS if chunk[-1] in MEDIUM_PUNCTUATION else STRONG_PAUSE_SECONDS


def _append_chunk(pieces: list[tuple[str, float]], chunk: str) -> None:
    chunk = chunk.strip()
    if chunk:
        pieces.append((chunk, _chunk_pause(chunk)))


def _split_long_chunk(chunk: str) -> list[str]:
    chunk = chunk.strip()
    if len(chunk) <= MAX_CHUNK_CHARS:
        return [chunk] if chunk else []

    parts = []
    current = chunk
    split_candidates = "，,、：:；;。！？!? "
    while len(current) > MAX_CHUNK_CHARS:
        window = current[:MAX_CHUNK_CHARS]
        split_at = max(window.rfind(sep) for sep in split_candidates)
        if split_at <= 0:
            split_at = MAX_CHUNK_CHARS
        parts.append(current[:split_at + 1].strip())
        current = current[split_at + 1 :].strip()
    if current:
        parts.append(current)
    return [part for part in parts if part]


def _split_for_speech(text: str) -> list[tuple[str, float]]:
    paragraphs = []
    current_paragraph = []
    for raw in text.replace("\r", "").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph).strip())
                current_paragraph = []
            continue
        current_paragraph.append(line)
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph).strip())

    grouped = []
    for paragraph in paragraphs:
        pieces = []
        line = paragraph
        if not line:
            continue

        current = ""
        for char in line:
            current += char
            current_stripped = current.strip()
            if not current_stripped:
                continue

            should_split = False
            if char in STRONG_PUNCTUATION:
                should_split = True
            elif char in MEDIUM_PUNCTUATION and len(current_stripped) >= MIN_MEDIUM_SPLIT_CHARS:
                should_split = True
            elif len(current_stripped) >= MAX_CHUNK_CHARS:
                should_split = True

            if should_split:
                for part in _split_long_chunk(current):
                    _append_chunk(pieces, part)
                current = ""

        if current.strip():
            for part in _split_long_chunk(current):
                _append_chunk(pieces, part)

        merged = []
        for chunk, pause in pieces:
            if merged and len(chunk) <= 6 and len(merged[-1][0]) + len(chunk) <= MAX_CHUNK_CHARS:
                prev_chunk, _ = merged[-1]
                merged[-1] = (f"{prev_chunk}{chunk}", pause)
            else:
                merged.append((chunk, pause))

        if not merged:
            continue

        buffer_text = ""
        buffer_pause = STRONG_PAUSE_SECONDS
        buffer_count = 0

        def flush_buffer() -> None:
            nonlocal buffer_text, buffer_pause, buffer_count
            if buffer_text:
                grouped.append((buffer_text, buffer_pause))
            buffer_text = ""
            buffer_pause = STRONG_PAUSE_SECONDS
            buffer_count = 0

        for chunk, pause in merged:
            if not buffer_text:
                buffer_text = chunk
                buffer_pause = pause
                buffer_count = 1
                continue

            prospective = f"{buffer_text}{chunk}"
            should_merge = (
                len(buffer_text) < MIN_STANDALONE_CHARS
                or (buffer_count < 2 and len(prospective) <= MERGE_TARGET_CHARS)
            )

            if should_merge:
                buffer_text = prospective
                buffer_pause = pause
                buffer_count += 1
                continue

            flush_buffer()
            buffer_text = chunk
            buffer_pause = pause
            buffer_count = 1

        flush_buffer()
    return grouped


async def _edge_save(text: str, mp3_path: str, voice: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=DEFAULT_EDGE_RATE,
        pitch=DEFAULT_EDGE_PITCH,
    )
    await communicate.save(mp3_path)


def _synthesize_edge_chunk(text: str, voice: str, output_wav: str) -> str:
    fd, temp_audio = tempfile.mkstemp(prefix="moneyprinter-edge-", suffix=".mp3")
    os.close(fd)
    try:
        asyncio.run(_edge_save(text, temp_audio, voice))
        converted = _ffmpeg_convert(temp_audio, output_wav)
        trimmed = str(Path(output_wav).with_name(Path(output_wav).stem + "-trim.wav"))
        return _trim_chunk_silence(converted, trimmed)
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def _synthesize_gtts_chunk(text: str, lang: str, output_wav: str) -> str:
    fd, temp_audio = tempfile.mkstemp(prefix="moneyprinter-gtts-", suffix=".mp3")
    os.close(fd)
    try:
        gTTS(text=text, lang=lang).save(temp_audio)
        converted = _ffmpeg_convert(temp_audio, output_wav)
        trimmed = str(Path(output_wav).with_name(Path(output_wav).stem + "-trim.wav"))
        return _trim_chunk_silence(converted, trimmed)
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def _synthesize_espeak_chunk(text: str, voice: str, output_wav: str) -> str:
    fd, temp_wav = tempfile.mkstemp(prefix="moneyprinter-espeak-", suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "espeak-ng",
                "-v",
                voice,
                "-s",
                "155",
                "-w",
                temp_wav,
                text,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        converted = _ffmpeg_convert(temp_wav, output_wav)
        trimmed = str(Path(output_wav).with_name(Path(output_wav).stem + "-trim.wav"))
        return _trim_chunk_silence(converted, trimmed)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def _create_silence(duration: float, output_wav: str) -> str:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={KITTEN_SAMPLE_RATE}:cl=mono",
            "-t",
            f"{duration:.2f}",
            output_wav,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_wav


def _concat_wavs(parts: list[str], output_wav: str) -> str:
    if not parts:
        raise ValueError("No WAV parts were provided for concatenation.")
    if len(parts) == 1:
        shutil.copyfile(parts[0], output_wav)
        return output_wav

    list_file = Path(output_wav).with_suffix(".concat.txt")
    list_content = "".join(f"file '{Path(part).as_posix()}'\n" for part in parts)
    list_file.write_text(list_content, encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-ar",
                str(KITTEN_SAMPLE_RATE),
                "-ac",
                "1",
                output_wav,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_wav
    finally:
        list_file.unlink(missing_ok=True)


def _apply_playback_speed(input_wav: str, output_wav: str, speed: float) -> str:
    if abs(speed - 1.0) < 0.01:
        return input_wav
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_wav,
            "-filter:a",
            f"atempo={speed}",
            "-ar",
            str(KITTEN_SAMPLE_RATE),
            "-ac",
            "1",
            output_wav,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_wav


class TTS:
    def __init__(self) -> None:
        self._voice = None
        self.chunk_durations: list[float] = []

    def synthesize(self, text, output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav")):
        if not text or not text.strip():
            raise ValueError("Text for speech synthesis is empty.")

        use_zh = _has_non_ascii(text)
        if use_zh:
            text = _prepare_tts_text(text)
        edge_voice = self._voice or (DEFAULT_ZH_EDGE_VOICE if use_zh else DEFAULT_EN_EDGE_VOICE)
        espeak_voice = DEFAULT_ZH_ESPEAK if use_zh else DEFAULT_EN_ESPEAK
        chunks = _split_for_speech(text)
        if not chunks:
            chunks = [(text.strip(), STRONG_PAUSE_SECONDS)]

        self.chunk_durations = []
        temp_dir = Path(tempfile.mkdtemp(prefix="moneyprinter-tts-"))
        try:
            parts = []
            lang = "zh-TW" if use_zh else "en"
            for idx, (chunk, pause_seconds) in enumerate(chunks, start=1):
                speech_path = str(temp_dir / f"speech-{idx:03d}.wav")
                try:
                    _synthesize_edge_chunk(chunk, edge_voice, speech_path)
                except Exception:
                    try:
                        _synthesize_gtts_chunk(chunk, lang, speech_path)
                    except Exception:
                        _synthesize_espeak_chunk(chunk, espeak_voice, speech_path)

                # Record actual WAV duration of this speech chunk
                try:
                    with wave.open(speech_path, "r") as wf:
                        self.chunk_durations.append(wf.getnframes() / wf.getframerate())
                except Exception:
                    self.chunk_durations.append(-1.0)

                parts.append(speech_path)

                if idx != len(chunks):
                    silence_path = str(temp_dir / f"pause-{idx:03d}.wav")
                    _create_silence(pause_seconds, silence_path)
                    parts.append(silence_path)

            merged_path = str(temp_dir / "merged.wav")
            _concat_wavs(parts, merged_path)
            result = _apply_playback_speed(merged_path, output_file, PLAYBACK_SPEED)

            # Save chunk-level timing metadata alongside the audio file so
            # the subtitle generator can use actual speech durations instead
            # of character-count estimation.  We include the chunk texts so
            # that the subtitle generator does not need to re-split and can
            # guarantee a 1:1 mapping with the recorded durations.
            try:
                meta = {
                    "chunk_durations": self.chunk_durations,
                    "chunk_pauses": [p for _, p in chunks],
                    "chunk_texts": [t for t, _ in chunks],
                    "playback_speed": PLAYBACK_SPEED,
                }
                with open(output_file + ".timing.json", "w") as f:
                    json.dump(meta, f, ensure_ascii=False)
            except Exception:
                pass

            return result
        finally:
            for item in temp_dir.glob("*"):
                item.unlink(missing_ok=True)
            temp_dir.rmdir()
