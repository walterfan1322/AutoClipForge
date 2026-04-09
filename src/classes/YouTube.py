import re
import base64
import json
import time
import os
import atexit
import shutil
import tempfile
import textwrap
import requests
import assemblyai as aai

from utils import *
from cache import *
from .Tts import TTS, PLAYBACK_SPEED, _prepare_tts_text, _split_for_speech
from llm_provider import generate_text
from config import *
from status import *
from uuid import uuid4
from constants import *
from typing import List
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored
from selenium_firefox import *
from selenium import webdriver
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from moviepy.video.tools.subtitles import SubtitlesClip
from webdriver_manager.firefox import GeckoDriverManager
from datetime import datetime

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})


class YouTube:
    """
    Class for YouTube Automation.

    Steps to create a YouTube Short:
    1. Generate a topic [DONE]
    2. Generate a script [DONE]
    3. Generate metadata (Title, Description, Tags) [DONE]
    4. Generate AI Image Prompts [DONE]
    4. Generate Images based on generated Prompts [DONE]
    5. Convert Text-to-Speech [DONE]
    6. Show images each for n seconds, n: Duration of TTS / Amount of images [DONE]
    7. Combine Concatenated Images with the Text-to-Speech [DONE]
    """

    @staticmethod
    def _prepare_runtime_profile(profile_path: str) -> str:
        runtime_root = tempfile.mkdtemp(prefix="moneyprinter-firefox-")
        runtime_profile_path = os.path.join(runtime_root, "profile")
        shutil.copytree(
            profile_path,
            runtime_profile_path,
            ignore=shutil.ignore_patterns("lock", ".parentlock", "parent.lock"),
            ignore_dangling_symlinks=True,
        )

        for lock_name in ("lock", ".parentlock", "parent.lock"):
            lock_path = os.path.join(runtime_profile_path, lock_name)
            try:
                if os.path.lexists(lock_path):
                    os.unlink(lock_path)
            except OSError:
                pass

        atexit.register(lambda: shutil.rmtree(runtime_root, ignore_errors=True))
        return runtime_profile_path

    def __init__(
        self,
        account_uuid: str,
        account_nickname: str,
        fp_profile_path: str,
        niche: str,
        language: str,
    ) -> None:
        """
        Constructor for YouTube Class.

        Args:
            account_uuid (str): The unique identifier for the YouTube account.
            account_nickname (str): The nickname for the YouTube account.
            fp_profile_path (str): Path to the firefox profile that is logged into the specificed YouTube Account.
            niche (str): The niche of the provided YouTube Channel.
            language (str): The language of the Automation.

        Returns:
            None
        """
        self._account_uuid: str = account_uuid
        self._account_nickname: str = account_nickname
        self._fp_profile_path: str = fp_profile_path
        self._niche: str = niche
        self._language: str = language

        self.images = []

        # Initialize the Firefox profile
        self.options: Options = Options()

        # Set headless state of browser
        if get_headless():
            self.options.add_argument("--headless")

        if not os.path.isdir(self._fp_profile_path):
            raise ValueError(
                f"Firefox profile path does not exist or is not a directory: {self._fp_profile_path}"
            )

        self._runtime_profile_path = self._prepare_runtime_profile(self._fp_profile_path)

        self.options.add_argument("-profile")
        self.options.add_argument(self._runtime_profile_path)
        self.options.add_argument("-no-remote")

        # Set the service
        self.service: Service = Service(GeckoDriverManager().install())

        # Initialize the browser
        self.browser: webdriver.Firefox = webdriver.Firefox(
            service=self.service, options=self.options
        )

    @property
    def niche(self) -> str:
        """
        Getter Method for the niche.

        Returns:
            niche (str): The niche
        """
        return self._niche

    @property
    def language(self) -> str:
        """
        Getter Method for the language to use.

        Returns:
            language (str): The language
        """
        return self._language

    def generate_response(self, prompt: str, model_name: str = None) -> str:
        """
        Generates an LLM Response based on a prompt and the user-provided model.

        Args:
            prompt (str): The prompt to use in the text generation.

        Returns:
            response (str): The generated AI Repsonse.
        """
        return generate_text(prompt, model_name=model_name)

    def generate_topic(self) -> str:
        """
        Generates a topic based on the YouTube Channel niche.

        Returns:
            topic (str): The generated topic.
        """
        completion = self.generate_response(
            f"Please generate a specific video idea that takes about the following topic: {self.niche}. Make it exactly one sentence. Only return the topic, nothing else."
        )

        if not completion:
            error("Failed to generate Topic.")

        self.subject = completion

        return completion

    def generate_script(self) -> str:
        """
        Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.

        Returns:
            script (str): The script of the video.
        """
        sentence_length = get_script_sentence_length()
        prompt = f"""
        Generate a script for a video in {sentence_length} sentences, depending on the subject of the video.

        The script is to be returned as a string with the specified number of paragraphs.

        Here is an example of a string:
        "This is an example string."

        Do not under any circumstance reference this prompt in your response.

        Get straight to the point, don't start with unnecessary things like, "welcome to this video".

        Obviously, the script should be related to the subject of the video.
        
        YOU MUST NOT EXCEED THE {sentence_length} SENTENCES LIMIT. MAKE SURE THE {sentence_length} SENTENCES ARE SHORT.
        YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
        YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
        ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT
        
        Subject: {self.subject}
        Language: {self.language}
        """
        completion = self.generate_response(prompt)

        # Apply regex to remove *
        completion = re.sub(r"\*", "", completion)

        if not completion:
            error("The generated script is empty.")
            return

        if len(completion) > 5000:
            if get_verbose():
                warning("Generated Script is too long. Retrying...")
            return self.generate_script()

        self.script = completion

        return completion

    def generate_metadata(self) -> dict:
        """
        Generates Video metadata for the to-be-uploaded YouTube Short (Title, Description).

        Returns:
            metadata (dict): The generated metadata.
        """
        title = self.generate_response(
            f"Please generate a YouTube Video Title for the following subject, including hashtags: {self.subject}. Only return the title, nothing else. Limit the title under 100 characters."
        )

        if len(title) > 100:
            if get_verbose():
                warning("Generated Title is too long. Retrying...")
            return self.generate_metadata()

        description = self.generate_response(
            f"Please generate a YouTube Video Description for the following script: {self.script}. Only return the description, nothing else."
        )

        self.metadata = {"title": title, "description": description}

        return self.metadata

    def generate_prompts(self) -> List[str]:
        """
        Generates AI Image Prompts based on the provided Video Script.

        Returns:
            image_prompts (List[str]): Generated List of image prompts.
        """
        n_prompts = len(self.script) / 3

        prompt = f"""
        Generate {n_prompts} Image Prompts for AI Image Generation,
        depending on the subject of a video.
        Subject: {self.subject}

        The image prompts are to be returned as
        a JSON-Array of strings.

        Each search term should consist of a full sentence,
        always add the main subject of the video.

        Be emotional and use interesting adjectives to make the
        Image Prompt as detailed as possible.

        YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
        YOU MUST NOT RETURN ANYTHING ELSE.
        YOU MUST NOT RETURN THE SCRIPT.

        The search terms must be related to the subject of the video.
        Here is an example of a JSON-Array of strings:
        ["image prompt 1", "image prompt 2", "image prompt 3"]

        For context, here is the full text:
        {self.script}
        """

        completion = (
            str(self.generate_response(prompt))
            .replace("```json", "")
            .replace("```", "")
        )

        image_prompts = []

        if "image_prompts" in completion:
            image_prompts = json.loads(completion)["image_prompts"]
        else:
            try:
                image_prompts = json.loads(completion)
                if get_verbose():
                    info(f" => Generated Image Prompts: {image_prompts}")
            except Exception:
                if get_verbose():
                    warning(
                        "LLM returned an unformatted response. Attempting to clean..."
                    )

                # Get everything between [ and ], and turn it into a list
                r = re.compile(r"\[.*\]")
                image_prompts = r.findall(completion)
                if len(image_prompts) == 0:
                    if get_verbose():
                        warning("Failed to generate Image Prompts. Retrying...")
                    return self.generate_prompts()

        if len(image_prompts) > n_prompts:
            image_prompts = image_prompts[: int(n_prompts)]

        self.image_prompts = image_prompts

        success(f"Generated {len(image_prompts)} Image Prompts.")

        return image_prompts

    def _persist_image(self, image_bytes: bytes, provider_label: str) -> str:
        """
        Writes generated image bytes to a PNG file in .mp.

        Args:
            image_bytes (bytes): Image payload
            provider_label (str): Label for logging

        Returns:
            path (str): Absolute image path
        """
        image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")

        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        if get_verbose():
            info(f' => Wrote image from {provider_label} to "{image_path}"')

        self.images.append(image_path)
        return image_path

    def generate_image_nanobanana2(self, prompt: str) -> str:
        """
        Generates an AI Image using Nano Banana 2 API (Gemini image API).

        Args:
            prompt (str): Prompt for image generation

        Returns:
            path (str): The path to the generated image.
        """
        print(f"Generating Image using Nano Banana 2 API: {prompt}")

        api_key = get_nanobanana2_api_key()
        if not api_key:
            warning("nanobanana2_api_key is not configured. Falling back to placeholder image.")
            return self.generate_placeholder_image(prompt)

        base_url = get_nanobanana2_api_base_url().rstrip("/")
        model = get_nanobanana2_model()
        aspect_ratio = get_nanobanana2_aspect_ratio()

        endpoint = f"{base_url}/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio},
            },
        }

        try:
            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            body = response.json()

            candidates = body.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if not inline_data:
                        continue
                    data = inline_data.get("data")
                    mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
                    if data and str(mime_type).startswith("image/"):
                        image_bytes = base64.b64decode(data)
                        return self._persist_image(image_bytes, "Nano Banana 2 API")

            if get_verbose():
                warning(f"Nano Banana 2 did not return an image payload. Response: {body}")
            return None
        except Exception as e:
            if get_verbose():
                warning(f"Failed to generate image with Nano Banana 2 API: {str(e)}")
            return self.generate_placeholder_image(prompt)

    def generate_image_flux(self, prompt: str) -> str | None:
        """
        Generates an AI image using the FLUX image service, if configured.

        Args:
            prompt (str): Prompt for image generation

        Returns:
            path (str | None): Path to the generated image, or None if FLUX is disabled.
        """
        base_url = get_flux_api_base_url().rstrip("/")
        if not base_url:
            return None

        if get_verbose():
            info(f" => Generating image via FLUX service: {base_url}")

        payload = {
            "prompt": prompt,
            "width": 720,
            "height": 1280,
            "num_inference_steps": 10,
            "guidance_scale": 3.5,
            "max_sequence_length": 256,
        }

        try:
            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                timeout=1800,
            )
            response.raise_for_status()
            body = response.json()
            data = body.get("image_base64", "")
            if not data:
                raise ValueError("FLUX service returned no image payload")
            image_bytes = base64.b64decode(data)
            return self._persist_image(image_bytes, "FLUX service")
        except Exception as exc:
            if get_verbose():
                warning(f"Failed to generate image with FLUX service: {exc}")
            return None

    def generate_placeholder_image(self, prompt: str) -> str:
        """
        Generates a simple placeholder image when no image API is configured.

        Args:
            prompt (str): Prompt text to render

        Returns:
            path (str): The path to the generated image.
        """
        parsed = self._parse_visual_prompt(prompt)
        accent = self._placeholder_accent(parsed["scene_type"])
        image = self._build_placeholder_template(parsed, accent)

        image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")
        image.save(image_path, format="PNG")
        self.images.append(image_path)
        return image_path

    @staticmethod
    def _extract_prompt_field(prompt: str, label: str, next_labels: List[str]) -> str:
        if next_labels:
            lookahead = "|".join(re.escape(next_label) for next_label in next_labels)
            pattern = rf"{re.escape(label)}:\s*(.*?)(?=(?:{lookahead}):|$)"
        else:
            pattern = rf"{re.escape(label)}:\s*(.*)$"
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        lines = [re.sub(r"\s+", " ", line).strip() for line in match.group(1).splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()

    def _parse_visual_prompt(self, prompt: str) -> dict:
        scene_type = self._extract_prompt_field(
            prompt,
            "Scene type",
            ["On-screen text", "Primary stat", "Secondary stat", "Summary", "Reason", "Tags", "Direction"],
        )
        on_screen_text = self._extract_prompt_field(
            prompt,
            "On-screen text",
            ["Primary stat", "Secondary stat", "Summary", "Reason", "Tags", "Direction"],
        )
        primary_stat = self._extract_prompt_field(
            prompt,
            "Primary stat",
            ["Secondary stat", "Summary", "Reason", "Tags", "Direction"],
        )
        secondary_stat = self._extract_prompt_field(
            prompt,
            "Secondary stat",
            ["Summary", "Reason", "Tags", "Direction"],
        )
        summary = self._extract_prompt_field(
            prompt,
            "Summary",
            ["Reason", "Tags", "Direction"],
        )
        reason = self._extract_prompt_field(
            prompt,
            "Reason",
            ["Tags", "Direction"],
        )
        tags_raw = self._extract_prompt_field(prompt, "Tags", ["Direction"])
        direction = self._extract_prompt_field(prompt, "Direction", [])

        title_lines = [line.strip() for line in on_screen_text.splitlines() if line.strip()]
        if not title_lines:
            title_lines = [on_screen_text or "Open Source Spotlight"]

        tags = []
        normalized_tags = (tags_raw or "").replace("\u3001", ",")
        for part in re.split(r"[|/,]+", normalized_tags):
            clean = re.sub(r"\s+", " ", part).strip(" .")
            if clean and clean not in tags:
                tags.append(clean)

        return {
            "scene_type": scene_type or "Tech spotlight",
            "title": on_screen_text or "Open Source Spotlight",
            "title_lines": title_lines[:2],
            "primary_stat": primary_stat,
            "secondary_stat": secondary_stat,
            "summary": summary,
            "reason": reason,
            "tags": tags[:4],
            "direction": direction or "Editorial infographic card",
        }

    @staticmethod
    def _placeholder_accent(scene_type: str) -> tuple[int, int, int]:
        lowered = (scene_type or "").lower()
        if "cover" in lowered:
            return (83, 174, 255)
        if "spotlight" in lowered:
            return (255, 188, 83)
        if "fast" in lowered:
            return (255, 110, 163)
        if "closing" in lowered:
            return (94, 223, 165)
        return (109, 146, 255)

    @staticmethod
    def _mix_color(base: tuple[int, int, int], other: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
        return tuple(int(base[i] * (1 - ratio) + other[i] * ratio) for i in range(3))

    def _fit_font(self, text: str, preferred_size: int, max_width: int) -> ImageFont.FreeTypeFont:
        size = preferred_size
        while size >= 30:
            font = self._load_subtitle_font(size)
            temp_image = Image.new("RGB", (10, 10))
            temp_draw = ImageDraw.Draw(temp_image)
            bbox = temp_draw.multiline_textbbox((0, 0), text, font=font, spacing=8, stroke_width=0)
            if (bbox[2] - bbox[0]) <= max_width:
                return font
            size -= 4
        return self._load_subtitle_font(30)

    def _draw_chip(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int, int],
        text_fill: tuple[int, int, int, int],
    ) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0] + 36
        height = bbox[3] - bbox[1] + 20
        x, y = xy
        draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill=fill)
        draw.text((x + 18, y + 10), text, font=font, fill=text_fill)
        return width

    @staticmethod
    def _keyword_chips(text: str, limit: int = 4) -> List[str]:
        chips = []
        normalized_text = (text or "").replace("\u3001", ",")
        for part in re.split(r"[|/,]+", normalized_text):
            clean = re.sub(r"\s+", " ", part).strip(" .")
            if not clean:
                continue
            if len(clean) > 18:
                clean = clean[:18].rstrip() + "…"
            if clean not in chips:
                chips.append(clean)
            if len(chips) >= limit:
                break
        return chips

    def _build_placeholder_template(self, parsed: dict, accent: tuple[int, int, int]) -> Image.Image:
        width, height = 1080, 1920
        background_top = self._mix_color((14, 20, 32), accent, 0.18)
        background_bottom = (10, 12, 20)
        image = Image.new("RGB", (width, height), background_bottom)
        draw = ImageDraw.Draw(image, "RGBA")

        for y in range(height):
            ratio = y / max(height - 1, 1)
            color = self._mix_color(background_top, background_bottom, ratio)
            draw.line((0, y, width, y), fill=color, width=1)

        draw.rounded_rectangle((48, 56, 1032, 1864), radius=34, outline=(255, 255, 255, 22), width=2)
        draw.rounded_rectangle((62, 70, 1018, 1850), radius=30, fill=(12, 16, 28, 140))
        draw.ellipse((760, -90, 1180, 330), fill=(*accent, 46))
        draw.ellipse((-120, 1320, 260, 1700), fill=(*self._mix_color(accent, (255, 255, 255), 0.1), 28))

        label_font = self._load_subtitle_font(28)
        badge_font = self._load_subtitle_font(26)
        title_font = self._fit_font("\n".join(parsed.get("title_lines") or [parsed.get("title", "")]), 74, 760)
        stat_font = self._load_subtitle_font(44)
        body_font = self._load_subtitle_font(34)
        small_font = self._load_subtitle_font(28)

        draw.text((88, 102), "OPEN SOURCE WEEKLY", fill=(225, 232, 244), font=label_font)
        self._draw_chip(draw, (792, 90), parsed.get("scene_type", "Spotlight"), badge_font, (*accent, 225), (8, 12, 20, 255))

        title_lines = parsed.get("title_lines") or [parsed.get("title", "Open Source Spotlight")]
        title_text = "\n".join(title_lines[:2])
        draw.multiline_text((88, 230), title_text, font=title_font, fill=(248, 250, 255), spacing=6)

        primary_stat = parsed.get("primary_stat", "").replace("?", "stars")
        secondary_stat = parsed.get("secondary_stat", "").replace("?", "stars")

        stat_top = 470
        if primary_stat:
            draw.rounded_rectangle((88, stat_top, 490, stat_top + 118), radius=28, fill=(255, 255, 255, 18))
            draw.text((118, stat_top + 30), primary_stat, font=stat_font, fill=(255, 245, 130))
        if secondary_stat:
            draw.rounded_rectangle((530, stat_top, 992, stat_top + 118), radius=28, fill=(255, 255, 255, 18))
            draw.text((560, stat_top + 30), secondary_stat, font=stat_font, fill=(225, 233, 244))

        summary = parsed.get("summary", "")
        reason = parsed.get("reason", "")
        if summary:
            draw.text((88, 660), "???", font=small_font, fill=(158, 177, 203))
            draw.multiline_text((88, 706), textwrap.fill(summary, width=21), font=body_font, fill=(240, 244, 250), spacing=10)
        if reason:
            draw.text((88, 930), "??????", font=small_font, fill=(158, 177, 203))
            draw.multiline_text((88, 976), textwrap.fill(reason, width=21), font=body_font, fill=(214, 224, 237), spacing=10)

        tags = parsed.get("tags") or []
        chip_y = 1240
        chip_x = 88
        for chip in tags[:4]:
            used = self._draw_chip(draw, (chip_x, chip_y), chip, badge_font, (255, 255, 255, 22), (235, 241, 250, 255))
            chip_x += used + 14

        draw.line((88, 1770, 992, 1770), fill=(*accent, 110), width=3)
        draw.text((88, 1790), "Tech ranking card", font=small_font, fill=(123, 141, 166))

        return image

    def generate_image(self, prompt: str) -> str:
        """
        Generates an AI Image using FLUX if configured, otherwise Nano Banana 2.

        Args:
            prompt (str): Reference for image generation

        Returns:
            path (str): The path to the generated image.
        """
        flux_path = self.generate_image_flux(prompt)
        if flux_path:
            return flux_path
        return self.generate_image_nanobanana2(prompt)

    def generate_script_to_speech(self, tts_instance: TTS) -> str:
        """
        Converts the generated script into Speech using KittenTTS and returns the path to the wav file.

        Args:
            tts_instance (tts): Instance of TTS Class.

        Returns:
            path_to_wav (str): Path to generated audio (WAV Format).
        """
        path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".wav")

        # Keep the original script intact for subtitle generation. The previous
        # version mutated self.script in place and stripped all Chinese
        # punctuation, which made the subtitle layer lose commas and periods.
        tts_script = str(self.script or "")
        tts_script = re.sub(r"\r\n?", "\n", tts_script)
        tts_script = re.sub(r"[ \t]+", " ", tts_script)
        tts_script = re.sub(r"\n{3,}", "\n\n", tts_script).strip()

        # Remove only obvious control characters that TTS engines do not need,
        # while preserving both CJK and ASCII punctuation for natural pauses.
        tts_script = re.sub(
            r"[^\w\s\u3400-\u9fff"
            r"\.,!\?;:，。！？；：、"
            r"%％\$¥￥\+\-—/／&＆"
            r"\(\)（）\[\]【】「」『』《》〈〉"
            r"\"'“”‘’·…]",
            "",
            tts_script,
        )

        tts_instance.synthesize(tts_script, path)

        self.tts_path = path

        if get_verbose():
            info(f' => Wrote TTS to "{path}"')

        return path

    def add_video(self, video: dict) -> None:
        """
        Adds a video to the cache.

        Args:
            video (dict): The video to add

        Returns:
            None
        """
        videos = self.get_videos()
        videos.append(video)

        cache = get_youtube_cache_path()

        with open(cache, "r") as file:
            previous_json = json.loads(file.read())

            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    account["videos"].append(video)

            # Commit changes
            with open(cache, "w") as f:
                f.write(json.dumps(previous_json))

    def generate_subtitles(self, audio_path: str) -> str:
        """
        Generates subtitles for the audio using the configured STT provider.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            path (str): The path to the generated SRT File.
        """
        if str(getattr(self, "script", "")).strip():
            try:
                return self.generate_subtitles_from_script(audio_path)
            except Exception as e:
                warning(f"Failed to build subtitles from script, falling back to STT: {e}")

        provider = str(get_stt_provider() or "local_whisper").lower()

        if provider == "local_whisper":
            return self.generate_subtitles_local_whisper(audio_path)

        if provider == "third_party_assemblyai":
            return self.generate_subtitles_assemblyai(audio_path)

        warning(f"Unknown stt_provider '{provider}'. Falling back to local_whisper.")
        return self.generate_subtitles_local_whisper(audio_path)

    def generate_subtitles_assemblyai(self, audio_path: str) -> str:
        """
        Generates subtitles using AssemblyAI.

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        aai.settings.api_key = get_assemblyai_api_key()
        config = aai.TranscriptionConfig()
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        subtitles = transcript.export_subtitles_srt()

        srt_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".srt")

        with open(srt_path, "w") as file:
            file.write(subtitles)

        return srt_path

    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Formats a timestamp in seconds to SRT format.

        Args:
            seconds (float): Seconds

        Returns:
            ts (str): HH:MM:SS,mmm
        """
        total_millis = max(0, int(round(seconds * 1000)))
        hours = total_millis // 3600000
        minutes = (total_millis % 3600000) // 60000
        secs = (total_millis % 60000) // 1000
        millis = total_millis % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_subtitles_from_script(self, audio_path: str) -> str:
        """
        Generates subtitles directly from the prepared script instead of re-running
        STT on the rendered speech. This preserves proper nouns and Chinese terms.

        When a ``.timing.json`` sidecar file exists next to the audio (written by
        ``TTS.synthesize``), the method uses the **actual per-chunk WAV durations**
        for precise subtitle sync.  Otherwise it falls back to proportional
        character-count estimation.

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        script = str(getattr(self, "script", "") or "").strip()
        if not script:
            raise ValueError("Script is empty.")

        chunks = self._split_script_for_subtitles(script)
        if not chunks:
            raise ValueError("Script could not be split into subtitle chunks.")

        with AudioFileClip(audio_path) as audio_clip:
            total_duration = max(float(audio_clip.duration or 0), 0.1)

        # ------ Try precise timing from TTS chunk durations ------
        timing_path = audio_path + ".timing.json"
        if os.path.exists(timing_path):
            try:
                with open(timing_path, encoding="utf-8") as f:
                    timing = json.load(f)
                chunk_durations = timing.get("chunk_durations", [])
                chunk_pauses = timing.get("chunk_pauses", [])
                chunk_texts = timing.get("chunk_texts", [])
                speed = max(float(timing.get("playback_speed", 1.0)), 0.1)

                # Prefer the texts stored by TTS (guaranteed 1:1 with durations).
                # Fall back to the re-split chunks only when texts are absent AND
                # the counts happen to match.
                if chunk_texts and len(chunk_texts) == len(chunk_durations):
                    timing_chunks = [(t, 0.0) for t in chunk_texts]
                elif len(chunk_durations) == len(chunks):
                    timing_chunks = chunks
                else:
                    timing_chunks = None

                if timing_chunks and all(d > 0 for d in chunk_durations):
                    return self._build_srt_from_timing(
                        timing_chunks, chunk_durations, chunk_pauses, speed, total_duration
                    )
            except Exception as e:
                warning(f"Failed to load TTS timing sidecar: {e}")

        # ------ Fallback: proportional character-count estimation ------
        pause_scale = max(float(PLAYBACK_SPEED or 1.0), 0.1)
        normalized_chunks = []
        for index, (text, pause_seconds) in enumerate(chunks):
            clean_text = str(text).strip()
            if not clean_text:
                continue
            adjusted_pause = float(pause_seconds or 0.0) / pause_scale
            if index == len(chunks) - 1:
                adjusted_pause = 0.0
            normalized_chunks.append((clean_text, adjusted_pause))

        if not normalized_chunks:
            raise ValueError("No non-empty subtitle chunks were produced from the script.")

        total_pause = sum(pause for _, pause in normalized_chunks)
        speech_budget = max(0.1, total_duration - total_pause)
        weights = [max(1, len(re.sub(r"\s+", "", text))) for text, _ in normalized_chunks]
        total_weight = max(sum(weights), 1)

        lines = []
        cursor = 0.0
        for idx, ((text, pause_seconds), weight) in enumerate(zip(normalized_chunks, weights), start=1):
            speech_duration = speech_budget * (weight / total_weight)
            segment_duration = max(0.3, speech_duration + pause_seconds)
            end = total_duration if idx == len(normalized_chunks) else min(total_duration, cursor + segment_duration)

            lines.append(str(idx))
            lines.append(f"{self._format_srt_timestamp(cursor)} --> {self._format_srt_timestamp(end)}")
            lines.append(text)
            lines.append("")

            cursor = end

        subtitles = "\n".join(lines)
        srt_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        return srt_path

    def _build_srt_from_timing(
        self,
        chunks: list,
        chunk_durations: list[float],
        chunk_pauses: list[float],
        speed: float,
        total_duration: float,
    ) -> str:
        """Build SRT using actual TTS chunk durations for precise subtitle sync."""
        lines: list[str] = []
        cursor = 0.0
        n = len(chunks)

        for idx, ((text, _), raw_dur) in enumerate(zip(chunks, chunk_durations), start=1):
            actual_speech = raw_dur / speed

            start = cursor
            end = min(total_duration, cursor + actual_speech)
            if idx == n:
                end = total_duration

            lines.append(str(idx))
            lines.append(
                f"{self._format_srt_timestamp(start)} --> "
                f"{self._format_srt_timestamp(end)}"
            )
            lines.append(text)
            lines.append("")

            # Advance cursor past the pause (if not the last chunk)
            if idx < n and idx - 1 < len(chunk_pauses):
                actual_pause = float(chunk_pauses[idx - 1]) / speed
            else:
                actual_pause = 0.0
            cursor = end + actual_pause

        subtitles = "\n".join(lines)
        srt_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        return srt_path

    def _split_script_for_subtitles(self, script: str) -> List[tuple[str, float]]:
        prepared = _prepare_tts_text(script)
        return [
            (str(text).strip(), float(pause or 0.0))
            for text, pause in _split_for_speech(prepared)
            if str(text).strip()
        ]

    def generate_subtitles_local_whisper(self, audio_path: str) -> str:
        """
        Generates subtitles using local Whisper (faster-whisper).

        Args:
            audio_path (str): Audio file path

        Returns:
            path (str): Path to SRT file
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            error(
                "Local STT selected but 'faster-whisper' is not installed. "
                "Install it or switch stt_provider to third_party_assemblyai."
            )
            raise

        model = WhisperModel(
            get_whisper_model(),
            device=get_whisper_device(),
            compute_type=get_whisper_compute_type(),
        )
        segments, _ = model.transcribe(audio_path, vad_filter=True)

        lines = []
        for idx, segment in enumerate(segments, start=1):
            start = self._format_srt_timestamp(segment.start)
            end = self._format_srt_timestamp(segment.end)
            text = str(segment.text).strip()

            if not text:
                continue

            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        subtitles = "\n".join(lines)
        srt_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)

        return srt_path

    def _parse_srt_entries(self, srt_path: str) -> List[dict]:
        with open(srt_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            return []

        entries = []
        blocks = re.split(r"\n\s*\n", content)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 3 or "-->" not in lines[1]:
                continue

            start_raw, end_raw = [part.strip() for part in lines[1].split("-->")]
            text = " ".join(lines[2:]).strip()
            if not text:
                continue

            entries.append(
                {
                    "start": self._srt_timestamp_to_seconds(start_raw),
                    "end": self._srt_timestamp_to_seconds(end_raw),
                    "text": text,
                }
            )

        return entries

    def _srt_timestamp_to_seconds(self, value: str) -> float:
        hours, minutes, sec_millis = value.split(":")
        seconds, millis = sec_millis.split(",")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000.0
        )

    def _wrap_subtitle_text(
        self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> str:
        def tokenize(value: str) -> List[str]:
            tokens: List[str] = []
            buffer = ""
            for char in value:
                if char == "\n":
                    if buffer:
                        tokens.append(buffer)
                        buffer = ""
                    tokens.append("\n")
                    continue
                if re.match(r"[A-Za-z0-9%$#@&+_.:/-]", char):
                    buffer += char
                    continue
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                if char.strip():
                    tokens.append(char)
                else:
                    tokens.append(" ")
            if buffer:
                tokens.append(buffer)
            return tokens

        tokens = tokenize(text)
        if not tokens:
            return text

        lines = []
        current = ""

        for token in tokens:
            if token == "\n":
                if current.strip():
                    lines.append(current.strip())
                current = ""
                continue
            candidate = f"{current}{token}"
            bbox = draw.multiline_textbbox((0, 0), candidate, font=font, spacing=16, stroke_width=4)
            if not current or (bbox[2] - bbox[0]) <= max_width:
                current = candidate
            else:
                lines.append(current.strip())
                current = token

        if current.strip():
            lines.append(current.strip())
        return "\n".join(lines)

    @staticmethod
    def _subtitle_font_candidates() -> List[str]:
        configured_font = os.path.join(get_fonts_dir(), get_font())
        return [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            configured_font,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

    def _load_subtitle_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        for font_path in self._subtitle_font_candidates():
            if not os.path.exists(font_path):
                continue
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _create_subtitle_card(
        self, text: str, canvas_size=(1080, 1920), max_width_ratio: float = 0.84
    ) -> str:
        width, height = canvas_size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        font_size = 60
        max_width = int(width * max_width_ratio)
        padding_x = 44
        padding_y = 24
        spacing = 12
        stroke_width = 4

        wrapped = text
        font = self._load_subtitle_font(font_size)
        text_width = text_height = 0
        box_width = box_height = 0

        while font_size >= 42:
            font = self._load_subtitle_font(font_size)
            wrapped = self._wrap_subtitle_text(draw, text, font, max_width)
            line_count = max(1, wrapped.count("\n") + 1)
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, stroke_width=stroke_width)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            box_width = text_width + padding_x * 2
            box_height = text_height + padding_y * 2
            if line_count <= 3 and box_width <= int(width * 0.92):
                break
            font_size -= 4

        box_width = text_width + padding_x * 2
        box_height = text_height + padding_y * 2
        x0 = int((width - box_width) / 2)
        y0 = int(height - box_height - 150)
        x1 = x0 + box_width
        y1 = y0 + box_height

        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=28,
            fill=(8, 14, 22, 180),
            outline=(255, 255, 255, 36),
            width=2,
        )
        draw.multiline_text(
            (x0 + padding_x, y0 + padding_y),
            wrapped,
            font=font,
            fill=(255, 245, 130, 255),
            spacing=spacing,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
            align="center",
        )

        subtitle_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")
        image.save(subtitle_path, format="PNG")
        return subtitle_path

    def _build_subtitle_overlays(self, subtitles_path: str, video_size=(1080, 1920)) -> List[ImageClip]:
        clips = []
        entries = self._parse_srt_entries(subtitles_path)

        for entry in entries:
            duration = max(0.1, entry["end"] - entry["start"])
            subtitle_card_path = self._create_subtitle_card(entry["text"], canvas_size=video_size)
            clip = (
                ImageClip(subtitle_card_path)
                .set_start(entry["start"])
                .set_duration(duration)
                .set_position(("center", "center"))
            )
            clips.append(clip)

        return clips

    def combine(self) -> str:
        """
        Combines everything into the final video.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        combined_image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".mp4")
        threads = get_threads()
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        req_dur = max_duration / len(self.images)

        print(colored("[+] Combining images...", "blue"))

        clips = []
        tot_dur = 0
        # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
        while tot_dur < max_duration:
            for image_path in self.images:
                clip = ImageClip(image_path)
                clip.duration = req_dur
                clip = clip.set_fps(30)

                # Not all images are same size,
                # so we need to resize them
                if round((clip.w / clip.h), 4) < 0.5625:
                    if get_verbose():
                        info(f" => Resizing Image: {image_path} to 1080x1920")
                    clip = crop(
                        clip,
                        width=clip.w,
                        height=round(clip.w / 0.5625),
                        x_center=clip.w / 2,
                        y_center=clip.h / 2,
                    )
                else:
                    if get_verbose():
                        info(f" => Resizing Image: {image_path} to 1920x1080")
                    clip = crop(
                        clip,
                        width=round(0.5625 * clip.h),
                        height=clip.h,
                        x_center=clip.w / 2,
                        y_center=clip.h / 2,
                    )
                clip = clip.resize((1080, 1920))

                # FX (Fade In)
                # clip = clip.fadein(2)

                clips.append(clip)
                tot_dur += clip.duration

        final_clip = concatenate_videoclips(clips)
        final_clip = final_clip.set_fps(30)
        random_song = choose_random_song()

        subtitle_clips = []
        try:
            subtitles_path = self.generate_subtitles(self.tts_path)
            subtitle_clips = self._build_subtitle_overlays(subtitles_path, video_size=(1080, 1920))
        except Exception as e:
            warning(f"Failed to generate subtitles, continuing without subtitles: {e}")

        random_song_clip = AudioFileClip(random_song).set_fps(44100)

        # Turn down volume
        random_song_clip = random_song_clip.fx(afx.volumex, 0.1)
        comp_audio = CompositeAudioClip([tts_clip.set_fps(44100), random_song_clip])

        final_clip = final_clip.set_audio(comp_audio)
        final_clip = final_clip.set_duration(tts_clip.duration)

        if subtitle_clips:
            final_clip = CompositeVideoClip([final_clip, *subtitle_clips])

        temp_audio_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".m4a")
        final_clip.write_videofile(
            combined_image_path,
            threads=threads,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            ffmpeg_params=["-movflags", "+faststart"],
        )

        success(f'Wrote Video to "{combined_image_path}"')

        return combined_image_path

    def generate_video(self, tts_instance: TTS) -> str:
        """
        Generates a YouTube Short based on the provided niche and language.

        Args:
            tts_instance (TTS): Instance of TTS Class.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        # Generate the Topic
        self.generate_topic()

        # Generate the Script
        self.generate_script()

        # Generate the Metadata
        self.generate_metadata()

        # Generate the Image Prompts
        self.generate_prompts()

        # Generate the Images
        for prompt in self.image_prompts:
            self.generate_image(prompt)

        # Generate the TTS
        self.generate_script_to_speech(tts_instance)

        # Combine everything
        path = self.combine()

        if get_verbose():
            info(f" => Generated Video: {path}")

        self.video_path = os.path.abspath(path)

        return path

    def get_channel_id(self) -> str:
        """
        Gets the Channel ID of the YouTube Account.

        Returns:
            channel_id (str): The Channel ID.
        """
        driver = self.browser
        driver.get("https://studio.youtube.com")
        time.sleep(2)
        channel_id = driver.current_url.split("/")[-1]
        self.channel_id = channel_id

        return channel_id

    def upload_video(self) -> bool:
        """
        Uploads the video to YouTube.

        Returns:
            success (bool): Whether the upload was successful or not.
        """
        try:
            self.get_channel_id()

            driver = self.browser
            verbose = get_verbose()

            # Go to youtube.com/upload
            driver.get("https://www.youtube.com/upload")

            # Set video file
            FILE_PICKER_TAG = "ytcp-uploads-file-picker"
            file_picker = driver.find_element(By.TAG_NAME, FILE_PICKER_TAG)
            INPUT_TAG = "input"
            file_input = file_picker.find_element(By.TAG_NAME, INPUT_TAG)
            file_input.send_keys(self.video_path)

            # Wait for upload to finish
            time.sleep(5)

            # Set title
            textboxes = driver.find_elements(By.ID, YOUTUBE_TEXTBOX_ID)

            title_el = textboxes[0]
            description_el = textboxes[-1]

            if verbose:
                info("\t=> Setting title...")

            title_el.click()
            time.sleep(1)
            title_el.clear()
            title_el.send_keys(self.metadata["title"])

            if verbose:
                info("\t=> Setting description...")

            # Set description
            time.sleep(10)
            description_el.click()
            time.sleep(0.5)
            description_el.clear()
            description_el.send_keys(self.metadata["description"])

            time.sleep(0.5)

            # Set `made for kids` option
            if verbose:
                info("\t=> Setting `made for kids` option...")

            is_for_kids_checkbox = driver.find_element(
                By.NAME, YOUTUBE_MADE_FOR_KIDS_NAME
            )
            is_not_for_kids_checkbox = driver.find_element(
                By.NAME, YOUTUBE_NOT_MADE_FOR_KIDS_NAME
            )

            if not get_is_for_kids():
                is_not_for_kids_checkbox.click()
            else:
                is_for_kids_checkbox.click()

            time.sleep(0.5)

            # Click next
            if verbose:
                info("\t=> Clicking next...")

            next_button = driver.find_element(By.ID, YOUTUBE_NEXT_BUTTON_ID)
            next_button.click()

            # Click next again
            if verbose:
                info("\t=> Clicking next again...")
            next_button = driver.find_element(By.ID, YOUTUBE_NEXT_BUTTON_ID)
            next_button.click()

            # Wait for 2 seconds
            time.sleep(2)

            # Click next again
            if verbose:
                info("\t=> Clicking next again...")
            next_button = driver.find_element(By.ID, YOUTUBE_NEXT_BUTTON_ID)
            next_button.click()

            # Set as unlisted
            if verbose:
                info("\t=> Setting as unlisted...")

            radio_button = driver.find_elements(By.XPATH, YOUTUBE_RADIO_BUTTON_XPATH)
            radio_button[2].click()

            if verbose:
                info("\t=> Clicking done button...")

            # Click done button
            done_button = driver.find_element(By.ID, YOUTUBE_DONE_BUTTON_ID)
            done_button.click()

            # Wait for 2 seconds
            time.sleep(2)

            # Get latest video
            if verbose:
                info("\t=> Getting video URL...")

            # Get the latest uploaded video URL
            driver.get(
                f"https://studio.youtube.com/channel/{self.channel_id}/videos/short"
            )
            time.sleep(2)
            videos = driver.find_elements(By.TAG_NAME, "ytcp-video-row")
            first_video = videos[0]
            anchor_tag = first_video.find_element(By.TAG_NAME, "a")
            href = anchor_tag.get_attribute("href")
            if verbose:
                info(f"\t=> Extracting video ID from URL: {href}")
            video_id = href.split("/")[-2]

            # Build URL
            url = build_url(video_id)

            self.uploaded_video_url = url

            if verbose:
                success(f" => Uploaded Video: {url}")

            # Add video to cache
            self.add_video(
                {
                    "title": self.metadata["title"],
                    "description": self.metadata["description"],
                    "url": url,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # Close the browser
            driver.quit()

            return True
        except:
            self.browser.quit()
            return False

    def get_videos(self) -> List[dict]:
        """
        Gets the uploaded videos from the YouTube Channel.

        Returns:
            videos (List[dict]): The uploaded videos.
        """
        if not os.path.exists(get_youtube_cache_path()):
            # Create the cache file
            with open(get_youtube_cache_path(), "w") as file:
                json.dump({"videos": []}, file, indent=4)
            return []

        videos = []
        # Read the cache file
        with open(get_youtube_cache_path(), "r") as file:
            previous_json = json.loads(file.read())
            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    videos = account["videos"]

        return videos
