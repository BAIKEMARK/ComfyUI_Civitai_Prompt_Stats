import requests
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import folder_paths
import time

# --- Utility Functions (These remain the same) ---

def get_metadata(filepath, type):
    """Extracts metadata from a safetensors file."""
    filepath = folder_paths.get_full_path(type, filepath)
    if not filepath:
        return None
    try:
        with open(filepath, "rb") as file:
            header_size = int.from_bytes(file.read(8), "little", signed=False)
            if header_size <= 0:
                return None
            header = file.read(header_size)
            header_json = json.loads(header)
            return header_json.get("__metadata__")
    except Exception as e:
        print(f"[Civitai Prompt Stats] Error reading metadata: {e}")
        return None

def sort_tags_by_frequency(meta_tags):
    """Parses the __metadata__ json looking for trained tags and sorts them by frequency."""
    if not meta_tags or "ss_tag_frequency" not in meta_tags:
        return []
    try:
        tag_freq_json = json.loads(meta_tags["ss_tag_frequency"])
        tag_counts = Counter()
        for _, dataset in tag_freq_json.items():
            for tag, count in dataset.items():
                tag_counts[str(tag).strip()] += count
        sorted_tags = [tag for tag, _ in tag_counts.most_common()]
        return sorted_tags
    except Exception as e:
        print(f"[Civitai Prompt Stats] Error parsing tag frequency: {e}")
        return []

# --- Refactored Base Class ---

class BaseCivitaiPromptStatsNode:
    """
    Base class containing common logic to fetch community prompts from Civitai.
    It does NOT handle trigger words, as that is specific to certain model types.
    """

    FOLDER_KEY = None

    # Base class only defines the common inputs
    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = folder_paths.get_filename_list(cls.FOLDER_KEY) or []
        except Exception:
            files = []
        file_list = sorted(files, key=str.lower) if files else [""]
        return {
            "required": {
                "file_name": (file_list, {}),
                "top_n": ("INT", {"default": 20, "min": 1, "max": 200}),
                "max_pages": ("INT", {"default": 3, "min": 1, "max": 50}),
                "sort": (
                    ["Most Reactions", "Most Comments", "Newest"],
                    {"default": "Most Reactions"},
                ),
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 5}),
                "force_refresh": (["no", "yes"], {"default": "no"}),
            }
        }

    # Base class only defines the common outputs
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "execute"
    CATEGORY = "Civitai"

    # --- Caching and Helper Methods (can be used by subclasses) ---
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(CACHE_DIR, exist_ok=True)
    HASH_CACHE_FILE = os.path.join(CACHE_DIR, "hash_cache.json")
    CIVITAI_TRIGGERS_CACHE = os.path.join(CACHE_DIR, "civitai_triggers_cache.json")

    @staticmethod
    def calculate_sha256(file_path):
        print(f"[{__class__.__name__}] Calculating SHA256 for: {os.path.basename(file_path)}...")
        start_time = time.time()
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        digest = sha256_hash.hexdigest()
        duration = time.time() - start_time
        print(f"[{__class__.__name__}] SHA256 calculated in {duration:.2f} seconds.")
        return digest

    def get_cached_sha256(self, file_path):
        try:
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            cache_key = f"{file_path}|{mtime}|{size}"
            try:
                with open(self.HASH_CACHE_FILE, "r", encoding="utf-8") as f:
                    hash_cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                hash_cache = {}
            if cache_key in hash_cache:
                print(f"[{self.__class__.__name__}] Loaded hash from cache for: {os.path.basename(file_path)}")
                return hash_cache[cache_key]
            file_hash = self.calculate_sha256(file_path)
            hash_cache[cache_key] = file_hash
            with open(self.HASH_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(hash_cache, f, indent=2)
            return file_hash
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error handling hash cache: {e}")
            return self.calculate_sha256(file_path)

    @staticmethod
    def _get_model_version_info_by_hash(sha256_hash, timeout=10):
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256_hash}"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[{__class__.__name__}] Failed to fetch model version info by hash: {e}")
            return None

    def _fetch_images_page(self, model_version_id, page, sort, timeout):
        url = "https://civitai.com/api/v1/images"
        params = {
            "modelVersionId": model_version_id,
            "limit": 100,
            "page": page,
            "sort": sort,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_prompts(prompt_text: str):
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return []
        pattern = re.compile(r"<[^>]+>|\[[^\]]+\]|\([^)]+\)|[^,]+")
        tags = pattern.findall(prompt_text)
        return [tag.strip() for tag in tags if tag.strip()]

    def _format_tags_with_counts(self, items, top_n):
        out_lines = [f'{i} : "{tag}" ({count})' for i, (tag, count) in enumerate(items[:top_n])]
        return "\n".join(out_lines)

    def execute(
        self, file_name, top_n, max_pages, sort, timeout, retries, force_refresh
    ):
        """This base execute method ONLY fetches community prompts."""
        file_path = folder_paths.get_full_path(self.FOLDER_KEY, file_name)
        if not file_path or not os.path.exists(file_path):
            print(f"[{self.__class__.__name__}] File not found: {file_path}")
            return ("", "")

        try:
            file_hash = self.get_cached_sha256(file_path)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to get file hash: {e}")
            return ("", "")

        cache_file = os.path.join(self.CACHE_DIR, f"{file_hash}_{sort}_{max_pages}.json")
        if force_refresh == "no" and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                pos_counts = cached_data.get("pos_counts", [])
                neg_counts = cached_data.get("neg_counts", [])
                pos_text = self._format_tags_with_counts(pos_counts, top_n)
                neg_text = self._format_tags_with_counts(neg_counts, top_n)
                print(f"[{self.__class__.__name__}] Loaded prompt stats from cache: {os.path.basename(cache_file)}")
                return (pos_text, neg_text)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Failed to load from cache, fetching new data. Error: {e}")

        model_info = self._get_model_version_info_by_hash(file_hash, timeout=timeout)
        if not model_info or not model_info.get("id"):
            print(f"[{self.__class__.__name__}] Could not find model info on Civitai.")
            return ("", "")

        model_version_id = model_info.get("id")
        pages, pos_tokens, neg_tokens = range(1, max_pages + 1), [], []
        sort_param = (sort if sort in ("Most Reactions", "Most Comments", "Newest") else "Newest")

        with ThreadPoolExecutor(max_workers=min(10, len(pages))) as executor:
            future_to_page = {executor.submit(self._fetch_images_page, model_version_id, p, sort_param, timeout): p for p in pages}
            for future in as_completed(future_to_page):
                try:
                    data = future.result()
                    for item in data.get("items", []):
                        if meta := item.get("meta", {}):
                            pos_tokens.extend(self._parse_prompts(meta.get("prompt", "")))
                            neg_tokens.extend(self._parse_prompts(meta.get("negativePrompt", "")))
                except Exception as exc:
                    print(f"[{self.__class__.__name__}] Page generated an exception: {exc}")

        pos_counts = Counter(pos_tokens).most_common()
        neg_counts = Counter(neg_tokens).most_common()
        pos_text = self._format_tags_with_counts(pos_counts, top_n)
        neg_text = self._format_tags_with_counts(neg_counts, top_n)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"pos_counts": pos_counts, "neg_counts": neg_counts},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to save cache: {e}")

        return (pos_text, neg_text)

# --- Subclass for CKPT (Simple, inherits everything it needs) ---

class CivitaiPromptStatsCKPT(BaseCivitaiPromptStatsNode):
    FOLDER_KEY = "checkpoints"

    # We can explicitly define these, but it inherits them from the base class anyway
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")

# --- Subclass for LORA (Adds its own logic and outputs) ---

class CivitaiPromptStatsLORA(BaseCivitaiPromptStatsNode):
    FOLDER_KEY = "loras"

    # LORA node has four outputs
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "metadata_triggers",
        "civitai_triggers",
    )

    def _get_civitai_triggers(self, file_name, file_hash, force_refresh):
        """Gets trigger words from Civitai API and uses its own cache."""
        try:
            with open(self.CIVITAI_TRIGGERS_CACHE, "r", encoding="utf-8") as f:
                trigger_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trigger_cache = {}

        if force_refresh == "no" and file_name in trigger_cache:
            print(f"[{self.__class__.__name__}] Loaded civitai triggers from cache for: {file_name}")
            return trigger_cache[file_name]

        print(f"[{self.__class__.__name__}] Requesting civitai triggers from API for: {file_name}")
        model_info = self._get_model_version_info_by_hash(file_hash)
        triggers = (
            model_info.get("trainedWords", [])
            if model_info and isinstance(model_info.get("trainedWords"), list)
            else []
        )

        trigger_cache[file_name] = triggers
        try:
            with open(self.CIVITAI_TRIGGERS_CACHE, "w", encoding="utf-8") as f:
                json.dump(trigger_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to save civitai triggers cache: {e}")

        return triggers

    def execute(
        self, file_name, top_n, max_pages, sort, timeout, retries, force_refresh
    ):
        """LORA-specific execute method that adds trigger word fetching."""

        # --- Step 1: Perform LORA-specific logic (get trigger words) ---
        file_path = folder_paths.get_full_path(self.FOLDER_KEY, file_name)
        if not file_path or not os.path.exists(file_path):
            print(f"[{self.__class__.__name__}] File not found: {file_path}")
            return ("", "", "", "")

        # Source 1: Local file metadata
        metadata = get_metadata(file_name, self.FOLDER_KEY)
        metadata_triggers_list = sort_tags_by_frequency(metadata)

        # Source 2: Civitai API (trainedWords)
        civitai_triggers_list = []
        try:
            file_hash = self.get_cached_sha256(file_path)
            civitai_triggers_list = self._get_civitai_triggers(file_name, file_hash, force_refresh)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to get file hash for civitai triggers: {e}")

        metadata_triggers_str = (
            ", ".join(metadata_triggers_list)
            if metadata_triggers_list
            else "[空: 未在模型元数据中找到触发词]"
        )
        civitai_triggers_str = (
            ", ".join(civitai_triggers_list)
            if civitai_triggers_list
            else "[空: 未在 Civitai API 中找到触发词]"
        )

        # --- Step 2: Call the base class's execute method to get common data ---
        # This reuses all the logic for fetching community prompts
        pos_text, neg_text = super().execute(file_name, top_n, max_pages, sort, timeout, retries, force_refresh)

        # --- Step 3: Combine and return all results ---
        return (pos_text, neg_text, metadata_triggers_str, civitai_triggers_str)

# --- Node Mappings ---

NODE_CLASS_MAPPINGS = {
    "CivitaiPromptStatsCKPT": CivitaiPromptStatsCKPT,
    "CivitaiPromptStatsLORA": CivitaiPromptStatsLORA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiPromptStatsCKPT": "Civitai Prompt Stats (CKPT)",
    "CivitaiPromptStatsLORA": "Civitai Prompt Stats (LORA)",
}
