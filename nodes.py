from inspect import cleandoc
import requests
import hashlib
import json
import os
import re  # 导入 re 模块
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import folder_paths
import time


class BaseCivitaiPromptStatsNode:
    """
    Base Civitai Prompt Stats Node
    提供通用逻辑：file -> hash -> model-version -> images
    子类只需定义 FOLDER_KEY 即可
    """

    FOLDER_KEY = None  # 由子类定义 ("checkpoints" / "loras")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = folder_paths.get_filename_list(cls.FOLDER_KEY) or []
        except Exception:
            files = []
        file_list = sorted(files, key=str.lower) if files else [""]

        return {
            "required": {
                "file_name": (file_list, {"default": file_list[0], "tooltip": f"选择{cls.FOLDER_KEY}文件"}),
                "top_n": ("INT", {"default": 20, "min": 1, "max": 200}),
                "max_pages": ("INT", {"default": 3, "min": 1, "max": 50}),
                "sort": (["Most Reactions", "Most Comments", "Newest"], {"default": "Most Reactions"}),
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 5}),
                "force_refresh": (["no", "yes"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "execute"
    CATEGORY = "Civitai"

    # PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(CACHE_DIR, exist_ok=True)
    HASH_CACHE_FILE = os.path.join(CACHE_DIR, "hash_cache.json")

    @staticmethod
    def calculate_sha256(file_path):
        """计算文件的 SHA256 哈希"""
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
        """
        获取文件的 SHA256 哈希，优先从缓存读取。
        缓存键基于文件路径、修改时间和大小。
        """
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
            # Fallback to direct calculation if caching fails
            return self.calculate_sha256(file_path)

    @staticmethod
    def _get_model_version_info_by_hash(sha256_hash, timeout=10):
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256_hash}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch_images_page(self, model_version_id, page, sort, timeout):
        url = "https://civitai.com/api/v1/images"
        params = {"modelVersionId": model_version_id, "limit": 100, "page": page, "sort": sort}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_prompts(prompt_text: str):
        """
        增强的提示词解析器，可以处理带权重的词组、LoRA等。
        例如：(masterpiece:1.2), <lora:name:1>, [word1|word2]
        """
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return []
        # 正则表达式：匹配 <...>、[...]、(...) 或被逗号分隔的普通词组
        pattern = re.compile(r"<[^>]+>|\[[^\]]+\]|\([^)]+\)|[^,]+")
        tags = pattern.findall(prompt_text)
        return [tag.strip() for tag in tags if tag.strip()]

    def _format_tags_with_counts(self, items, top_n):
        """
        items: list of (tag, count)
        output each line: index : "tag" (count)
        """
        out_lines = []
        idx = 0
        for tag, cnt in items:
            if not tag:
                continue
            t = str(tag).strip()
            if not t:
                continue
            out_lines.append(f'{idx} : "{t}" ({cnt})')
            idx += 1
            if idx >= top_n:
                break
        return "\n".join(out_lines)

    def execute(self, file_name, top_n, max_pages, sort, timeout, retries, force_refresh):
        file_path = folder_paths.get_full_path(self.FOLDER_KEY, file_name)
        if not file_path or not os.path.exists(file_path):
            print(f"[{self.__class__.__name__}] 本地文件未找到: {file_path}")
            return ("", "")

        # 优化1: 优先从缓存获取文件哈希
        try:
            file_hash = self.get_cached_sha256(file_path)
        except Exception as e:
            print(f"[{self.__class__.__name__}] 计算或获取文件 hash 失败: {e}")
            return ("", "")

        # 优化2: 优化缓存键，移除 top_n
        cache_file = os.path.join(self.CACHE_DIR, f"{file_hash}_{sort}_{max_pages}.json")
        if force_refresh == "no" and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                pos_counts = cached_data.get("pos_counts", [])
                neg_counts = cached_data.get("neg_counts", [])

                # 直接使用缓存数据进行格式化输出
                pos_text = self._format_tags_with_counts(pos_counts, top_n)
                neg_text = self._format_tags_with_counts(neg_counts, top_n)
                print(f"[{self.__class__.__name__}] Loaded prompt stats from cache: {os.path.basename(cache_file)}")
                return (pos_text, neg_text)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Failed to load from cache, fetching new data. Error: {e}")
                pass

        # step1: get model version via hash
        try:
            model_info = self._get_model_version_info_by_hash(file_hash, timeout=timeout)
        except Exception as e:
            print(f"[{self.__class__.__name__}] 获取 model-version 失败: {e}")
            return ("", "")

        model_version_id = model_info.get("id") if isinstance(model_info, dict) else None
        if not model_version_id:
            print(f"[{self.__class__.__name__}] modelVersionId 未找到")
            return ("", "")

        # step2: fetch pages concurrently
        pages = list(range(1, max_pages + 1))
        pos_tokens, neg_tokens = [], []

        sort_param = sort if sort in ("Most Reactions", "Most Comments", "Newest") else "Newest"

        max_workers = min(10, max(1, len(pages)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for p in pages:

                def make_task(page_num):
                    attempt = 0
                    while True:
                        try:
                            return self._fetch_images_page(model_version_id, page_num, sort_param, timeout)
                        except requests.Timeout:
                            attempt += 1
                            print(f"[{self.__class__.__name__}] Timeout page {page_num}, attempt {attempt}")
                            if attempt > retries:
                                return {"items": []}
                            time.sleep(0.2)
                        except requests.RequestException as e:
                            attempt += 1
                            print(f"[{self.__class__.__name__}] RequestException page {page_num}: {e} attempt {attempt}")
                            if attempt > retries:
                                return {"items": []}
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"[{self.__class__.__name__}] Unexpected error page {page_num}: {e}")
                            return {"items": []}

                futures[executor.submit(make_task, p)] = p

            for fut in as_completed(futures):
                data = fut.result()
                items = data.get("items", []) if isinstance(data, dict) else []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    meta = item.get("meta") or {}
                    if not isinstance(meta, dict):
                        continue
                    prompt = meta.get("prompt") or ""
                    negprompt = meta.get("negativePrompt") or ""

                    # 优化3: 使用增强的解析器
                    pos_tokens.extend(self._parse_prompts(prompt))
                    neg_tokens.extend(self._parse_prompts(negprompt))

        pos_counts = Counter(pos_tokens).most_common()
        neg_counts = Counter(neg_tokens).most_common()

        # 格式化输出
        pos_text = self._format_tags_with_counts(pos_counts, top_n)
        neg_text = self._format_tags_with_counts(neg_counts, top_n)

        # 优化2: 缓存原始统计数据而非格式化文本
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                cache_content = {"pos_counts": pos_counts, "neg_counts": neg_counts}
                json.dump(cache_content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.__class__.__name__}] 保存缓存失败: {e}")

        return (pos_text, neg_text)


# 子类: CKPT
class CivitaiPromptStatsCKPT(BaseCivitaiPromptStatsNode):
    FOLDER_KEY = "checkpoints"

# 子类: LORA
class CivitaiPromptStatsLORA(BaseCivitaiPromptStatsNode):
    FOLDER_KEY = "loras"


# 注册节点
NODE_CLASS_MAPPINGS = {
    "CivitaiPromptStatsCKPT": CivitaiPromptStatsCKPT,
    "CivitaiPromptStatsLORA": CivitaiPromptStatsLORA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiPromptStatsCKPT": "Civitai Prompt Stats (CKPT)",
    "CivitaiPromptStatsLORA": "Civitai Prompt Stats (LORA)",
}
