from inspect import cleandoc
import requests
import hashlib
import json
import os
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

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(PROJECT_ROOT,"data")
    os.makedirs(CACHE_DIR, exist_ok=True)

    @staticmethod
    def calculate_sha256(file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

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

        # compute file hash
        try:
            file_hash = self.calculate_sha256(file_path)
        except Exception as e:
            print(f"[{self.__class__.__name__}] 计算文件 hash 失败: {e}")
            return ("", "")

        cache_file = os.path.join(self.CACHE_DIR, f"{file_hash}_{sort}_{max_pages}_{top_n}.json")
        if force_refresh == "no" and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return (cached.get("positive_text",""), cached.get("negative_text",""))
            except Exception:
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
                    if isinstance(prompt, str) and prompt.strip():
                        pos_tokens.extend([t.strip() for t in prompt.split(",") if t.strip()])
                    if isinstance(negprompt, str) and negprompt.strip():
                        neg_tokens.extend([t.strip() for t in negprompt.split(",") if t.strip()])

        pos_counts = Counter(pos_tokens).most_common()
        neg_counts = Counter(neg_tokens).most_common()

        pos_text = self._format_tags_with_counts(pos_counts, top_n)
        neg_text = self._format_tags_with_counts(neg_counts, top_n)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"positive_text": pos_text, "negative_text": neg_text}, f, ensure_ascii=False, indent=2)
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
