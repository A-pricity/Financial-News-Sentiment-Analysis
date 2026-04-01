import os
import json
import logging
import hashlib
from typing import Dict, List, Set, Optional
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, checkpoint_interval: int = 50):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.crawled_urls: Dict[str, Set[str]] = {}
        self.articles_buffer: List[Dict] = []
        self.counter = 0
        self.lock = Lock()

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, "crawler_checkpoint.json")
        self.data_file = os.path.join(checkpoint_dir, "crawled_data.json")

        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for source, urls in data.get("crawled_urls", {}).items():
                        self.crawled_urls[source] = set(urls)
                    self.counter = data.get("counter", 0)
                    logger.info(
                        f"Loaded checkpoint: {sum(len(v) for v in self.crawled_urls.values())} URLs already crawled"
                    )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def is_crawled(self, source: str, url: str) -> bool:
        with self.lock:
            if source not in self.crawled_urls:
                self.crawled_urls[source] = set()
            return self._url_hash(url) in self.crawled_urls[source]

    def add_article(self, source: str, url: str, article: Dict) -> bool:
        with self.lock:
            if source not in self.crawled_urls:
                self.crawled_urls[source] = set()

            url_hash = self._url_hash(url)
            if url_hash in self.crawled_urls[source]:
                return False

            self.crawled_urls[source].add(url_hash)
            article["crawled_at"] = datetime.now().isoformat()
            self.articles_buffer.append(article)
            self.counter += 1

            if self.counter % self.checkpoint_interval == 0:
                self._save_checkpoint()
                self._save_data()
                logger.info(f"Checkpoint saved: {self.counter} articles")

            return True

    def _save_checkpoint(self):
        data = {
            "crawled_urls": {k: list(v) for k, v in self.crawled_urls.items()},
            "counter": self.counter,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _save_data(self):
        existing = []
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except:
                existing = []

        existing.extend(self.articles_buffer)
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        self.articles_buffer = []

    def finalize(self) -> List[Dict]:
        with self.lock:
            self._save_checkpoint()
            self._save_data()

            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []

    def get_progress(self) -> Dict:
        with self.lock:
            return {
                "total_crawled": self.counter,
                "by_source": {k: len(v) for k, v in self.crawled_urls.items()},
            }
