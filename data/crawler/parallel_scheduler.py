import os
import json
import logging
import threading
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm

from .domestic_crawlers import EastMoneyCrawler, SinaCrawler, PhoenixCrawler
from .international_crawlers import (
    ReutersCrawler,
    BloombergCrawler,
    CNBCCrawler,
    YahooFinanceCrawler,
)
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class ParallelCrawlerScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.max_workers = config.get("max_workers", 4)
        self.checkpoint_interval = config.get("checkpoint_interval", 50)
        self.target_total = config.get("target_total", 100000)

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.checkpoint_dir = os.path.join(project_root, "checkpoints")
        self.output_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(self.output_dir, exist_ok=True)

        self.checkpoint = CheckpointManager(
            self.checkpoint_dir, self.checkpoint_interval
        )

        self._init_crawlers(config)

    def _init_crawlers(self, config: dict):
        common_kwargs = {
            "min_interval": config.get("min_interval", 1),
            "max_interval": config.get("max_interval", 2),
            "max_retries": config.get("max_retries", 3),
            "timeout": config.get("timeout", 30),
            "proxy_http": config.get("proxy_http", ""),
            "proxy_https": config.get("proxy_https", ""),
        }

        self.crawlers = {
            "eastmoney": EastMoneyCrawler(**common_kwargs),
            "sina": SinaCrawler(**common_kwargs),
            "ifeng": PhoenixCrawler(**common_kwargs),
            "reuters": ReutersCrawler(**common_kwargs),
            "bloomberg": BloombergCrawler(**common_kwargs),
            "cnbc": CNBCCrawler(**common_kwargs),
            "yahoo": YahooFinanceCrawler(**common_kwargs),
        }

    def _crawl_source(self, source_name: str, max_articles: int) -> tuple:
        crawler = self.crawlers[source_name]
        articles = []
        errors = []

        try:
            news_list = crawler.fetch_news_list()
            logger.info(f"[{source_name}] Found {len(news_list)} articles")

            for item in news_list[:max_articles]:
                url = item.get("url", "")

                if self.checkpoint.is_crawled(source_name, url):
                    logger.debug(
                        f"[{source_name}] Skipping already crawled: {url[:50]}"
                    )
                    continue

                try:
                    article = crawler.parse_article(url)
                    if article:
                        article["source"] = source_name
                        if self.checkpoint.add_article(source_name, url, article):
                            articles.append(article)
                            logger.debug(
                                f"[{source_name}] Crawled: {article.get('title', '')[:40]}"
                            )
                except Exception as e:
                    errors.append(f"[{source_name}] Error parsing {url}: {e}")
                    logger.warning(f"[{source_name}] Error: {e}")

        except Exception as e:
            errors.append(f"[{source_name}] Crawler error: {e}")
            logger.error(f"[{source_name}] Crawler failed: {e}")

        return source_name, articles, errors

    def crawl_all_parallel(self) -> List[Dict]:
        sources = list(self.crawlers.keys())
        articles_per_source = self.target_total // len(sources)

        logger.info(f"Starting parallel crawl with {self.max_workers} workers")
        logger.info(
            f"Target: {self.target_total} articles, ~{articles_per_source} per source"
        )

        progress = self.checkpoint.get_progress()
        if progress["total_crawled"] > 0:
            logger.info(
                f"Resuming from checkpoint: {progress['total_crawled']} already crawled"
            )

        all_articles = []
        all_errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._crawl_source, source, articles_per_source): source
                for source in sources
            }

            with tqdm(total=len(sources), desc="Crawling sources") as pbar:
                for future in as_completed(futures):
                    source_name, articles, errors = future.result()
                    all_articles.extend(articles)
                    all_errors.extend(errors)
                    pbar.update(1)
                    pbar.set_postfix(
                        {"source": source_name, "total": len(all_articles)}
                    )

        final_articles = self.checkpoint.finalize()

        output_file = os.path.join(
            self.output_dir, f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_articles, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(final_articles)} articles to {output_file}")

        if all_errors:
            logger.warning(f"Total errors: {len(all_errors)}")

        return final_articles

    def get_status(self) -> Dict:
        return {
            "progress": self.checkpoint.get_progress(),
            "config": {
                "max_workers": self.max_workers,
                "checkpoint_interval": self.checkpoint_interval,
                "target_total": self.target_total,
            },
        }
