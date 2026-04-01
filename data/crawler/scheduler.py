import logging
import json
import os
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

from .domestic_crawlers import EastMoneyCrawler, SinaCrawler, PhoenixCrawler
from .international_crawlers import (
    ReutersCrawler,
    BloombergCrawler,
    CNBCCrawler,
    YahooFinanceCrawler,
)

logger = logging.getLogger(__name__)


class CrawlerScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("batch_size", 20)
        self.min_interval = config.get("min_interval", 3)
        self.max_interval = config.get("max_interval", 8)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        self.target_total = config.get("target_total", 100000)

        self.crawlers = {
            "eastmoney": EastMoneyCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "sina": SinaCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "ifeng": PhoenixCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "reuters": ReutersCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "bloomberg": BloombergCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "cnbc": CNBCCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
            "yahoo": YahooFinanceCrawler(
                batch_size=self.batch_size,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                max_retries=self.max_retries,
                timeout=self.timeout,
            ),
        }

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.output_dir = os.path.join(project_root, "data/processed")
        os.makedirs(self.output_dir, exist_ok=True)

    def crawl_all(self) -> List[Dict]:
        all_articles = []

        sources = list(self.crawlers.keys())
        articles_per_source = self.target_total // len(sources)

        logger.info(
            f"Target: {self.target_total} articles, ~{articles_per_source} per source"
        )

        for source_name in tqdm(sources, desc="Crawling sources"):
            crawler = self.crawlers[source_name]
            logger.info(f"Starting crawl for {source_name}...")

            try:
                articles = crawler.crawl(max_articles=articles_per_source)
                all_articles.extend(articles)
                logger.info(f"Crawled {len(articles)} articles from {source_name}")
            except Exception as e:
                logger.error(f"Error crawling {source_name}: {e}")

        output_file = os.path.join(
            self.output_dir, f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(all_articles)} articles to {output_file}")

        return all_articles

    def load_checkpoint(self, checkpoint_file: str) -> List[Dict]:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_checkpoint(self, articles: List[Dict], checkpoint_file: str):
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    scheduler = CrawlerScheduler(config["crawler"])
    scheduler.crawl_all()


if __name__ == "__main__":
    main()
