import os
import sys
import logging
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.crawler.parallel_scheduler import ParallelCrawlerScheduler
from data.crawler.logger import setup_logging


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["crawler"].get("log_dir", "logs")
    log_level = config["crawler"].get("log_level", "INFO")
    setup_logging(log_dir, log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting parallel crawler...")
    logger.info(f"Target: {config['crawler']['target_total']} articles")
    logger.info(f"Workers: {config['crawler']['max_workers']}")

    scheduler = ParallelCrawlerScheduler(config["crawler"])

    status = scheduler.get_status()
    logger.info(f"Checkpoint status: {status['progress']}")

    articles = scheduler.crawl_all_parallel()

    logger.info(f"Crawling completed! Total articles: {len(articles)}")


if __name__ == "__main__":
    main()
