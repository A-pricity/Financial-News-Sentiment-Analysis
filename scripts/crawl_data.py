import os
import sys
import logging
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.crawler.scheduler import CrawlerScheduler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Starting data crawling...")
    logger.info(f"Target: {config['crawler']['target_total']} articles")

    scheduler = CrawlerScheduler(config["crawler"])

    articles = scheduler.crawl_all()

    logger.info(f"Crawling completed! Total articles: {len(articles)}")


if __name__ == "__main__":
    main()
