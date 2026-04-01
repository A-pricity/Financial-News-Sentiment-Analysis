import logging
from typing import List, Dict, Optional
from .base_crawler import BaseCrawler

logger = logging.getLogger(__name__)


class ReutersCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="reuters", **kwargs)
        self.base_urls = [
            "https://www.reuters.com/business/finance/global-markets-view-usa-2026-03-30/",
            "https://www.reuters.com/business/energy/what-g7-countries-are-doing-cap-energy-prices-2026-03-30/",
            "https://www.reuters.com/business/finance/swiss-cling-cash-survey-shows-payment-app-use-stalling-2026-03-30/",
            "https://www.reuters.com/markets/quote/.FTSE/",
        ]

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        news_list = []

        for url in self.base_urls:
            response = self._get_with_retry(url)
            if not response:
                news_list.append(
                    {"url": url, "title": url.split("/")[-2] if "/" in url else url}
                )
                continue

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "lxml")
                title_elem = soup.select_one("h1")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title:
                        news_list.append({"url": url, "title": title})
                        continue

                news_list.append(
                    {"url": url, "title": url.split("/")[-2] if "/" in url else url}
                )
            except Exception as e:
                logger.error(f"Error parsing Reuters URL {url}: {e}")
                news_list.append(
                    {"url": url, "title": url.split("/")[-2] if "/" in url else url}
                )

        return news_list[:10]

    def parse_article(self, url: str) -> Optional[Dict]:
        if not url:
            return None

        response = self._get_with_retry(url)
        if not response:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            title = (
                soup.select_one("h1").get_text(strip=True)
                if soup.select_one("h1")
                else ""
            )
            content_elem = soup.select_one("article, div.article-body")
            content = content_elem.get_text(strip=True) if content_elem else ""

            time_elem = soup.select_one("time")
            publish_time = time_elem.get("datetime", "") if time_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing Reuters article: {e}")
            return None


class BloombergCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="bloomberg", **kwargs)
        self.base_urls = [
            "https://www.bloomberg.com/quote/SPX:IND",
            "https://www.bloomberg.com/news/live-blog/2026-03-29/iran-latest?srnd=homepage-americas",
            "https://www.bloomberg.com/quote/TPX:IND",
        ]

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        news_list = []

        for url in self.base_urls:
            response = self._get_with_retry(url)
            if not response:
                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )
                continue

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "lxml")
                title_elem = soup.select_one("h1")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title:
                        news_list.append({"url": url, "title": title})
                        continue

                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )
            except Exception as e:
                logger.error(f"Error parsing Bloomberg URL {url}: {e}")
                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )

        return news_list[:10]

    def parse_article(self, url: str) -> Optional[Dict]:
        if not url:
            return None

        response = self._get_with_retry(url)
        if not response:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            title = (
                soup.select_one("h1").get_text(strip=True)
                if soup.select_one("h1")
                else ""
            )
            content_elem = soup.select_one("article, div.article-body")
            content = content_elem.get_text(strip=True) if content_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": "",
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing Bloomberg article: {e}")
            return None


class CNBCCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="cnbc", **kwargs)
        self.base_url = "https://www.cnbc.com"

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        url = f"{self.base_url}/"
        response = self._get_with_retry(url)
        if not response:
            return []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")
            news_list = []

            for item in soup.select("div.news-item, article, a, div.card"):
                title_elem = (
                    item.select_one("a") if hasattr(item, "select_one") else item
                )
                if title_elem and hasattr(title_elem, "get"):
                    href = title_elem.get("href", "")
                    if href and not href.startswith("http"):
                        href = self.base_url + href
                    if href.startswith("http") and "cnbc.com" in href:
                        title_text = title_elem.get_text(strip=True)
                        if title_text and len(title_text) > 10:
                            news_list.append(
                                {
                                    "url": href,
                                    "title": title_text,
                                }
                            )

            return news_list[:20]
        except Exception as e:
            logger.error(f"Error parsing CNBC news list: {e}")
            return []

    def parse_article(self, url: str) -> Optional[Dict]:
        if not url:
            return None

        response = self._get_with_retry(url)
        if not response:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            title = (
                soup.select_one("h1").get_text(strip=True)
                if soup.select_one("h1")
                else ""
            )
            content_elem = soup.select_one("article, div.article-body")
            content = content_elem.get_text(strip=True) if content_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": "",
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing CNBC article: {e}")
            return None


class YahooFinanceCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="yahoo", **kwargs)
        self.base_urls = [
            "https://finance.yahoo.com/news/live/tech-stocks-today-big-tech-stocks-sell-off-anthropic-considers-ipo-as-soon-as-q4-144220271.html",
            "https://finance.yahoo.com/news/trump-threatens-iran-energy-sites-114346101.html",
        ]

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        news_list = []

        for url in self.base_urls:
            response = self._get_with_retry(url)
            if not response:
                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )
                continue

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "lxml")
                title_elem = soup.select_one("h1")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title:
                        news_list.append({"url": url, "title": title})
                        continue

                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )
            except Exception as e:
                logger.error(f"Error parsing Yahoo URL {url}: {e}")
                news_list.append(
                    {"url": url, "title": url.split("/")[-1] if "/" in url else url}
                )

        return news_list[:10]

    def parse_article(self, url: str) -> Optional[Dict]:
        if not url:
            return None

        response = self._get_with_retry(url)
        if not response:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            title = (
                soup.select_one("h1").get_text(strip=True)
                if soup.select_one("h1")
                else ""
            )
            content_elem = soup.select_one("article, div.caas-body")
            content = content_elem.get_text(strip=True) if content_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": "",
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance article: {e}")
            return None
