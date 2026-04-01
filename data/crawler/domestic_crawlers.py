import logging
from typing import List, Dict, Optional
from .base_crawler import BaseCrawler

logger = logging.getLogger(__name__)


class EastMoneyCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="eastmoney", **kwargs)
        self.news_list_url = "https://finance.eastmoney.com/"
        self.article_base = "https://finance.eastmoney.com"

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        news_list = []

        response = self._get_with_retry(self.news_list_url)
        if response and response.status_code == 200:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if href and title and len(title) > 10:
                    if href.startswith("http"):
                        if "eastmoney.com" in href and ("/a/" in href or "/c/" in href):
                            news_list.append({"url": href, "title": title})
                    elif "eastmoney" in href:
                        news_list.append(
                            {"url": self.article_base + href, "title": title}
                        )

        return news_list[:50]

    def parse_article(self, url: str) -> Optional[Dict]:
        if not url:
            return None

        response = self._get_with_retry(url)
        if not response:
            return None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            title_elem = soup.find("h1") or soup.select_one("title")
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Try multiple selectors for content
            content_elem = (
                soup.select_one("div.txtinfos")
                or soup.select_one("#Content")
                or soup.select_one(".article_content")
                or soup.select_one(".content")
                or soup.select_one("article")
                or soup.select_one("main")
            )
            content = ""
            if content_elem:
                # Get text from paragraphs
                paragraphs = content_elem.find_all("p")
                if paragraphs:
                    content = " ".join(p.get_text(strip=True) for p in paragraphs)
                else:
                    content = content_elem.get_text(strip=True)

            # Try to find publish time
            time_elem = soup.select_one(".time, .date, .publish_time, time")
            publish_time = time_elem.get_text(strip=True) if time_elem else ""

            return {
                "title": title,
                "content": content[:1000] if content else "",  # Limit content length
                "publish_time": publish_time,
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing East Money article: {e}")
            return None


class SinaCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="sina", **kwargs)
        self.news_list_url = "https://finance.sina.com.cn/"
        self.article_base = "https://finance.sina.com.cn"

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        news_list = []

        response = self._get_with_retry(self.news_list_url)
        if not response:
            return []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if href and title and len(title) > 10:
                    if href.startswith("http"):
                        if "sina.com.cn" in href and (
                            "/roll/" in href or "/stock/" in href
                        ):
                            news_list.append({"url": href, "title": title})
                    elif "sina.com.cn" in href:
                        news_list.append(
                            {"url": self.article_base + href, "title": title}
                        )

        except Exception as e:
            logger.error(f"Error parsing Sina news list: {e}")

        return news_list[:50]

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
                soup.select_one("h1.main-title, h1").get_text(strip=True)
                if soup.select_one("h1.main-title, h1")
                else ""
            )
            content_elem = soup.select_one("div.article-content, div#artibody")
            content = content_elem.get_text(strip=True) if content_elem else ""

            time_elem = soup.select_one("span.date, .time")
            publish_time = time_elem.get_text(strip=True) if time_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing Sina article: {e}")
            return None


class PhoenixCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(name="ifeng", **kwargs)
        self.news_list_url = "https://finance.ifeng.com"
        self.article_base = "https://finance.ifeng.com"

    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        response = self._get_with_retry(self.news_list_url)
        if not response:
            return []

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "lxml")
            news_list = []

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if href and title and len(title) > 10:
                    if href.startswith("http"):
                        if "ifeng.com" in href and ("/c/" in href or "/a/" in href):
                            news_list.append({"url": href, "title": title})
                    elif "ifeng.com" in href:
                        news_list.append(
                            {"url": self.article_base + href, "title": title}
                        )

            return news_list[:50]
        except Exception as e:
            logger.error(f"Error parsing Phoenix news list: {e}")
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
                soup.select_one("h1#artical_title, h1").get_text(strip=True)
                if soup.select_one("h1#artical_title, h1")
                else ""
            )
            content_elem = soup.select_one(
                "div[class*='main_content'], div[class*='text_D'], div[class*='articleBox']"
            )
            content = ""
            if content_elem:
                paragraphs = content_elem.find_all("p")
                if paragraphs:
                    content = " ".join(p.get_text(strip=True) for p in paragraphs)
                else:
                    content = content_elem.get_text(strip=True)

            time_elem = soup.select_one("span.ss01, .time")
            publish_time = time_elem.get_text(strip=True) if time_elem else ""

            return {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "url": url,
            }
        except Exception as e:
            logger.error(f"Error parsing Phoenix article: {e}")
            return None
