import time
import random
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 20,
        min_interval: int = 3,
        max_interval: int = 8,
        max_retries: int = 3,
        timeout: int = 30,
        proxy_http: str = "",
        proxy_https: str = "",
    ):
        self.name = name
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

        self.use_proxy = name not in ["eastmoney", "sina", "ifeng"]
        if self.use_proxy:
            proxies = {}
            env_http = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
            env_https = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")

            config_http = proxy_http if proxy_http else env_http
            config_https = proxy_https if proxy_https else env_https

            if not config_http and not config_https:
                try:
                    import winreg

                    internet_settings = winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                        0,
                        winreg.KEY_READ,
                    )
                    try:
                        proxy_enable = winreg.QueryValueEx(
                            internet_settings, "ProxyEnable"
                        )[0]
                        if proxy_enable:
                            proxy_server = winreg.QueryValueEx(
                                internet_settings, "ProxyServer"
                            )[0]
                            if ":" in proxy_server:
                                host, port = proxy_server.split(":")
                                proxies["http"] = f"http://{host}:{port}"
                                proxies["https"] = f"http://{host}:{port}"
                            else:
                                proxies["http"] = f"http://{proxy_server}"
                                proxies["https"] = f"http://{proxy_server}"
                    except FileNotFoundError:
                        pass
                    finally:
                        winreg.CloseKey(internet_settings)
                except Exception:
                    pass

            if config_http:
                proxies["http"] = config_http
            if config_https:
                proxies["https"] = config_https

            if proxies:
                self.session.proxies.update(proxies)
                logger.info(f"Using proxy for {name}: {proxies}")
            else:
                logger.info(
                    f"No proxy configured for {name} - may fail to reach international sites"
                )
        else:
            logger.info(f"No proxy for domestic site: {name}")

        self._setup_headers()

    def _setup_headers(self):
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        self.session.headers.update(
            {
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def _refresh_headers(self):
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        self.session.headers["User-Agent"] = random.choice(user_agents)

    def _get_with_retry(self, url: str) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                self._refresh_headers()
                self._random_delay()
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                if response.encoding is None or response.encoding == "ISO-8859-1":
                    response.encoding = "utf-8"
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        return None

    def _random_delay(self):
        delay = random.uniform(self.min_interval, self.max_interval)
        time.sleep(delay)

    @abstractmethod
    def fetch_news_list(self, page: int = 1) -> List[Dict]:
        pass

    @abstractmethod
    def parse_article(self, url: str) -> Optional[Dict]:
        pass

    def crawl(self, max_articles: int = None) -> List[Dict]:
        articles = []
        page = 1

        while max_articles is None or len(articles) < max_articles:
            try:
                news_list = self.fetch_news_list(page)
                if not news_list:
                    break

                for item in news_list:
                    if max_articles and len(articles) >= max_articles:
                        break

                    article = self.parse_article(item.get("url", ""))
                    if article:
                        article["source"] = self.name
                        articles.append(article)
                        logger.info(f"Crawled: {article.get('title', 'Unknown')[:50]}")

                page += 1
                self._random_delay()

            except Exception as e:
                logger.error(f"Error crawling page {page}: {e}")
                break

        return articles
