import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\S+@\S+")
        self.html_pattern = re.compile(r"<[^>]+>")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = re.sub(r"[^\u4e00-\u9fff\w\s]", " ", text)
        text = self.whitespace_pattern.sub(" ", text)
        text = text.strip()

        return text

    def truncate(self, text: str, max_length: int = 512) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length]

    def clean_and_truncate(self, text: str, max_length: int = 512) -> str:
        return self.truncate(self.clean(text), max_length)
