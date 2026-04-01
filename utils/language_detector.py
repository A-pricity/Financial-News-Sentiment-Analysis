import logging
import re

logger = logging.getLogger(__name__)


class LanguageDetector:
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str) -> str:
        if not text or len(text.strip()) < 5:
            return "unknown"

        try:
            from langdetect import detect, LangDetectException

            sample = text[:100] if len(text) > 100 else text
            lang = detect(sample)

            if lang == "zh-cn" or lang == "zh-tw" or lang == "zh":
                return "zh"
            elif lang == "en":
                return "en"
            else:
                return self._detect_fallback(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return self._detect_fallback(text)

    def _detect_fallback(self, text: str) -> str:
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))

        if chinese_chars > english_words:
            return "zh"
        elif english_words > chinese_chars:
            return "en"
        else:
            return "unknown"

    def is_chinese(self, text: str) -> bool:
        return self.detect(text) == "zh"

    def is_english(self, text: str) -> bool:
        return self.detect(text) == "en"
