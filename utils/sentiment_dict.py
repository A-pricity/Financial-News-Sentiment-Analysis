import os
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class SentimentDictionary:
    def __init__(self):
        self.chinese_positive: Set[str] = set()
        self.chinese_negative: Set[str] = set()
        self.english_positive: Set[str] = set()
        self.english_negative: Set[str] = set()

        self._load_default_dictionaries()

    def _load_default_dictionaries(self):

        self.chinese_positive = {
            "上涨",
            "增长",
            "盈利",
            "收益",
            "看涨",
            "利好",
            "升值",
            "牛市",
            "突破",
            "创新高",
            "强劲",
            "增长",
            "利润",
            "分红",
            "回购",
            "增持",
            "超预期",
            "提升",
            "改善",
            "复苏",
            "反弹",
            "稳定",
            "乐观",
            "积极",
            "涨",
            "升",
            "高",
            "多",
            "利",
            "好",
            "赢",
            "走强",
            "拉升",
            "飙升",
            "涨停",
            "收涨",
            "上扬",
            "攀升",
            "走高",
            "劲涨",
            "大红",
            "大涨",
            "攀升",
            "回暖",
            "回升",
            "向好",
            "景气",
            "繁荣",
            "活跃",
            "坚挺",
            "韧性",
        }

        self.chinese_negative = {
            "下跌",
            "亏损",
            "贬值",
            "看跌",
            "利空",
            "熊市",
            "暴跌",
            "风险",
            "违约",
            "破产",
            "亏损",
            "减持",
            "抛售",
            "压力",
            "衰退",
            "下行",
            "低于预期",
            "恶化",
            "不确定性",
            "波动",
            "恐慌",
            "谨慎",
            "跌",
            "降",
            "低",
            "少",
            "亏",
            "坏",
            "软",
            "弱",
            "走低",
            "下挫",
            "大跌",
            "狂跌",
            "失守",
            "收跌",
            "飘绿",
            "跳水",
            "崩盘",
            "腰斩",
            "踩雷",
            "爆雷",
            "冻结",
            "制裁",
            "警告",
            "威胁",
            "紧张",
            "危机",
            "冲突",
            "战争",
        }

        self.english_positive = {
            "up",
            "rise",
            "gain",
            "profit",
            "earn",
            "bullish",
            "growth",
            "increase",
            "surge",
            "rally",
            "positive",
            "strong",
            "beat",
            "exceed",
            "improve",
            "recovery",
            "rebound",
            "stable",
            "optimistic",
        }

        self.english_negative = {
            "down",
            "fall",
            "loss",
            "lose",
            "bearish",
            "decline",
            "decrease",
            "drop",
            "plunge",
            "negative",
            "weak",
            "miss",
            "below",
            "worsen",
            "risk",
            "uncertainty",
            "volatile",
            "recession",
            "crash",
            "fail",
        }

    def load_custom_dict(
        self, filepath: str, language: str = "zh", polarity: str = "positive"
    ):
        if not os.path.exists(filepath):
            logger.warning(f"Dictionary file not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

        if language == "zh":
            if polarity == "positive":
                self.chinese_positive.update(words)
            else:
                self.chinese_negative.update(words)
        else:
            if polarity == "positive":
                self.english_positive.update(words)
            else:
                self.english_negative.update(words)

        logger.info(f"Loaded {len(words)} {language} {polarity} words from {filepath}")

    def get_sentiment_score(self, text: str, language: str = "zh") -> float:
        if language == "zh":
            chars = list(text)
            pos_count = sum(1 for c in chars if c in self.chinese_positive)
            neg_count = sum(1 for c in chars if c in self.chinese_negative)

            for pos_word in self.chinese_positive:
                if pos_word in text:
                    pos_count += 1
            for neg_word in self.chinese_negative:
                if neg_word in text:
                    neg_count += 1

            total = len(chars)
        else:
            text_lower = text.lower()
            words = text_lower.split()
            pos_count = sum(1 for w in self.english_positive if w in text_lower)
            neg_count = sum(1 for w in self.english_negative if w in text_lower)
            total = len(words)

        if total == 0:
            return 0.0

        score = (pos_count - neg_count) / total
        return score

    def annotate(self, text: str, language: str, threshold: float = 0.05) -> tuple:
        score = self.get_sentiment_score(text, language)

        if score > threshold:
            return 2, abs(score)
        elif score < -threshold:
            return 0, abs(score)
        else:
            return 1, abs(score)
