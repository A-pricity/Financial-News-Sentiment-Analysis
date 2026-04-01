# Financial Sentiment Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dual-channel financial news sentiment analysis system (BERT + TextCNN + Attention Fusion) supporting Chinese and English news with 92.5% accuracy target.

**Architecture:** Dual-channel architecture: Chinese channel (bert-base-chinese + TextCNN) and English channel (finbert-tone + TextCNN), with language-agnostic attention fusion layer for sentiment classification (Positive/Negative/Neutral).

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, BERT, TextCNN

---

## Phase 1: Project Setup

### Task 1: Initialize Project Environment

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `README.md`

- [ ] **Step 1: Write requirements.txt**
- [ ] **Step 2: Write config.yaml**
- [ ] **Step 3: Write README.md**
- [ ] **Step 4: Commit**

---

## Phase 2: Data Crawler

### Task 2: Base Crawler Framework

**Files:**
- Create: `data/crawler/__init__.py`
- Create: `data/crawler/base_crawler.py`

### Task 3: Domestic Crawlers

**Files:**
- Create: `data/crawler/domestic_crawlers.py`

### Task 4: International Crawlers

**Files:**
- Create: `data/crawler/international_crawlers.py`

### Task 5: Crawler Scheduler

**Files:**
- Create: `data/crawler/scheduler.py`

---

## Phase 3: Data Preprocessing

### Task 6: Utility Functions

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/language_detector.py`
- Create: `utils/text_cleaner.py`
- Create: `utils/sentiment_dict.py`

### Task 7: Dataset Processing

**Files:**
- Create: `data/dataset.py`

---

## Phase 4: Model Implementation

### Task 8: BERT Encoder

**Files:**
- Create: `models/__init__.py`
- Create: `models/bert_encoder.py`

### Task 9: TextCNN

**Files:**
- Create: `models/textcnn.py`

### Task 10: Fusion Model

**Files:**
- Create: `models/fusion_model.py`

### Task 11: Sentiment Classifier

**Files:**
- Create: `models/sentiment_classifier.py`

---

## Phase 5: Training

### Task 12: Training Pipeline

**Files:**
- Create: `training/__init__.py`
- Create: `training/trainer.py`

### Task 13: Training Entry Point

**Files:**
- Create: `scripts/train.py`

---

## Phase 6: Inference

### Task 14: Inference Entry Point

**Files:**
- Create: `scripts/inference.py`

---

## Phase 7: Crawl Script

### Task 15: Crawl Entry Point

**Files:**
- Create: `scripts/crawl_data.py`

---

## Implementation Plan Complete

**15 tasks total**, covering:
- Project setup
- Data crawling (7 sources)
- Data preprocessing
- Model (BERT + TextCNN + Fusion)
- Training pipeline
- Inference

**Two execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks
2. **Inline Execution** - Execute tasks in this session using executing-plans

Which approach?
