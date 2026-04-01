# Financial News Sentiment Analysis - Design Specification

## 1. Project Overview

### 1.1 Background

Financial news data exhibits significant unstructured characteristics. Using natural language processing (NLP) technology to extract market sentiment can assist quantitative trading decisions.

### 1.2 Objective

Build a sentiment analysis system for financial news (Positive/Negative/Neutral), supporting both Chinese and English news sources through a dual-channel architecture.

### 1.3 Target Metrics

- Accuracy: ≥ 92.5%
- F1-Score: ≥ 0.91
- Support for 7 data sources (domestic + international)
- Total dataset: 100,000 news articles

---

## 2. Technical Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Collection Layer                        │
│  Domestic: 东方财富 | 新浪财经 | 凤凰网                              │
│  International: Reuters | Bloomberg | CNBC | Yahoo Finance          │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Preprocessing Layer                      │
│  Language Detection → Deduplication → Cleaning → Annotation       │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Dual-Channel Model                           │
│  ┌─────────────────────┐       ┌─────────────────────┐            │
│  │    Chinese Channel  │       │    English Channel  │            │
│  │                     │       │                     │            │
│  │  bert-base-chinese  │       │  finbert-tone       │            │
│  │  + TextCNN (zh)     │       │  + TextCNN (en)     │            │
│  └──────────┬──────────┘       └──────────┬──────────┘            │
│             │                               │                        │
│             └───────────────┬───────────────┘                        │
│                             ▼                                        │
│              ┌─────────────────────────┐                            │
│              │   Attention Fusion      │                            │
│              │ (Language-Agnostic)     │                            │
│              └────────────┬────────────┘                            │
│                             ▼                                        │
│                   Sentiment Classification                          │
│                   (Positive/Negative/Neutral)                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Training & Optimization                      │
│  Gradient Accumulation → AdamW → Linear LR Decay → Early Stopping │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Collection**: Crawl news from 7 sources with anti-blocking strategy
2. **Preprocessing**: Language detection → Clean → Annotate with sentiment dictionaries
3. **Encoding**: Dual-channel BERT+TextCNN encoding
4. **Fusion**: Attention-weighted fusion of dual-channel outputs
5. **Classification**: 3-class sentiment prediction
6. **Training**: Optimize with gradient accumulation and learning rate scheduling

---

## 3. Data Collection

### 3.1 Data Sources

| Category | Source | Language | Focus |
|----------|--------|----------|-------|
| Domestic | 东方财富 | Chinese | A-share market |
| | 新浪财经 | Chinese | Broad coverage |
| | 凤凰网 | Chinese | Financial channel |
| International | Reuters | English | Professional finance |
| | Bloomberg | English | Professional finance |
| | CNBC | English | Financial TV |
| | Yahoo Finance | English | Comprehensive finance |

### 3.2 Crawling Strategy

- **Batch size**: 10-20 articles per request
- **Interval**: Random delay 3-8 seconds between requests
- **Anti-blocking**:
  - Rotate User-Agent headers
  - Use proxy pool (if available)
  - Exponential backoff on 429 errors
- **Total target**: 100,000 articles (mixed Chinese/English)
- **Incremental crawling**: Support resume from last checkpoint

### 3.3 Data Fields

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| title | str | News title |
| content | str | News content |
| source | str | Data source |
| publish_time | datetime | Publication time |
| url | str | Original URL |
| language | str | Detected language (zh/en) |
| sentiment | int | Label: 0=negative, 1=neutral, 2=positive |

---

## 4. Data Preprocessing

### 4.1 Language Detection

- Use `langdetect` library
- Confidence threshold: 0.9 (discard low-confidence results)
- Fallback: Use first 100 characters for quick detection

### 4.2 Deduplication

- Exact match on title + content hash
- Similarity-based deduplication using TF-IDF (threshold: 0.85)

### 4.3 Text Cleaning

- Remove HTML tags
- Remove URLs and email addresses
- Remove special characters (keep Chinese/English/alphanumeric)
- Normalize whitespace
- Truncate to max 512 tokens (BERT limit)

### 4.4 Annotation Strategy

**Rule-based annotation using sentiment dictionaries:**

| Language | Dictionary Source |
|----------|-------------------|
| Chinese | Financial sentiment lexicon + SnowNLP |
| English | Loughran-McDonald Finance Dictionary + VADER |

**Annotation Rules:**

1. Count positive/negative words from dictionary
2. Calculate sentiment score:
   - `score = (positive_count - negative_count) / total_words`
3. Label assignment:
   - score > 0.05 → Positive (2)
   - score < -0.05 → Negative (0)
   - otherwise → Neutral (1)

**Confidence filtering:**

- High confidence: |score| > 0.2 → Keep (auto-label)
- Medium confidence: 0.1 < |score| ≤ 0.2 → Keep (flag for review)
- Low confidence: |score| ≤ 0.1 → Discard or manual annotation

---

## 5. Model Design

### 5.1 Chinese Channel

**BERT Encoder:**

- Model: `bert-base-chinese`
- Output: [CLS] token embedding (768-dim)

**TextCNN (Chinese):**

- Embedding: 768-dim (from BERT or standalone)
- Filter sizes: [2, 3, 4] (character-level n-grams for Chinese)
- Number of filters: 256 per filter size
- Pooling: Max pooling over time
- Output: 768-dim (256 × 3)

### 5.2 English Channel

**BERT Encoder:**

- Model: `yiyanghkust/finbert-tone`
- Pre-trained on financial domain
- Output: [CLS] token embedding (768-dim)

**TextCNN (English):**

- Filter sizes: [2, 3, 4, 5] (word-level n-grams)
- Number of filters: 256 per filter size
- Pooling: Max pooling
- Output: 1024-dim (256 × 4)

### 5.3 Attention Fusion Layer

```
Input:
  - BERT CLS: h_bert (768-dim)
  - TextCNN: h_cnn (768-dim for Chinese, 1024-dim for English)

Projection:
  - h_bert_proj = Linear(h_bert, hidden_dim)
  - h_cnn_proj = Linear(h_cnn, hidden_dim)
  - concatenated = Concat([h_bert_proj, h_cnn_proj])

Attention weights:
  - attention_score = Sigmoid(Linear(concatenated, 1))
  - α = attention_score (scalar between 0 and 1)

Fusion:
  - h_fused = α * h_bert_proj + (1 - α) * h_cnn_proj
  - h_fused = LayerNorm(h_fused)

Output:
  - Final hidden state (768-dim)
```

### 5.4 Classification Head

```
h_fused → Dropout(0.3) → Linear(768, 256) → ReLU → Dropout(0.2) → Linear(256, 3)
```

### 5.5 Loss Function

- Cross-Entropy Loss with class weights (handle imbalance)
- Label smoothing: 0.1

---

## 6. Training Strategy

### 6.1 Optimizer

- **Optimizer**: AdamW
- **Learning rate**: 2e-5 (BERT), 1e-3 (TextCNN, fusion layer)
- **Weight decay**: 0.01
- **Beta**: (0.9, 0.999)

### 6.2 Learning Rate Scheduler

- **Type**: Linear warmup + linear decay
- **Warmup steps**: 500
- **Total steps**: Calculate based on dataset size and batch size

### 6.3 Gradient Accumulation

- **Effective batch size**: 64
- **Micro batch size**: 16
- **Accumulation steps**: 4

### 6.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Early stopping patience | 3 |
| Max sequence length | 512 |
| Train/Val/Test split | 8:1:1 |
| Random seed | 42 |

### 6.5 Mixed Precision

- Use FP16 (PyTorch AMP) for memory efficiency

---

## 7. Evaluation

### 7.1 Metrics

- Accuracy
- Precision (per class + macro)
- Recall (per class + macro)
- F1-Score (per class + macro)
- Confusion Matrix

### 7.2 Evaluation Protocol

- Evaluate on validation set every 500 steps
- Evaluate on test set after training
- Save best model based on macro F1-score

---

## 8. Project Structure

```
financial-sentiment-analysis/
├── data/
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── base_crawler.py      # Base crawler class
│   │   ├── domestic_crawlers.py # Domestic news crawlers
│   │   ├── international_crawlers.py # International crawlers
│   │   └── scheduler.py         # Crawling scheduler
│   ├── dataset.py               # PyTorch Dataset
│   └── processed/               # Processed data storage
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── __init__.py
│   ├── bert_encoder.py          # BERT encoder wrapper
│   ├── textcnn.py               # TextCNN implementation
│   ├── fusion_model.py          # Dual-channel fusion model
│   └── sentiment_classifier.py  # Full classifier
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop
│   ├── optimizer.py             # Optimizer config
│   └── scheduler.py             # LR scheduler
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py             # Evaluation metrics
│   └── predictor.py             # Inference
├── scripts/
│   ├── crawl_data.py            # Main crawling script
│   ├── preprocess.py            # Data preprocessing
│   ├── train.py                 # Training entry point
│   └── inference.py             # Inference entry point
├── configs/
│   └── config.yaml              # Configuration file
├── utils/
│   ├── __init__.py
│   ├── sentiment_dict.py        # Sentiment dictionaries
│   ├── language_detector.py     # Language detection
│   └── text_cleaner.py          # Text cleaning
├── requirements.txt             # Dependencies
└── README.md                     # Documentation
```

---

## 9. Implementation Plan

### Phase 1: Data Collection (Week 1)

- [ ] Set up crawler framework
- [ ] Implement domestic crawlers (3 sources)
- [ ] Implement international crawlers (4 sources)
- [ ] Add anti-blocking mechanisms
- [ ] Test crawling on small scale
- [ ] Scale to 100,000 articles

### Phase 2: Data Preprocessing (Week 2)

- [ ] Implement language detection
- [ ] Implement text cleaning pipeline
- [ ] Build sentiment annotation system
- [ ] Apply sentiment dictionaries
- [ ] Create train/val/test splits
- [ ] Validate data quality

### Phase 3: Model Implementation (Week 3)

- [ ] Implement BERT encoder
- [ ] Implement TextCNN
- [ ] Implement attention fusion layer
- [ ] Build complete classifier
- [ ] Verify model forward pass

### Phase 4: Training (Week 4)

- [ ] Set up training pipeline
- [ ] Configure optimizer and scheduler
- [ ] Implement gradient accumulation
- [ ] Add mixed precision training
- [ ] Implement early stopping
- [ ] Train model and validate

### Phase 5: Evaluation & Inference (Week 5)

- [ ] Evaluate on test set
- [ ] Generate evaluation report
- [ ] Implement inference script
- [ ] Test inference pipeline
- [ ] Document usage

---

## 10. Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
langdetect>=1.0.9
beautifulsoup4>=4.12.0
requests>=2.31.0
lxml>=4.9.0
tqdm>=4.65.0
pyyaml>=6.0
scikit-learn>=1.3.0
vaderSentiment>=3.3.2
snownlp>=0.12.3
pandas>=2.0.0
numpy>=1.24.0
```

---

## 11. Notes

1. **Language-specific processing**: Chinese uses character-level n-grams, English uses word-level n-grams
2. **Model size**: Consider model compression (knowledge distillation) for deployment if needed
3. **Error handling**: Implement robust error handling for crawler failures
4. **Logging**: Comprehensive logging for debugging and monitoring
5. **Reproducibility**: Set random seeds throughout the pipeline
