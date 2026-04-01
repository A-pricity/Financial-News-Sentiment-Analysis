# Financial News Sentiment Analysis

BERT-based model for financial news sentiment analysis (positive/neutral/negative).

## Project Structure

```
financial-sentiment-analysis/
├── data/                    # Data processing
│   └── processed/           # Training data (CSV)
├── models/                  # Model architectures
├── training/                # Training logic
├── scripts/                 # Entry points
├── configs/                 # Configuration
├── utils/                   # Utilities
└── checkpoints/             # Model checkpoints
```

## Quick Start

### 1. Install Dependencies

```bash
cd financial-sentiment-analysis
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

Downloads from HuggingFace: `lwrf42/financial-sentiment-dataset` (95k+ samples)

### 3. Train Model

```bash
python scripts/train.py
```

### 4. Run Inference

```bash
python scripts/demo_inference.py
```

## Data Format

| Column | Description |
|--------|-------------|
| text | Input text |
| language | en/zh |
| label | 0=neutral, 1=positive, 2=negative |
| confidence | annotation confidence |
| source | data source |

## Configuration

Edit `configs/config.yaml`:

```yaml
training:
  epochs: 3
  batch_size: 16
  learning_rate_bert: 2e-5

model:
  english:
    bert_name: "bert-base-uncased"
```

## Features

- **BERT-based Classification**: Fine-tuned BERT for sentiment classification
- **Bilingual Support**: Chinese (bert-base-chinese) and English (bert-base-uncased) models
- **Public Dataset**: Uses professionally annotated financial sentiment data
- **Flexible Architecture**: Support for single and bilingual models