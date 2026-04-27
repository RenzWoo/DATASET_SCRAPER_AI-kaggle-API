# Kaggle Dataset Finder

> An AI-powered pipeline that takes a natural-language query and returns a ranked list of relevant Kaggle datasets — no manual searching required.

**Language:** Python 3.11 &nbsp;|&nbsp; **Models:** Gemini 2.5 Flash + MiniLM-L6-v2

---

## Pipeline Overview

```
Natural-language query
        │
        ▼
┌─────────────────────┐
│  1. Query Extraction │  Gemini parses query → structured spec
│     (Gemini LLM)    │  (topic, context, desired fields, extra fields)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Kaggle Search   │  Kaggle API fetches top-K datasets
│     + Filtering     │  filtered by usability rating ≥ 0.5
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Semantic        │  MiniLM-L6-v2 re-ranks results by
│     Re-ranking      │  cosine similarity to query embedding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Download Top-N  │  Downloads best 5 candidates into
│     Candidates      │  a managed temp directory (auto-cleaned)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. LLM Validation  │  Gemini inspects actual CSV columns +
│     + Scoring       │  sample values → relevance score + reasoning
└────────┬────────────┘
         │
         ▼
   Ranked results
   (name, relevance_score, keyword_score, reasoning, use_case, dataframe)
```

---

## Example

```python
results = find_datasets("I need a dataset of population in China by province")

# Output:
# #1  china-population-by-province
#     Relevance : 92%  |  Keyword: 80%
#     Reasoning : Dataset contains province, year, and population columns
#                 with values matching Chinese administrative regions.
#     Use case  : Time-series analysis of provincial population trends
#     Shape     : (340, 8)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/RenzWoo/kaggle-dataset-finder.git
cd kaggle-dataset-finder
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

You need two API credentials:

**Google Gemini API key**
```bash
export GOOGLE_API_KEY=your_gemini_api_key
```
Get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)

**Kaggle API credentials**
```bash
# Place kaggle.json in ~/.kaggle/kaggle.json
# Download it from: kaggle.com → Account → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run

Open `dataset_finder.ipynb` in Jupyter and run all cells. Edit the query in the last cell:

```python
query = "I need a dataset of population in China by province"
results = find_datasets(query)
```

> ⚠️ **Note:** Cell 3 will show an `OSError` if `GOOGLE_API_KEY` is not set — this is expected. Set the env var before running.

---

## Requirements

```
google-generativeai
kaggle
sentence-transformers
torch
pandas
```

> GPU is detected automatically (`cuda:0` if available). CPU works fine for MiniLM inference.

---

## Configuration

All tunable parameters are in the `Config` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USABILITY_THRESHOLD` | 0.5 | Minimum Kaggle usability score to consider a dataset |
| `TOP_K_RESULTS` | 10 | Number of Kaggle search results to fetch |
| `DOWNLOAD_TOP_N` | 5 | How many candidates to actually download for validation |
| `FIELD_MATCH_THRESHOLD` | 0.4 | Minimum keyword match score before LLM validation |
| `SAMPLE_VALUES_N` | 10 | Unique sample values per column sent to LLM |
| `SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for re-ranking |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for extraction and validation |

---

## Design Decisions

- **Two-stage LLM calls** (extraction + validation) keep prompts focused and reduce hallucination risk vs. a single end-to-end prompt.
- **Keyword pre-filter before LLM validation** saves API calls — datasets that don't match on field names are skipped before hitting Gemini.
- **Temp directory context manager** ensures downloaded datasets are always cleaned up, even on failure.
- **`lru_cache` on embedding model** prevents reloading the 80MB MiniLM weights on repeated calls.
- **Scoring weights**: desired fields weighted 0.7, extra fields 0.3 — prioritizes exact field matches over bonus fields.

