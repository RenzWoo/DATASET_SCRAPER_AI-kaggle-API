"""
Dataset Finder — AI-powered Kaggle dataset discovery pipeline
=============================================================
Flow:
  1. extract_query()       — LLM parses natural-language query → structured schema
  2. search_kaggle()       — Kaggle API search + MiniLM semantic re-ranking
  3. download_datasets()   — downloads only top candidates into a temp dir
  4. load_csvs()           — loads CSVs into dataframes
  5. validate_datasets()   — field-match scoring + LLM value-level validation
  6. find_datasets()       — orchestrates the full pipeline, returns ranked results
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

# ── third-party ─────────────────────────────────────────────────────────────
import google.generativeai as genai
import pandas as pd
import torch
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv() 

# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

class Config:
    # Kaggle
    USABILITY_THRESHOLD: float = 0.5
    TOP_K_RESULTS: int = 10
    DOWNLOAD_TOP_N: int = 5          # only download the best N candidates

    # Scoring
    FIELD_MATCH_THRESHOLD: float = 0.4   # min keyword score to pass pre-filter
    SAMPLE_VALUES_N: int = 10

    # Models
    SEMANTIC_MODEL: str = "all-MiniLM-L6-v2"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Secrets — never hardcode; read from environment
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Singletons  (initialised once at module load)
# ════════════════════════════════════════════════════════════════════════════

def _init_kaggle() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    log.info("Kaggle authenticated ✅")
    return api


def _init_gemini() -> genai.GenerativeModel:
    if not Config.GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Export it before running: export GOOGLE_API_KEY=<your-key>"
        )
    genai.configure(api_key=Config.GOOGLE_API_KEY)
    return genai.GenerativeModel(Config.GEMINI_MODEL)


_kaggle_api: KaggleApi = _init_kaggle()
_embed_model: SentenceTransformer = SentenceTransformer(Config.SEMANTIC_MODEL)
_gemini: genai.GenerativeModel = _init_gemini()


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Query extraction
# ════════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """
You are an AI system for extracting dataset requirements for data analysis.

Responsibilities:
1. Understand the dataset being requested, independent of the user's analytical goal.
2. Identify the data context: timeseries, geographic, cross-sectional, panel,
   list-based, event-log, or hierarchical.
3. Use the user's goal only to infer fields — do NOT include it in topic/sub_topic.
4. Extract:
   - topic         → short phrase summarising the dataset
   - sub_topic     → main subject category (1-2 words)
   - context       → one of the data-context categories above
   - desired_fields → fields explicitly or implicitly requested; infer if missing
   - extra_fields  → additional helpful fields beyond desired_fields

Return ONLY valid JSON. No markdown, no explanation.

### Examples

User: "Give me population data for China from past to present to predict future growth."
Output:
{{
  "topic": "Historical population of China",
  "sub_topic": "population",
  "context": "timeseries",
  "desired_fields": ["year", "population"],
  "extra_fields": ["province", "gdp", "fertility_rate"]
}}

User: "All volcanoes in the world with name, country, lat, lon, eruptions, casualties."
Output:
{{
  "topic": "World volcanoes",
  "sub_topic": "volcanoes",
  "context": "geographic",
  "desired_fields": ["name", "country", "latitude", "longitude", "eruptions", "casualties"],
  "extra_fields": ["type", "VEI", "elevation"]
}}

User: "{query}"
"""

_REQUIRED_KEYS = {"topic", "sub_topic", "context", "desired_fields"}


def extract_query(query: str) -> dict[str, Any]:
    """Parse a natural-language query into a structured dataset spec via Gemini."""
    prompt = _EXTRACTION_PROMPT.format(query=query)
    response = _gemini.generate_content(prompt)
    text = response.text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON:\n{text}") from exc

    missing = _REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}\nGot: {result}")

    # Guarantee extra_fields exists
    result.setdefault("extra_fields", [])
    log.info("Query extracted → topic: %r | context: %r", result["topic"], result["context"])
    return result


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Kaggle search + semantic re-ranking
# ════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=Config.TOP_K_RESULTS)
def search_kaggle(topic: str) -> list[str]:
    """
    Search Kaggle for datasets matching *topic*, filter by usability,
    re-rank with MiniLM cosine similarity, return top-K dataset refs.
    """
    try:
        candidates = [
            d for d in _kaggle_api.dataset_list(search=topic)
            if (d.usability_rating or 0) >= Config.USABILITY_THRESHOLD
        ]
    except Exception as exc:
        log.error("Kaggle search failed: %s", exc)
        return []

    if not candidates:
        log.warning("No Kaggle results for %r", topic)
        return []

    titles    = [d.title or "" for d in candidates]
    subtitles = [d.subtitle or d.title or "" for d in candidates]

    topic_emb    = _embed_model.encode(topic,     convert_to_tensor=True)
    titles_emb   = _embed_model.encode(titles,    convert_to_tensor=True)
    subtitles_emb = _embed_model.encode(subtitles, convert_to_tensor=True)

    scores = (
        util.cos_sim(topic_emb, titles_emb)[0]
        + util.cos_sim(topic_emb, subtitles_emb)[0]
    ) / 2

    k = min(Config.TOP_K_RESULTS, len(scores))
    top_indices = torch.topk(scores, k=k).indices.tolist()
    refs = [candidates[i].ref for i in top_indices]

    log.info("Found %d datasets for %r", len(refs), topic)
    return refs


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Download
# ════════════════════════════════════════════════════════════════════════════

@contextmanager
def download_datasets(refs: list[str], top_n: int = Config.DOWNLOAD_TOP_N):
    """
    Context manager: downloads the first *top_n* dataset refs into a temp
    directory, yields (temp_dir_path), cleans up on exit.

    Usage:
        with download_datasets(refs) as path:
            dfs = load_csvs(path)
    """
    tmp = tempfile.TemporaryDirectory()
    try:
        for ref in refs[:top_n]:
            log.info("Downloading %s …", ref)
            try:
                _kaggle_api.dataset_download_files(ref, path=tmp.name, unzip=True)
            except Exception as exc:
                log.warning("Skipping %s: %s", ref, exc)
        yield tmp.name
    finally:
        tmp.cleanup()
        log.info("Temp directory cleaned up.")


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Load CSVs
# ════════════════════════════════════════════════════════════════════════════

def load_csvs(folder: str) -> dict[str, pd.DataFrame]:
    """Recursively load all CSV files in *folder* → {filename: DataFrame}."""
    dfs: dict[str, pd.DataFrame] = {}
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(root, fname)
            try:
                dfs[fname] = pd.read_csv(path, encoding_errors="replace")
            except Exception as exc:
                log.warning("Could not read %s: %s", fname, exc)

    log.info("Loaded %d CSV files.", len(dfs))
    return dfs


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Validation
# ════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def _field_match_score(
    df: pd.DataFrame,
    desired: list[str],
    extra: list[str],
) -> tuple[float, float]:
    """
    Returns (desired_score, extra_score) in [0, 1].

    desired_score = fraction of desired_fields matched in df columns.
    extra_score   = fraction of extra_fields matched in df columns.
    Scored separately so they can be combined with different weights.
    """
    col_norms = {_normalize(c) for c in df.columns}

    def _hits(fields: list[str]) -> int:
        return sum(
            any(_normalize(f) in cn for cn in col_norms)
            for f in fields
        )

    d_score = _hits(desired) / len(desired) if desired else 0.0
    e_score = _hits(extra)   / len(extra)   if extra   else 0.0
    return min(d_score, 1.0), min(e_score, 1.0)


def _sample_values(df: pd.DataFrame, n: int = Config.SAMPLE_VALUES_N) -> str:
    lines = []
    for col in df.columns:
        unique = df[col].dropna().unique()[:n].tolist()
        lines.append(f"  {col}: {unique}")
    return "\n".join(lines)


_VALIDATION_PROMPT = """
You are validating whether a CSV dataset matches a user's data request.

User request:
  Topic    : {topic}
  Sub-topic: {sub_topic}
  Context  : {context}
  Original query: "{query}"

Dataset: {name}
Columns: {columns}
Keyword match score (desired fields): {keyword_score:.0%}

Sample values (up to {n} unique per column):
{sample_values}

Tasks:
1. Assess whether the sample VALUES match the query context.
   - Timeseries → proper date/time values present?
   - Geographic → location names, coordinates?
   - Entity list → expected entity names?
2. Produce an independent relevance_score in [0.0, 1.0].
3. Write a one-sentence reasoning citing specific observed values.
4. Generate a concrete use_case for this dataset (do NOT paraphrase the query;
   infer from actual column names and sample values).

Return ONLY valid JSON, no markdown:
{{
  "relevance_score": 0.85,
  "reasoning": "...",
  "use_case": "..."
}}
"""


def _llm_validate(
    name: str,
    df: pd.DataFrame,
    query_info: dict,
    query: str,
    keyword_score: float,
) -> dict[str, Any] | None:
    prompt = _VALIDATION_PROMPT.format(
        topic=query_info["topic"],
        sub_topic=query_info["sub_topic"],
        context=query_info["context"],
        query=query,
        name=name,
        columns=list(df.columns),
        keyword_score=keyword_score,
        n=Config.SAMPLE_VALUES_N,
        sample_values=_sample_values(df),
    )
    try:
        response = _gemini.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        log.warning("LLM validation failed for %s: %s", name, exc)
        return None


def validate_datasets(
    dataframes: dict[str, pd.DataFrame],
    query_info: dict,
    query: str,
) -> list[dict[str, Any]]:
    """
    For each dataframe:
      1. Compute keyword field-match score.
      2. Discard if below FIELD_MATCH_THRESHOLD.
      3. Run LLM value-level validation.
      4. Return results sorted by LLM relevance score (descending).
    """
    desired = query_info["desired_fields"]
    extra   = query_info.get("extra_fields", [])
    results = []

    for name, df in dataframes.items():
        d_score, e_score = _field_match_score(df, desired, extra)
        combined_keyword = 0.7 * d_score + 0.3 * e_score

        if combined_keyword < Config.FIELD_MATCH_THRESHOLD:
            log.debug("Pre-filter skip %s (keyword=%.0%)", name, combined_keyword)
            continue

        log.info("Validating %s (keyword=%.0%%)", name, combined_keyword)
        llm = _llm_validate(name, df, query_info, query, combined_keyword)
        if llm is None:
            continue

        results.append({
            "name":            name,
            "keyword_score":   round(combined_keyword, 3),
            "relevance_score": round(llm.get("relevance_score", 0.0), 3),
            "reasoning":       llm.get("reasoning", ""),
            "use_case":        llm.get("use_case", ""),
            "dataframe":       df,
        })

    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    log.info("Validated %d datasets.", len(results))
    return results


# ════════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ════════════════════════════════════════════════════════════════════════════

def find_datasets(query: str) -> list[dict[str, Any]]:
    """
    Full pipeline: natural-language query → ranked list of matching datasets.

    Each result dict contains:
      name, keyword_score, relevance_score, reasoning, use_case, dataframe
    """
    # 1. Extract structured schema
    query_info = extract_query(query)

    # 2. Search Kaggle
    search_term = f"{query_info['topic']} {query_info['sub_topic']}"
    refs = search_kaggle(search_term)
    if not refs:
        log.warning("No Kaggle datasets found.")
        return []

    # 3. Download top candidates + 4. Load + 5. Validate
    with download_datasets(refs, top_n=Config.DOWNLOAD_TOP_N) as folder:
        dataframes = load_csvs(folder)
        if not dataframes:
            log.warning("No CSV files found in downloaded datasets.")
            return []
        results = validate_datasets(dataframes, query_info, query)

    return results


# ════════════════════════════════════════════════════════════════════════════
# CLI / notebook quick test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    TEST_QUERY = "I need a dataset of population in China by province"

    print(f"\n{'='*60}")
    print(f"Query: {TEST_QUERY}")
    print(f"{'='*60}\n")

    datasets = find_datasets(TEST_QUERY)

    if not datasets:
        print("No matching datasets found.")
    else:
        for rank, d in enumerate(datasets, 1):
            print(f"#{rank}  {d['name']}")
            print(f"    Relevance : {d['relevance_score']:.0%}  |  Keyword: {d['keyword_score']:.0%}")
            print(f"    Reasoning : {d['reasoning']}")
            print(f"    Use case  : {d['use_case']}")
            print(f"    Shape     : {d['dataframe'].shape}")
            print()
