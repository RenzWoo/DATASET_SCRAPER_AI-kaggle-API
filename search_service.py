import os
import json
import re
import tempfile
import shutil
import pandas as pd
import numpy as np
import torch
import logging
from functools import lru_cache
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TOP_K_RESULTS = 5
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBI1_q_i9XgsGeu3eaxVnGJ6ZFy4lZObn8")

# Initialize APIs
def get_kaggle_api():
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.error(f"Kaggle Authentication failed: {e}")
        return None

def configure_genai():
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash")
    return None

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_dataset_request(query):
    model = configure_genai()
    if not model:
        return None
        
    prompt = f"""
    You are an AI system for extracting dataset requirements for data analysis.

    Your responsibilities:
    1. **Understand the dataset being requested**, independent of the user’s goal.
    2. **Identify the data context** (e.g., timeseries, geographic, cross-sectional, etc.).
    3. **Extract the following components:**
       - **topic** → a short phrase summarizing the dataset being requested.
       - **sub_topic** → the main subject category (one or two words)
       - **context** → inferred category such as "timeseries", "geographic", etc.
       - **desired_fields** → a list of fields explicitly or implicitly requested
       - **extra_fields** → a list of fields that might be helpful

    Return ONLY valid JSON. No explanation. No markdown.

    User: "{query}"
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        # Find the first { and last } to handle potential text around JSON
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            text = text[start_idx:end_idx]
        return json.loads(text)
    except Exception as e:
        logger.error(f"Error extracting request with LLM: {e}")
        return None

def kaggle_search(topic):
    api = get_kaggle_api()
    if not api:
        return []
        
    try:
        scraped_datasets = api.dataset_list(search=topic)
        # Filter by usability
        scraped_datasets = [data for data in scraped_datasets if getattr(data, 'usabilityRating', 0) >= 0.5]
        
        if not scraped_datasets:
            return []
            
        titles = [data.title for data in scraped_datasets]
        subtitles = [getattr(data, 'subtitle', "") or data.title or "" for data in scraped_datasets]
        
        model = get_embedding_model()
        titles_emb = model.encode(titles, convert_to_tensor=True)
        subtitles_emb = model.encode(subtitles, convert_to_tensor=True)
        topic_emb = model.encode(topic, convert_to_tensor=True)
        
        # Combine title and subtitle scores
        scores = (util.cos_sim(topic_emb, titles_emb)[0] + util.cos_sim(topic_emb, subtitles_emb)[0]) / 2
        
        k = min(TOP_K_RESULTS, len(scores))
        top_indices = torch.topk(scores, k=k).indices.tolist()
        
        results = [{"ref": scraped_datasets[i].ref, "title": scraped_datasets[i].title} for i in top_indices]
        return results
    except Exception as e:
        logger.error(f"Kaggle search error: {e}")
        return []

def download_and_load_dataset(dataset_ref):
    api = get_kaggle_api()
    if not api:
        return None, None
        
    temp_dir = tempfile.mkdtemp()
    try:
        api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
        
        # Load the largest CSV or the one that matches best (simplified for now: first CSV)
        csv_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            shutil.rmtree(temp_dir)
            return None, "No CSV files found in dataset."
            
        # Select the largest CSV file for analysis
        target_csv = max(csv_files, key=os.path.getsize)
        df = pd.read_csv(target_csv, encoding_errors="replace")
        
        # Cleanup temp dir after loading into memory
        shutil.rmtree(temp_dir)
        return df, None
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, str(e)

def normalize(s):
    return re.sub(r'[\s_-]+', '', str(s).lower())

def extract_sample_values(df, top_n=10):
    sample_summary = {}
    for col in df.columns:
        unique_vals = df[col].dropna().unique()[:top_n]
        sample_summary[col] = unique_vals.tolist()
    return sample_summary

def validate_dataset_with_llm(df, query_info, query, dataset_name):
    model = configure_genai()
    if not model:
        return {"relevance_score": 0.5, "reasoning": "LLM validation skipped (no API key).", "use_cases": "N/A"}
        
    sample_values = extract_sample_values(df)
    values_text = "\n".join([f"- {col}: {values}" for col, values in sample_values.items()])
    
    prompt = f"""
    Validate if this dataset matches the user's query.
    
    User Query: {query}
    Topic: {query_info.get('topic')}
    
    Dataset: {dataset_name}
    Columns: {list(df.columns)}
    Sample Values:
    {values_text}
    
    Return ONLY JSON with:
    - relevance_score (0.0 to 1.0)
    - reasoning (brief explanation)
    - use_cases (string)
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            text = text[start_idx:end_idx]
        return json.loads(text)
    except Exception as e:
        logger.error(f"Error validating with LLM: {e}")
        return {"relevance_score": 0.7, "reasoning": f"Automated scoring failed, but dataset seems relevant. ({e})", "use_cases": "General analysis."}
