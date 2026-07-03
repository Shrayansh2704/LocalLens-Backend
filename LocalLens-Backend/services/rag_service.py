"""
rag_service.py
Full Phase 2 RAG pipeline with disk caching:
  - FAISS indexes saved to faiss_indexes/ after first build
  - Subsequent startups load from disk instantly
  - Groq llama-3.3-70b for LLM answers
  - City-aware by default, cross-city when query demands it
  - ML predictions injected into context when relevant
"""

import os
import re
import numpy as np
import pandas as pd
import faiss 
import requests
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import (RAG_DATA_PATHS, SUPPORTED_CITIES, GROQ_API_KEY,
                    LLM_MODEL, GROWTH_RATES, DATASET_AGE_YEARS)

# Constants
TOP_K        = 8 # i'll try 9-12 vals further
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
INDEX_DIR    = Path("faiss_indexes")   # saved to LocalLens/faiss_indexes/

# Global stores
_dfs:      dict[str, pd.DataFrame]       = {}
_indexes:  dict[str, faiss.IndexFlatIP]  = {}
_texts:    dict[str, list[str]]          = {}
_embedder: SentenceTransformer | None    = None

# Cross-city detection
CROSS_CITY_TRIGGERS = [
    "vs", "versus", "compare", "better", "cheaper", "expensive",
    "difference", "which city", "both", "all cities",
    "mumbai and", "pune and", "bengaluru and", "bangalore and",
]
CITY_ALIASES = {
    "bangalore": "bengaluru",
    "bengalore": "bengaluru",
    "bombay":    "mumbai",
}


# Disk cache helpers

def _index_path(city: str) -> Path:
    return INDEX_DIR / f"{city}.index"

def _texts_path(city: str) -> Path:
    return INDEX_DIR / f"{city}_texts.pkl"

def _cache_exists(city: str) -> bool:
    return _index_path(city).exists() and _texts_path(city).exists()

def _save_cache(city: str, index, texts: list[str]):
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(_index_path(city)))
    with open(_texts_path(city), "wb") as f:
        pickle.dump(texts, f)
    print(f"[RAG] Cache saved for '{city}' ✓")

def _load_cache(city: str):
    index = faiss.read_index(str(_index_path(city)))
    with open(_texts_path(city), "rb") as f:
        texts = pickle.load(f)
    return index, texts

def _row_to_text(city: str, row) -> str:
    return (
        f"City: {city.title()}. "
        f"Locality: {row.get('locality', '')}. "
        f"Region: {row.get('region', '')}. "
        f"Type: {row.get('type', '')}. "
        f"BHK: {row.get('bhk', '')}. "
        f"Area: {row.get('area', '')} sqft. "
        f"Price per sqft: ₹{row.get('price_per_sqft', 0):.0f}. "
        f"Status: {row.get('status', '')}. "
        f"Age: {row.get('age', '')}."
    )


def load_all_rag_data():
    global _embedder

    print("[RAG] Loading sentence-transformer model...")
    _embedder = SentenceTransformer(EMBED_MODEL)
    print(f"[RAG] Embedder ready ✓")

    for city in SUPPORTED_CITIES:
        path = RAG_DATA_PATHS[city]
        if not path.exists():
            print(f"[RAG] WARNING — dataset missing for '{city}'")
            continue

        df = pd.read_csv(path).dropna(subset=["locality", "price_per_sqft"])
        _dfs[city] = df

        if _cache_exists(city):
            print(f"[RAG] Loading cached FAISS index for '{city}'...")
            index, texts = _load_cache(city)
            _indexes[city] = index
            _texts[city]   = texts
            print(f"[RAG] '{city}' loaded from cache — {index.ntotal} vectors ✓")

        else:
            print(f"[RAG] Building FAISS index for '{city}' ({len(df)} rows)...")
            texts = [_row_to_text(city, row) for _, row in df.iterrows()]
            _texts[city] = texts

            embeddings = _embedder.encode(
                texts, show_progress_bar=True, batch_size=256
            )
            embeddings = np.array(embeddings).astype("float32")
            faiss.normalize_L2(embeddings)

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            _indexes[city] = index

            _save_cache(city, index, texts)
            print(f"[RAG] '{city}' index built — {index.ntotal} vectors ✓")

    print("[RAG] All indexes ready ✅")



def get_localities(city: str) -> list[str]:
    if city not in _dfs or "locality" not in _dfs[city].columns:
        return []
    return sorted(_dfs[city]["locality"].dropna().unique().tolist())


def _detect_cities_in_query(query: str) -> list[str]:
    q = query.lower()
    found = []
    for city in SUPPORTED_CITIES:
        if city in q:
            found.append(city)
    for alias, canonical in CITY_ALIASES.items():
        if alias in q and canonical not in found:
            found.append(canonical)
    return found

def _is_cross_city_query(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in CROSS_CITY_TRIGGERS) or len(_detect_cities_in_query(query)) >= 2

def _resolve_cities(query: str, active_city: str) -> list[str]:
    if _is_cross_city_query(query):
        mentioned = _detect_cities_in_query(query)
        return mentioned if mentioned else SUPPORTED_CITIES
    return [active_city]


# FAISS retrieval

def _retrieve(query: str, cities: list[str]) -> list[str]:
    if _embedder is None:
        return []
    q_emb = _embedder.encode([query], normalize_embeddings=True).astype("float32")
    all_results = []
    for city in cities:
        if city not in _indexes:
            continue
        scores, indices = _indexes[city].search(q_emb, TOP_K)
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                all_results.append((float(score), _texts[city][idx]))
    all_results.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in all_results[:TOP_K]]


def _get_ml_context(query: str, cities: list[str]) -> str:
    bhk_match  = re.search(r'(\d)\s*bhk', query.lower())
    area_match = re.search(r'(\d+)\s*(?:sqft|sq\.?ft|square)', query.lower())
    if not bhk_match:
        return ""

    bhk   = int(bhk_match.group(1))
    lines = [f"\n[ML price data for {bhk}BHK]"]

    for city in cities:
        if city not in _dfs:
            continue
        filtered = _dfs[city][_dfs[city]["bhk"] == bhk]
        if filtered.empty:
            continue
        avg_pps     = filtered["price_per_sqft"].mean()
        growth      = GROWTH_RATES.get(city, 0.08)
        current_pps = avg_pps * ((1 + growth) ** DATASET_AGE_YEARS)

        if area_match:
            area          = int(area_match.group(1))
            current_total = current_pps * area
            future_3yr    = current_total * ((1 + growth) ** 3)
            lines.append(
                f"  {city.title()}: ₹{current_pps:,.0f}/sqft (2025). "
                f"{area} sqft → ₹{current_total/1e5:,.1f}L today, "
                f"₹{future_3yr/1e5:,.1f}L in 3 years."
            )
        else:
            lines.append(
                f"  {city.title()}: avg ₹{current_pps:,.0f}/sqft (2025)."
            )
    return "\n".join(lines) if len(lines) > 1 else ""


def _call_groq(system_prompt: str, user_message: str) -> str:
    if not GROQ_API_KEY:
        return "Groq API key not set. Add GROQ_API_KEY to your .env file."

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.4,
                "max_tokens":  1024,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Groq API error: {e.response.status_code} — {e.response.text}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def _build_system_prompt(active_city: str, retrieved: list[str], ml_context: str) -> str:
    return f"""You are LocalLens, a friendly and expert Indian real estate assistant for Mumbai, Bengaluru, and Pune.

The user is currently viewing: {active_city.title()}

RETRIEVED PROPERTY DATA (real listings from dataset):
{chr(10).join(retrieved)}
{ml_context}

INSTRUCTIONS:
- Answer conversationally and helpfully for ANY question — real estate or general
- For real estate questions, base answers on the retrieved data above
- Always quote prices in Indian format: Lakhs (L) or Crores (Cr)
- Mention specific localities and price ranges from the data when relevant
- For cross-city queries, clearly state which city is cheaper and by how much
- Growth rates: Mumbai 8%, Bengaluru 12%, Pune 6% per year
- All prices are inflation-adjusted to 2025 from 2022 dataset
- If asked something outside real estate, answer helpfully as a general assistant
- Keep answers concise: 3-6 sentences for simple queries, more for comparisons
- Never make up locality names or prices not present in the data
"""


def ask(query: str, active_city: str = "mumbai") -> dict:
    print(f"[DEBUG] Received active_city = '{active_city}'")  # ← add this line only
    active_city = CITY_ALIASES.get(active_city.lower().strip(), active_city.lower().strip())
    if active_city not in SUPPORTED_CITIES:
        active_city = "mumbai"

    cities    = _resolve_cities(query, active_city)
    retrieved = _retrieve(query, cities)
    ml_ctx    = _get_ml_context(query, cities)
    system    = _build_system_prompt(active_city, retrieved, ml_ctx)
    answer    = _call_groq(system, query)

    return {
        "answer":  answer,
        "sources": [f"LocalLens dataset — {c.title()}" for c in cities],
    }

