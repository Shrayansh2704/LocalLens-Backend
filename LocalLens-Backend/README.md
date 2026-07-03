# LocalLens Backend

> **Real estate price intelligence powered by machine learning and RAG**

LocalLens is a full-stack platform that brings transparency to Indian real estate pricing. This repository contains the **backend** — a FastAPI server that serves ML-based price predictions, inflation-adjusted forecasts, and a conversational RAG chatbot grounded in real property data.

---

## What This Does

LocalLens solves three real problems in the Indian housing market:

1. **Price opacity** — No one knows what a property is *actually* worth. Brokers give estimates, listings are outdated, and buyers are flying blind.
2. **Dataset staleness** — Public real estate data is very old. We apply rigorous inflation correction to make predictions relevant to today.
3. **Inaccessible insights** — Most people can't interpret "₹22,000/sqft." Our RAG chatbot answers questions in plain English, grounded in real data.

### What the backend provides:

-  **ML price prediction** — Random Forest and XGBoost models trained per city
-  **Inflation adjustment** — Compound growth for present market prices
-  **Future projection** — Project prices relevant to 3-5 years forward using verified growth rates
-  **RAG chatbot** — FAISS vector retrieval + Groq LLM for conversational queries
-  **Locality search** — Fetch all valid localities per city from cleaned datasets
-  **City comparison** — Side-by-side price rankings with automatic cheapest detection

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │ ML Service   │   │ RAG Service  │   │  Routers     │     │
│  │              │   │              │   │              │     │
│  │ • Load .pkl  │   │ • FAISS idx  │   │ • /predict   │     │
│  │ • Predict    │   │ • Embeddings │   │ • /future    │     │
│  │ • Inflate    │   │ • Groq API   │   │ • /chat      │     │
│  └──────────────┘   └──────────────┘   │ • /compare   │     │
│                                        │ • /localities│     │
│  ┌──────────────┐   ┌──────────────┐   └──────────────┘     │
│  │ models/      │   │ rag_dfs/     │                        │
│  │ •saved models│   │ •saved CSVs  │                        │
│  └──────────────┘   └──────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Next.js Frontend    │
              └───────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- A Groq API key (free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/locallens-backend.git
cd locallens-backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

> **Get a Groq API key:** Sign up at https://console.groq.com → API Keys → Create new key

### Run the server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

**Interactive docs:** http://127.0.0.1:8000/docs

---

## Model Files

The trained ML models (`.pkl` files) are **not included** in this repository due to GitHub's 100MB file size limit.

(You can directly reach out to the author for the same)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/cities` | Returns list of supported cities |
| `POST` | `/predict` | Get current price (2025 inflation-adjusted) |
| `POST` | `/future` | Get current + future price projection |
| `POST` | `/compare` | Compare prices across 2-3 cities |
| `GET` | `/localities/{city}` | Get all localities for a city |
| `POST` | `/chat` | Ask the RAG chatbot a question |

### Example: Price Prediction

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "mumbai",
    "bhk": 2,
    "area": 1000,
    "type": "Apartment",
    "region": "Andheri West",
    "status": "Ready to Move",
    "age": "0-5 years"
  }'
```

**Response:**
```json
{
  "city": "mumbai",
  "predicted_price_per_sqft": 22771.45,
  "total_price_inr": 22771450,
  "total_price_lac": 227.71,
  "current_total_price_inr": 28696912,
  "current_total_price_lac": 286.97
}
```

### Example: Chatbot Query

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the average price of a 2 BHK in Andheri?",
    "city": "mumbai"
  }'
```

**Response:**
```json
{
  "answer": "Based on the data, 2 BHK apartments in Andheri West average around ₹22,771 per sqft. For a 1000 sqft property, that comes to approximately ₹2.28 crores. Specific societies like MS H2O in Santacruz East and Pioneer Heritage Residency 2 in Santacruz West fall in this range.",
  "sources": [
    "City: Mumbai. Locality: Andheri. BHK: 2...",
    "City: Mumbai. Locality: Santacruz East. BHK: 2..."
  ]
}
```

---

## How It Works

### 1. **Data Preprocessing**

Three city datasets (Mumbai, Bengaluru, Pune) are cleaned and standardized:
- Price conversion: Lakhs/Crores → INR
- Area conversion: Sq. Metres/Yards → sqft
- Outlier removal: Domain-specific thresholds
- Target variable: `price_per_sqft = price_in_inr / area`

Output: `rag_dfs/city_cleaned_dataset.csv`

### 2. **Model Training**

Three algorithms tested per city:
- Linear Regression (baseline)
- Random Forest
- XGBoost

**Selection criteria:** R² + MAE. When R² is similar, MAE breaks the tie.

**Results:**

| City | Model | R² | MAE (₹/sqft) |
|------|-------|-----|--------------|
| Mumbai | Random Forest | 0.83 | 2000 |
| Bengaluru | XGBoost | 0.59 | 1576 |
| Pune | Random Forest | 0.28 | 1976 |

Models saved as: `models/city_price_model.pkl`

### 3. **Inflation Adjustment**
Predictions are compounded forward using verified growth rates:

| City | Annual Growth Rate |
|------|--------------------|
| Mumbai | 8% |
| Bengaluru | 12% |
| Pune | 6% |

Formula:
```
P_current = P_model × (1 + r)^3        (2022 → 2025)
P_future  = P_current × (1 + r)^n      (2025 → 2025+n years)
```

### 4. **RAG Pipeline**

**Embedding:**
- Each CSV row → sentence: "City: Mumbai. Locality: Andheri. BHK: 2..."
- Sentence → 384-dim vector using `all-MiniLM-L6-v2`
- Vectors stored in FAISS IndexFlatIP (cosine similarity)

**Query flow:**
1. User query → embedded using same model
2. FAISS retrieves top-5 similar property records
3. City detection: cross-city keywords → search all cities, else search active city only
4. Retrieved rows + ML context → system prompt
5. Groq API (`llama-3.3-70b-versatile`) generates grounded answer
6. Return answer + sources

**Key feature:** FAISS indexes are cached to disk. First build takes 3-5 minutes, subsequent startups < 5 seconds.

---

## Project Structure

```
locallens-backend/
├── main.py                  # FastAPI app, startup logic, CORS
├── config.py                # Paths, cities, growth rates, LLM settings
├── schemas.py               # Pydantic request/response models
├── requirements.txt         # Python dependencies
├── .env                     # GROQ_API_KEY (not committed)
│
├── services/
│   ├── ml_service.py        # Load models, predict, inflate
│   └── rag_service.py       # FAISS indexing, Groq API, retrieval
│
├── routers/
│   ├── predict.py           # POST /predict
│   ├── future.py            # POST /future
│   ├── compare.py           # POST /compare
│   ├── localities.py        # GET /localities/{city}
│   └── chat.py              # POST /chat
│
├── models/                  # .pkl model files (trained separately)
│   ├── mumbai_price_model.pkl
│   ├── bengaluru_price_model.pkl
│   └── pune_price_model.pkl
│
├── rag_dfs/                 # Cleaned datasets for RAG
│   ├── mumbai_cleaned_dataset.csv
│   ├── bengaluru_cleaned_dataset.csv
│   └── pune_cleaned_dataset.csv
│
├── faiss_indexes/           # Cached FAISS indexes (auto-generated)
│   ├── mumbai_index.faiss
│   ├── mumbai_texts.pkl
│   └── ...
│
└── model_files/             # Training scripts (reference only)
    ├── mum.py
    ├── beng.py
    └── pun.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI, Uvicorn |
| **ML** | scikit-learn, XGBoost, joblib |
| **RAG** | FAISS (faiss-cpu), sentence-transformers |
| **LLM** | Groq API (llama-3.3-70b-versatile) |
| **Validation** | Pydantic |
| **Environment** | python-dotenv |
| **HTTP Client** | requests |

**Why these choices?**

- **FastAPI:** Auto-generates OpenAPI docs, native async, Pydantic validation out of the box
- **XGBoost/Random Forest:** Best performers on tabular real estate data (proven by benchmarks)
- **FAISS:** Industry standard for billion-scale vector search, runs locally
- **Groq:** Free tier, low latency, strong instruction-following with llama-3.3-70b
- **sentence-transformers:** Pre-trained semantic embeddings, no training required

---

## Development

### Add a New City

1. **Prepare dataset:** Clean CSV with columns matching the schema in `config.py`
2. **Train model:** Run training script (see `model_files/mum.py` as template)
3. **Save model:** `joblib.dump(pipeline, "models/newcity_price_model.pkl")`
4. **Update config:** Add city to `CITIES` dict in `config.py` with growth rate
5. **Add cleaned CSV:** Save to `rag_dfs/newcity_cleaned_dataset.csv`
6. **Restart server:** FAISS index auto-builds on first startup

### Run Tests

```bash
# Test all endpoints via Swagger UI
open http://127.0.0.1:8000/docs

# Or use curl/Postman with example requests above
```

### Debug Mode

```bash
# Enable auto-reload on code changes
uvicorn main:app --reload --log-level debug
```

---

## Performance

| Operation | Latency |
|-----------|---------|
| `/predict` (ML inference) | < 100ms |
| `/future` | < 100ms |
| `/compare` (2 cities) | < 200ms |
| `/localities/{city}` | < 50ms |
| `/chat` (FAISS + Groq) | 5-8 seconds |
| FAISS index build (first startup) | 3-5 minutes |
| FAISS index load (cached) | < 5 seconds |

**Model Accuracy:**

- **Mumbai:** R² = 0.83, MAE = ₹2,000/sqft (±₹20L on 1000 sqft)
- **Bengaluru:** R² = 0.59, MAE = ₹1,576/sqft
- **Pune:** R² = 0.28, MAE = ₹1,976/sqft (dataset is inherently noisy)

---

## Known Issues & Limitations

1. **Dataset age:** Currently using old data with inflation correction. Future version will integrate live listings.
2. **Locality encoding:** Excluded from model features to prevent overfitting on smaller datasets. This limits price variation capture within cities.
3. **Groq free tier:** Rate limits and occasional latency spikes. Consider upgrading for production.
4. **FAISS disk cache:** Not thread-safe during rebuild. Don't run multiple server instances during first startup.

---

## Acknowledgments

- **Open-source communities:** FastAPI, scikit-learn, XGBoost, FAISS, sentence-transformers, Hugging Face
- **Data sources:** Public real estate listing datasets
- **Market research:** Anarock Property Consultants, Knight Frank India
- **LLM API:** Groq for providing free-tier access to llama-3.3-70b-versatile

---

🔗 **Frontend work:** : 

**Built at LNMIIT Jaipur**
