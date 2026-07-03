from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.ml_service import load_all_models
from services.rag_service import load_all_rag_data
from routers import predict, future, compare, localities, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LocalLens starting up...")
    load_all_models()       # joblib loads all .pkl files
    load_all_rag_data()     # loads all cleaned CSVs for locality lookup
    print("Ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title       = "LocalLens API",
    description = "Housing price prediction for Mumbai, Bengaluru & Pune",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(predict.router)
app.include_router(future.router)
app.include_router(compare.router)
app.include_router(localities.router)
app.include_router(chat.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "project": "LocalLens", "version": "1.0.0"}


@app.get("/cities", tags=["Health"])
def cities():
    from config import SUPPORTED_CITIES
    return {"cities": SUPPORTED_CITIES}
