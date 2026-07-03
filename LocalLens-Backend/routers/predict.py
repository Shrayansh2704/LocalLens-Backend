from fastapi import APIRouter, HTTPException
from schemas import PredictRequest, PredictResponse
from services.ml_service import predict_price_per_sqft
from config import GROWTH_RATES, DATASET_AGE_YEARS

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = {
        "bhk":    req.bhk,
        "area":   req.area,
        "type":   req.type,
        "region": req.region,
        "status": req.status,
        "age":    req.age,
    }
    try:
        raw_pps = predict_price_per_sqft(req.city, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    growth_rate = GROWTH_RATES.get(req.city, 0.08)

    
    current_pps   = raw_pps * ((1 + growth_rate) ** DATASET_AGE_YEARS)

    raw_total_lac     = round((raw_pps * req.area) / 1_00_000, 2)
    current_total_inr = round(current_pps * req.area, 2)
    current_total_lac = round(current_total_inr / 1_00_000, 2)

    return PredictResponse(
        city                    = req.city,
        raw_price_per_sqft      = round(raw_pps, 2),
        raw_total_price_lac     = raw_total_lac,
        current_price_per_sqft  = round(current_pps, 2),
        current_total_price_inr = current_total_inr,
        current_total_price_lac = current_total_lac,
        area_sqft               = req.area,
        growth_rate_used        = growth_rate,
        inflation_years_applied = DATASET_AGE_YEARS,
    )
