from fastapi import APIRouter, HTTPException
from schemas import CompareRequest, CompareResponse, PredictResponse
from services.ml_service import predict_price_per_sqft
from config import GROWTH_RATES, DATASET_AGE_YEARS

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.post("", response_model=CompareResponse)
def compare(req: CompareRequest):
    results: list[PredictResponse] = []

    for city in req.cities:
        features = {
            "bhk":    req.bhk,
            "area":   req.area,
            "type":   req.type,
            "region": req.region,
            "status": req.status,
            "age":    req.age,
        }
        try:
            raw_pps = predict_price_per_sqft(city, features)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for '{city}': {str(e)}"
            )

        growth_rate       = GROWTH_RATES.get(city, 0.08)
        current_pps       = raw_pps * ((1 + growth_rate) ** DATASET_AGE_YEARS)
        current_total_inr = round(current_pps * req.area, 2)

        results.append(PredictResponse(
            city                    = city,
            raw_price_per_sqft      = round(raw_pps, 2),
            raw_total_price_lac     = round((raw_pps * req.area) / 1_00_000, 2),
            current_price_per_sqft  = round(current_pps, 2),
            current_total_price_inr = current_total_inr,
            current_total_price_lac = round(current_total_inr / 1_00_000, 2),
            area_sqft               = req.area,
            growth_rate_used        = growth_rate,
            inflation_years_applied = DATASET_AGE_YEARS,
        ))

    sorted_results   = sorted(results, key=lambda r: r.current_total_price_inr)
    cheapest         = sorted_results[0]
    most_expensive   = sorted_results[-1]
    difference_lac   = round(
        (most_expensive.current_total_price_inr - cheapest.current_total_price_inr) / 1_00_000, 2
    )

    return CompareResponse(
        results             = results,
        cheapest_city       = cheapest.city,
        most_expensive_city = most_expensive.city,
        difference_lac      = difference_lac,
    )
