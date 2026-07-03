from fastapi import APIRouter, HTTPException
from schemas import LocalitiesResponse
from services.rag_service import get_localities
from config import SUPPORTED_CITIES

router = APIRouter(prefix="/localities", tags=["Localities"])


@router.get("/{city}", response_model=LocalitiesResponse)
def localities(city: str):
    city = city.lower().strip()
    if city not in SUPPORTED_CITIES:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not supported. Options: {SUPPORTED_CITIES}"
        )
    return LocalitiesResponse(city=city, localities=get_localities(city))
