from pydantic import BaseModel, Field, field_validator
from typing import Optional
from config import SUPPORTED_CITIES


class PredictRequest(BaseModel):
    city:    str   = Field(..., example="mumbai")
    bhk:     int   = Field(..., ge=1, le=10, example=2)
    area:    float = Field(..., gt=0, example=1000)
    type:    str   = Field(..., example="Apartment")
    region:  str   = Field(..., example="Andheri West")
    status:  str   = Field(..., example="Ready to Move")
    age:     str   = Field(..., example="0-5 years")

    @field_validator("city")
    @classmethod
    def city_must_be_supported(cls, v):
        v = v.lower().strip()
        if v not in SUPPORTED_CITIES:
            raise ValueError(f"City '{v}' not supported. Choose from: {SUPPORTED_CITIES}")
        return v


class PredictResponse(BaseModel):
    city:                    str
    raw_price_per_sqft:      float
    raw_total_price_lac:     float
    current_price_per_sqft:  float
    current_total_price_inr: float
    current_total_price_lac: float
    area_sqft:               float
    growth_rate_used:        float
    inflation_years_applied: int


class FutureRequest(PredictRequest):
    years: int = Field(..., ge=1, le=20, example=3)


class FutureResponse(PredictResponse):
    projection_years:      int
    future_price_per_sqft: float
    future_price_inr:      float
    future_price_lac:      float


class CompareRequest(BaseModel):
    cities:  list[str] = Field(..., min_length=2, max_length=3, example=["mumbai", "pune"])
    bhk:     int       = Field(..., ge=1, le=10, example=2)
    area:    float     = Field(..., gt=0, example=1000)
    type:    str       = Field(..., example="Apartment")
    region:  str       = Field(..., example="")
    status:  str       = Field(..., example="Ready to Move")
    age:     str       = Field(..., example="0-5 years")

    @field_validator("cities", mode="before")
    @classmethod
    def cities_must_be_supported(cls, v):
        result = []
        for city in v:
            city = city.lower().strip()
            if city not in SUPPORTED_CITIES:
                raise ValueError(f"City '{city}' not supported.")
            result.append(city)
        return result


class CompareResponse(BaseModel):
    results:             list[PredictResponse]
    cheapest_city:       str
    most_expensive_city: str
    difference_lac:      float


class LocalitiesResponse(BaseModel):
    city:       str
    localities: list[str]


class ChatRequest(BaseModel):
    query: str = Field(..., example="What is the average price of 2BHK in Andheri?")
    city:  str = Field("mumbai", example="mumbai")   # active city from map

    @field_validator("city")
    @classmethod
    def city_must_be_valid(cls, v):
        v = v.lower().strip()
        if v not in SUPPORTED_CITIES:
            return "mumbai"  
        return v


class ChatResponse(BaseModel):
    answer:  str
    sources: list[str] = []
