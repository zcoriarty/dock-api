"""Shared Pydantic models and enums for Dock Property API."""

from enum import Enum
from typing import Optional, List, Any, Dict
from pydantic import BaseModel


# Enums
class ListingType(str, Enum):
    for_sale = "for_sale"
    for_rent = "for_rent"
    sold = "sold"
    pending = "pending"


class SiteSource(str, Enum):
    zillow = "zillow"
    redfin = "redfin"
    realtor = "realtor.com"


# Response Models
class PropertyResponse(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    price: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft: Optional[int] = None
    lot_sqft: Optional[int] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    price_per_sqft: Optional[float] = None
    hoa_fee: Optional[float] = None
    days_on_mls: Optional[int] = None
    list_date: Optional[str] = None
    sold_price: Optional[float] = None
    last_sold_date: Optional[str] = None
    assessed_value: Optional[float] = None
    estimated_value: Optional[float] = None
    annual_taxes: Optional[float] = None
    tax_rate: Optional[float] = None
    tax_history: Optional[List[Dict[str, Any]]] = None
    price_history: Optional[List[Dict[str, Any]]] = None
    mls_id: Optional[str] = None
    listing_url: Optional[str] = None
    primary_photo: Optional[str] = None
    alt_photos: Optional[List[str]] = None
    source: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    agent_name: Optional[str] = None
    broker_name: Optional[str] = None
    stories: Optional[int] = None
    parking_garage: Optional[int] = None


class SearchResponse(BaseModel):
    count: int
    properties: List[PropertyResponse]


class MarketSummaryResponse(BaseModel):
    city: str
    state: str
    average_rent: Optional[float] = None
    median_rent: Optional[float] = None
    new_listings_last_week: int
    sample_size: int
    source: str = "HomeHarvest"


# Investment Search Models
class InvestmentSearchRequest(BaseModel):
    location: str
    listing_type: ListingType = ListingType.for_sale
    site: Optional[SiteSource] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_beds: Optional[int] = None
    max_beds: Optional[int] = None
    min_baths: Optional[int] = None
    min_sqft: Optional[int] = None
    max_sqft: Optional[int] = None
    min_lot_sqft: Optional[int] = None
    max_lot_sqft: Optional[int] = None
    min_year_built: Optional[int] = None
    max_year_built: Optional[int] = None
    max_days_on_market: Optional[int] = None
    min_cap_rate: Optional[float] = None
    min_cash_on_cash: Optional[float] = None
    min_dscr: Optional[float] = None
    target_cap_rate: float = 0.06
    target_cash_on_cash: float = 0.08
    target_dscr: float = 1.25
    interest_rate: float = 0.07
    down_payment_percent: float = 0.25
    closing_cost_percent: float = 0.03
    vacancy_rate: float = 0.05
    management_fee_percent: float = 0.08
    repairs_per_year: float = 1200
    insurance_rate: float = 0.003
    other_expenses_annual: float = 0
    rent_sensitivity: float = 0.0
    limit: int = 50
    past_days: Optional[int] = None


class InvestmentMetricsResponse(BaseModel):
    estimated_rent: Optional[float] = None
    effective_gross_income: Optional[float] = None
    net_operating_income: Optional[float] = None
    cap_rate: Optional[float] = None
    cash_on_cash: Optional[float] = None
    dscr: Optional[float] = None
    annual_cash_flow: Optional[float] = None
    annual_debt_service: Optional[float] = None
    total_cash_required: Optional[float] = None
    score: float


class InvestmentResultResponse(BaseModel):
    id: str
    property: PropertyResponse
    metrics: InvestmentMetricsResponse


class InvestmentSearchResponse(BaseModel):
    count: int
    results: List[InvestmentResultResponse]
    sorted_by: str = "score"
