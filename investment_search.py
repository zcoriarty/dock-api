"""Investment search endpoints and scoring utilities."""

from typing import Optional, List, Dict
import logging
import math

from fastapi import APIRouter

from schemas import (
    InvestmentSearchRequest,
    InvestmentSearchResponse,
    InvestmentResultResponse,
    InvestmentMetricsResponse,
)
from property_utils import (
    safe_scrape,
    raw_property_to_response,
    safe_float,
    safe_int,
    get_first,
    median,
)


logger = logging.getLogger(__name__)
router = APIRouter()


def extract_rent_comps(location: str, limit: int = 200) -> List[Dict[str, Optional[float]]]:
    params = {
        "location": location,
        "listing_type": "for_rent",
        "limit": limit,
    }
    raw_rents = safe_scrape(params) or []
    comps: List[Dict[str, Optional[float]]] = []
    for prop in raw_rents:
        rent = safe_float(
            get_first(
                prop,
                "list_price",
                "price",
                "rent",
                ("listing", "price"),
                ("listing", "list_price"),
            )
        )
        if not rent:
            continue
        beds = safe_int(
            get_first(
                prop,
                ("description", "beds"),
                "beds",
                ("property", "beds"),
            )
        )
        sqft = safe_int(
            get_first(
                prop,
                ("description", "sqft"),
                "sqft",
                ("property", "sqft"),
                ("building", "size", "living"),
            )
        )
        comps.append({"rent": rent, "beds": beds, "sqft": sqft})
    return comps


def estimate_monthly_rent(prop, comps: List[Dict[str, Optional[float]]], fallback_rate: float) -> Optional[float]:
    if not comps:
        if prop.price:
            return prop.price * fallback_rate
        return None

    filtered = comps
    if prop.bedrooms:
        bed_filtered = [c for c in filtered if c.get("beds") and abs(c["beds"] - prop.bedrooms) <= 1]
        if bed_filtered:
            filtered = bed_filtered

    if prop.sqft:
        sqft_comps = [c for c in filtered if c.get("sqft")]
        if sqft_comps:
            lower = prop.sqft * 0.6
            upper = prop.sqft * 1.4
            sqft_filtered = [c for c in sqft_comps if lower <= c["sqft"] <= upper]
            filtered = sqft_filtered or sqft_comps

    rent_values = [c["rent"] for c in filtered if c.get("rent")]
    rent_median = median(rent_values)
    if rent_median:
        return rent_median

    if prop.sqft:
        per_sqft = [c["rent"] / c["sqft"] for c in comps if c.get("rent") and c.get("sqft")]
        per_sqft_median = median(per_sqft)
        if per_sqft_median:
            return per_sqft_median * prop.sqft

    if prop.price:
        return prop.price * fallback_rate

    return None


def estimate_taxes(prop, default_tax_rate: float) -> float:
    if prop.annual_taxes:
        return prop.annual_taxes
    if prop.tax_rate and prop.price:
        return prop.price * prop.tax_rate
    if prop.assessed_value and prop.tax_rate:
        return prop.assessed_value * prop.tax_rate
    if prop.price:
        return prop.price * default_tax_rate
    return 0


def mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0 or years <= 0:
        return 0
    monthly_rate = annual_rate / 12
    total_payments = years * 12
    if monthly_rate <= 0:
        return principal / total_payments
    factor = math.pow(1 + monthly_rate, total_payments)
    return principal * (monthly_rate * factor) / (factor - 1)


def ratio_score(value: Optional[float], target: float) -> float:
    if value is None or target <= 0:
        return 0
    ratio = value / target
    return min(max(ratio, 0), 1.5) / 1.5


def compute_score(prop, metrics: InvestmentMetricsResponse, request: InvestmentSearchRequest) -> float:
    cap_score = ratio_score(metrics.cap_rate, request.target_cap_rate)
    coc_score = ratio_score(metrics.cash_on_cash, request.target_cash_on_cash)
    dscr_score = ratio_score(metrics.dscr, request.target_dscr)

    price_score = 1.0
    if request.max_price and prop.price:
        price_score = min(request.max_price / prop.price, 1.0)

    size_score = 1.0
    if request.min_sqft and prop.sqft:
        size_score = min(prop.sqft / request.min_sqft, 1.0)
    if request.max_sqft and prop.sqft:
        size_score = min(size_score, min(request.max_sqft / prop.sqft, 1.0))

    bed_score = 1.0
    if request.min_beds and prop.bedrooms:
        bed_score = min(prop.bedrooms / request.min_beds, 1.0)

    weights = {
        "cap": 0.3,
        "coc": 0.25,
        "dscr": 0.2,
        "price": 0.15,
        "size": 0.05,
        "beds": 0.05,
    }

    composite = (
        cap_score * weights["cap"]
        + coc_score * weights["coc"]
        + dscr_score * weights["dscr"]
        + price_score * weights["price"]
        + size_score * weights["size"]
        + bed_score * weights["beds"]
    )

    return round(composite * 100, 1)


def build_result_id(prop) -> str:
    if prop.mls_id:
        return prop.mls_id
    if prop.listing_url:
        return prop.listing_url
    address = prop.address or ""
    zip_code = prop.zip_code or ""
    return f"{address}-{zip_code}".strip("-") or "unknown"


def apply_property_filters(prop, request: InvestmentSearchRequest) -> bool:
    if request.min_price and (not prop.price or prop.price < request.min_price):
        return False
    if request.max_price and (not prop.price or prop.price > request.max_price):
        return False
    if request.min_beds and (not prop.bedrooms or prop.bedrooms < request.min_beds):
        return False
    if request.max_beds and prop.bedrooms and prop.bedrooms > request.max_beds:
        return False
    if request.min_baths and (not prop.bathrooms or prop.bathrooms < request.min_baths):
        return False
    if request.min_sqft and (not prop.sqft or prop.sqft < request.min_sqft):
        return False
    if request.max_sqft and prop.sqft and prop.sqft > request.max_sqft:
        return False
    if request.min_lot_sqft and (not prop.lot_sqft or prop.lot_sqft < request.min_lot_sqft):
        return False
    if request.max_lot_sqft and prop.lot_sqft and prop.lot_sqft > request.max_lot_sqft:
        return False
    if request.min_year_built and (not prop.year_built or prop.year_built < request.min_year_built):
        return False
    if request.max_year_built and prop.year_built and prop.year_built > request.max_year_built:
        return False
    if request.max_days_on_market and prop.days_on_mls and prop.days_on_mls > request.max_days_on_market:
        return False
    return True


def compute_metrics(prop, request: InvestmentSearchRequest, rent_comps: List[Dict[str, Optional[float]]]) -> Optional[InvestmentMetricsResponse]:
    if not prop.price or prop.price <= 0:
        return None

    fallback_rate = 0.008 + request.rent_sensitivity
    monthly_rent = estimate_monthly_rent(prop, rent_comps, fallback_rate)
    if not monthly_rent:
        return None

    annual_gross = monthly_rent * 12
    effective_gross = annual_gross * (1 - request.vacancy_rate)
    taxes = estimate_taxes(prop, default_tax_rate=0.012)
    insurance = prop.price * request.insurance_rate if prop.price else 0
    management = effective_gross * request.management_fee_percent
    repairs = request.repairs_per_year
    other_expenses = request.other_expenses_annual

    total_operating = taxes + insurance + management + repairs + other_expenses
    noi = effective_gross - total_operating

    loan_amount = prop.price * (1 - request.down_payment_percent)
    annual_debt_service = mortgage_payment(loan_amount, request.interest_rate, 30) * 12

    total_cash_required = prop.price * request.down_payment_percent + prop.price * request.closing_cost_percent
    annual_cash_flow = noi - annual_debt_service

    cap_rate = noi / prop.price if prop.price else None
    cash_on_cash = annual_cash_flow / total_cash_required if total_cash_required > 0 else None
    dscr = noi / annual_debt_service if annual_debt_service > 0 else None

    metrics = InvestmentMetricsResponse(
        estimated_rent=monthly_rent,
        effective_gross_income=effective_gross,
        net_operating_income=noi,
        cap_rate=cap_rate,
        cash_on_cash=cash_on_cash,
        dscr=dscr,
        annual_cash_flow=annual_cash_flow,
        annual_debt_service=annual_debt_service,
        total_cash_required=total_cash_required,
        score=0,
    )

    metrics.score = compute_score(prop, metrics, request)
    return metrics


@router.post("/investment-search", response_model=InvestmentSearchResponse)
async def investment_search(request: InvestmentSearchRequest):
    logger.info("Investment search: location=%s", request.location)

    params = {
        "location": request.location,
        "listing_type": request.listing_type.value,
        "limit": request.limit,
    }
    if request.site:
        params["site_name"] = request.site.value
    if request.past_days:
        params["past_days"] = request.past_days
    if request.min_price:
        params["price_min"] = request.min_price
    if request.max_price:
        params["price_max"] = request.max_price
    if request.min_beds:
        params["beds_min"] = request.min_beds
    if request.max_beds:
        params["beds_max"] = request.max_beds
    if request.min_baths:
        params["baths_min"] = request.min_baths
    if request.min_sqft:
        params["sqft_min"] = request.min_sqft
    if request.max_sqft:
        params["sqft_max"] = request.max_sqft

    raw_properties = safe_scrape(params)
    if not raw_properties:
        return InvestmentSearchResponse(count=0, results=[])

    rent_comps = extract_rent_comps(request.location, limit=min(200, request.limit * 4))

    results: List[InvestmentResultResponse] = []
    for raw_prop in raw_properties:
        prop = raw_property_to_response(raw_prop)
        if not apply_property_filters(prop, request):
            continue
        metrics = compute_metrics(prop, request, rent_comps)
        if not metrics:
            continue
        if request.min_cap_rate and (not metrics.cap_rate or metrics.cap_rate < request.min_cap_rate):
            continue
        if request.min_cash_on_cash and (not metrics.cash_on_cash or metrics.cash_on_cash < request.min_cash_on_cash):
            continue
        if request.min_dscr and (not metrics.dscr or metrics.dscr < request.min_dscr):
            continue
        results.append(
            InvestmentResultResponse(
                id=build_result_id(prop),
                property=prop,
                metrics=metrics,
            )
        )

    results_sorted = sorted(results, key=lambda item: item.metrics.score, reverse=True)
    return InvestmentSearchResponse(count=len(results_sorted), results=results_sorted)
