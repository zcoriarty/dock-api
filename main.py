"""
Dock Property API - FastAPI backend using HomeHarvest
Provides property data from Zillow, Redfin, and Realtor.com
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Any, Dict
from datetime import datetime, timedelta
import logging

from investment_search import router as investment_search_router
from schemas import ListingType, SiteSource, PropertyResponse, SearchResponse, MarketSummaryResponse
from property_utils import (
    safe_float,
    get_first,
    median,
    average,
    raw_property_to_response,
    safe_scrape,
    extract_photo_url,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dock Property API",
    description="Property data API powered by HomeHarvest",
    version="1.0.0"
)

# Simple in-memory cache for market summaries to avoid frequent scraping.
MARKET_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}
MARKET_SUMMARY_CACHE_TTL = timedelta(hours=12)
# CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(investment_search_router)




@app.get("/")
async def root():
    return {
        "service": "Dock Property API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/search": "Search properties by location",
            "/market-summary": "City-level rent summary for tracked markets",
            "/property": "Get property by address",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/search", response_model=SearchResponse)
async def search_properties(
    location: str = Query(..., description="City, state or ZIP code (e.g., 'Austin, TX' or '78701')"),
    listing_type: ListingType = Query(ListingType.for_sale, description="Type of listing"),
    site: Optional[SiteSource] = Query(None, description="Source site (zillow, redfin, realtor.com)"),
    min_price: Optional[int] = Query(None, description="Minimum price"),
    max_price: Optional[int] = Query(None, description="Maximum price"),
    min_beds: Optional[int] = Query(None, description="Minimum bedrooms"),
    max_beds: Optional[int] = Query(None, description="Maximum bedrooms"),
    min_baths: Optional[int] = Query(None, description="Minimum bathrooms"),
    min_sqft: Optional[int] = Query(None, description="Minimum square feet"),
    max_sqft: Optional[int] = Query(None, description="Maximum square feet"),
    past_days: Optional[int] = Query(None, description="Filter to listings from past N days"),
    limit: int = Query(50, description="Maximum results to return", le=500),
):
    """
    Search for properties by location with optional filters.
    """
    logger.info(f"Search request: location={location}, listing_type={listing_type.value}")
    
    # Build scrape parameters
    params = {
        "location": location,
        "listing_type": listing_type.value,
        "limit": limit,
    }
    
    if site:
        params["site_name"] = site.value
    
    if past_days:
        params["past_days"] = past_days
        
    # Add price filters
    if min_price:
        params["price_min"] = min_price
    if max_price:
        params["price_max"] = max_price
        
    # Add bed filters
    if min_beds:
        params["beds_min"] = min_beds
    if max_beds:
        params["beds_max"] = max_beds
        
    # Add bath filter
    if min_baths:
        params["baths_min"] = min_baths
        
    # Add sqft filters
    if min_sqft:
        params["sqft_min"] = min_sqft
    if max_sqft:
        params["sqft_max"] = max_sqft
    
    # Scrape properties with error handling (now returns list of dicts)
    raw_properties = safe_scrape(params)
    
    if not raw_properties:
        logger.info(f"No properties found for {location}")
        return SearchResponse(count=0, properties=[])
    
    # Convert to response
    properties = [raw_property_to_response(prop) for prop in raw_properties[:limit]]
    
    logger.info(f"Found {len(properties)} properties for {location}")
    return SearchResponse(count=len(properties), properties=properties)


@app.get("/market-summary", response_model=MarketSummaryResponse)
async def market_summary(
    city: str = Query(..., description="City name (e.g., 'Austin')"),
    state: str = Query(..., description="State code (e.g., 'TX')"),
    limit: int = Query(200, description="Maximum results to analyze", le=500),
):
    """
    Get city-level rent summary from HomeHarvest.
    """
    location = f"{city}, {state}"
    cache_key = f"{city.strip().lower()}|{state.strip().lower()}|{limit}"
    cached = MARKET_SUMMARY_CACHE.get(cache_key)
    now = datetime.utcnow()
    if cached and now - cached["timestamp"] < MARKET_SUMMARY_CACHE_TTL:
        return cached["data"]
    logger.info(f"Market summary request: location={location}")
    
    rent_params = {
        "location": location,
        "listing_type": "for_rent",
        "limit": limit,
    }
    
    raw_rent_listings = safe_scrape(rent_params)
    
    rent_prices: List[float] = []
    if raw_rent_listings:
        for prop in raw_rent_listings:
            price = safe_float(
                get_first(
                    prop,
                    "list_price",
                    "price",
                    "rent",
                    ("listing", "price"),
                    ("listing", "list_price"),
                )
            )
            if price:
                rent_prices.append(price)
    
    recent_params = dict(rent_params)
    recent_params["past_days"] = 7
    recent_listings = safe_scrape(recent_params) or []
    
    response = MarketSummaryResponse(
        city=city,
        state=state,
        average_rent=average(rent_prices),
        median_rent=median(rent_prices),
        new_listings_last_week=len(recent_listings),
        sample_size=len(rent_prices),
        source="HomeHarvest",
    )
    
    MARKET_SUMMARY_CACHE[cache_key] = {"timestamp": now, "data": response}
    return response


@app.get("/property", response_model=PropertyResponse)
async def get_property_by_address(
    address: str = Query(..., description="Full address (e.g., '123 Main St, Austin, TX 78701')"),
    site: Optional[SiteSource] = Query(None, description="Source site preference"),
):
    """
    Get a specific property by its address.
    """
    logger.info(f"=== PROPERTY REQUEST START ===")
    logger.info(f"Property request: address={address}")
    
    params = {
        "location": address,
        "listing_type": "for_sale",
        "limit": 1,
    }
    
    if site:
        params["site_name"] = site.value
    
    raw_properties = safe_scrape(params)
    
    if not raw_properties:
        # Try sold listings
        logger.info(f"No for_sale listings, trying sold for {address}")
        params["listing_type"] = "sold"
        raw_properties = safe_scrape(params)
        
    if not raw_properties:
        logger.info(f"Property not found: {address}")
        raise HTTPException(status_code=404, detail="Property not found")
    
    prop = raw_properties[0]
    # Log the raw property data to debug field names
    logger.info(f"Found property: {address}")
    logger.info(f"Raw property keys: {list(prop.keys())}")
    
    # Debug nested structures
    location = prop.get('location', {})
    details = prop.get('description', {})  # Note: 'description' key holds property details
    logger.info(f"Location keys: {list(location.keys()) if isinstance(location, dict) else 'not a dict'}")
    logger.info(f"Description/details keys: {list(details.keys()) if isinstance(details, dict) else 'not a dict'}")
    logger.info(f"Location content: {location}")
    logger.info(f"Description content: {details}")
    
    logger.info(f"Raw flat data - street: {prop.get('street')}, beds: {prop.get('beds')}, sqft: {prop.get('sqft')}, list_price: {prop.get('list_price')}")
    
    raw_photo = prop.get('primary_photo')
    extracted_url = extract_photo_url(raw_photo)
    logger.info(f"Raw primary_photo: {type(raw_photo).__name__} -> extracted URL: {extracted_url[:80] if extracted_url else 'None'}")
    
    # Convert and log the result
    result = raw_property_to_response(prop)
    logger.info(f"Parsed result - address: {result.address}, beds: {result.bedrooms}, baths: {result.bathrooms}, sqft: {result.sqft}")
    logger.info(f"=== PROPERTY REQUEST END ===")
    return result


@app.get("/property/url", response_model=PropertyResponse)
async def get_property_by_url(
    url: str = Query(..., description="Property URL from Zillow, Redfin, or Realtor.com"),
):
    """
    Get property details from a listing URL.
    This extracts the address from the URL and searches for it.
    """
    logger.info(f"URL request: url={url}")
    
    # Determine source from URL
    site = None
    if "zillow.com" in url.lower():
        site = SiteSource.zillow
    elif "redfin.com" in url.lower():
        site = SiteSource.redfin
    elif "realtor.com" in url.lower():
        site = SiteSource.realtor
        
    # Try to extract address from URL path
    import re
    from urllib.parse import urlparse, unquote
    
    parsed = urlparse(url)
    path = unquote(parsed.path)
    
    # Try various extraction patterns
    address_parts = []
    
    # Zillow pattern
    zillow_match = re.search(r'/homedetails/([^/]+)/', path)
    if zillow_match:
        addr = zillow_match.group(1).replace('-', ' ')
        # Remove zpid suffix if present
        addr = re.sub(r'\s+\d+$', '', addr)
        address_parts = [addr]
        
    # Redfin pattern
    if not address_parts:
        redfin_match = re.search(r'/([A-Z]{2})/([^/]+)/(\d{5})/([^/]+)', path)
        if redfin_match:
            state, city, zipcode, street = redfin_match.groups()
            address_parts = [f"{street.replace('-', ' ')}, {city}, {state} {zipcode}"]
            
    # Realtor pattern
    if not address_parts:
        realtor_match = re.search(r'/([^_]+)_([^_]+)_([A-Z]{2})_(\d{5})', path)
        if realtor_match:
            street, city, state, zipcode = realtor_match.groups()
            address_parts = [f"{street.replace('-', ' ')}, {city}, {state} {zipcode}"]
    
    if not address_parts:
        raise HTTPException(status_code=400, detail="Could not parse address from URL")
        
    location = address_parts[0]
    logger.info(f"Extracted location from URL: {location}")
    
    # Search for the property
    params = {
        "location": location,
        "listing_type": "for_sale",
        "limit": 1,
    }
    
    if site:
        params["site_name"] = site.value
        
    raw_properties = safe_scrape(params)
    
    if not raw_properties:
        params["listing_type"] = "sold"
        raw_properties = safe_scrape(params)
        
    if not raw_properties:
        raise HTTPException(status_code=404, detail="Property not found")
    
    prop = raw_properties[0]
    logger.info(f"Found property from URL, has photo: {prop.get('primary_photo') is not None}")
    return raw_property_to_response(prop)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
