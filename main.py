"""
Dock Property API - FastAPI backend using HomeHarvest
Provides property data from Zillow, Redfin, and Realtor.com
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from homeharvest import scrape_property
import pandas as pd
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dock Property API",
    description="Property data API powered by HomeHarvest",
    version="1.0.0"
)

# CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def safe_int(value) -> Optional[int]:
    """Safely convert to int, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_float(value) -> Optional[float]:
    """Safely convert to float, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_str(value) -> Optional[str]:
    """Safely convert to string, handling NaN and None"""
    if pd.isna(value) or value is None:
        return None
    return str(value)


def parse_alt_photos(value) -> Optional[List[str]]:
    """Parse alt_photos which may be a string or list"""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Could be comma-separated or other format
        return [v.strip() for v in value.split(",") if v.strip()]
    return None


def df_row_to_property(row) -> PropertyResponse:
    """Convert a DataFrame row to PropertyResponse"""
    # Handle bathrooms - might be full_baths + half_baths or just baths
    bathrooms = None
    if row.get("full_baths") is not None:
        full = safe_float(row.get("full_baths")) or 0
        half = safe_float(row.get("half_baths")) or 0
        bathrooms = full + (half * 0.5)
    else:
        bathrooms = safe_float(row.get("baths"))
    
    return PropertyResponse(
        address=safe_str(row.get("street")),
        city=safe_str(row.get("city")),
        state=safe_str(row.get("state")),
        zip_code=safe_str(row.get("zip_code")),
        latitude=safe_float(row.get("latitude")),
        longitude=safe_float(row.get("longitude")),
        price=safe_float(row.get("list_price")),
        bedrooms=safe_int(row.get("beds")),
        bathrooms=bathrooms,
        sqft=safe_int(row.get("sqft")),
        lot_sqft=safe_int(row.get("lot_sqft")),
        year_built=safe_int(row.get("year_built")),
        property_type=safe_str(row.get("style")),
        price_per_sqft=safe_float(row.get("ppsqft")),
        hoa_fee=safe_float(row.get("hoa_fee")),
        days_on_mls=safe_int(row.get("days_on_mls")),
        list_date=safe_str(row.get("list_date")),
        sold_price=safe_float(row.get("sold_price")),
        last_sold_date=safe_str(row.get("last_sold_date")),
        assessed_value=safe_float(row.get("assessed_value")),
        estimated_value=safe_float(row.get("estimated_value")),
        mls_id=safe_str(row.get("mls_id")),
        listing_url=safe_str(row.get("property_url")),
        primary_photo=safe_str(row.get("primary_photo")),
        alt_photos=parse_alt_photos(row.get("alt_photos")),
        source=safe_str(row.get("source")),
        status=safe_str(row.get("status")),
        description=safe_str(row.get("text")),
        agent_name=safe_str(row.get("agent")),
        broker_name=safe_str(row.get("broker")),
        stories=safe_int(row.get("stories")),
        parking_garage=safe_int(row.get("parking_garage")),
    )


def safe_scrape(params: dict) -> Optional[pd.DataFrame]:
    """Safely scrape properties, returning None on any error"""
    try:
        df = scrape_property(**params)
        return df
    except Exception as e:
        logger.warning(f"Scrape failed for params {params}: {str(e)}")
        return None


@app.get("/")
async def root():
    return {
        "service": "Dock Property API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/search": "Search properties by location",
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
    }
    
    if site:
        params["site_name"] = site.value
    
    if past_days:
        params["past_days"] = past_days
        
    # Add price filters
    if min_price or max_price:
        params["min_price"] = min_price
        params["max_price"] = max_price
        
    # Add bed filters
    if min_beds or max_beds:
        params["min_beds"] = min_beds
        params["max_beds"] = max_beds
        
    # Add bath filter
    if min_baths:
        params["min_baths"] = min_baths
        
    # Add sqft filters
    if min_sqft or max_sqft:
        params["min_sqft"] = min_sqft
        params["max_sqft"] = max_sqft
    
    # Scrape properties with error handling
    df = safe_scrape(params)
    
    if df is None or df.empty:
        logger.info(f"No properties found for {location}")
        return SearchResponse(count=0, properties=[])
    
    # Limit results
    df = df.head(limit)
    
    # Convert to response
    properties = [df_row_to_property(row) for _, row in df.iterrows()]
    
    logger.info(f"Found {len(properties)} properties for {location}")
    return SearchResponse(count=len(properties), properties=properties)


@app.get("/property", response_model=PropertyResponse)
async def get_property_by_address(
    address: str = Query(..., description="Full address (e.g., '123 Main St, Austin, TX 78701')"),
    site: Optional[SiteSource] = Query(None, description="Source site preference"),
):
    """
    Get a specific property by its address.
    """
    logger.info(f"Property request: address={address}")
    
    params = {
        "location": address,
        "listing_type": "for_sale",
    }
    
    if site:
        params["site_name"] = site.value
    
    df = safe_scrape(params)
    
    if df is None or df.empty:
        # Try sold listings
        logger.info(f"No for_sale listings, trying sold for {address}")
        params["listing_type"] = "sold"
        df = safe_scrape(params)
        
    if df is None or df.empty:
        logger.info(f"Property not found: {address}")
        raise HTTPException(status_code=404, detail="Property not found")
    
    logger.info(f"Found property: {address}")
    return df_row_to_property(df.iloc[0])


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
    }
    
    if site:
        params["site_name"] = site.value
        
    df = safe_scrape(params)
    
    if df is None or df.empty:
        params["listing_type"] = "sold"
        df = safe_scrape(params)
        
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Property not found")
        
    return df_row_to_property(df.iloc[0])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
