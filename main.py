"""
Dock Property API - FastAPI backend using HomeHarvest
Provides property data from Zillow, Redfin, and Realtor.com
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from homeharvest import scrape_property
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
    """Safely convert to int, handling None"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_float(value) -> Optional[float]:
    """Safely convert to float, handling None"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_str(value) -> Optional[str]:
    """Safely convert to string, handling None"""
    if value is None:
        return None
    s = str(value)
    return s if s else None


def get_nested(data: Dict, *keys, default=None):
    """Safely get nested dictionary values"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
        if data is None:
            return default
    return data


def raw_property_to_response(prop: Dict) -> PropertyResponse:
    """Convert a raw property dict (from return_type='raw') to PropertyResponse"""
    # Handle bathrooms - might be full_baths + half_baths
    full_baths = safe_float(prop.get("full_baths")) or 0
    half_baths = safe_float(prop.get("half_baths")) or 0
    bathrooms = full_baths + (half_baths * 0.5) if (full_baths or half_baths) else None
    
    # Get primary photo - this is the key field we need!
    raw_primary_photo = prop.get("primary_photo")
    primary_photo = safe_str(raw_primary_photo)
    
    # Get alt photos as list
    alt_photos = prop.get("alt_photos")
    if alt_photos and isinstance(alt_photos, list):
        alt_photos = [str(p) for p in alt_photos if p]
    else:
        alt_photos = None
    
    # Log photo info for debugging - include actual URL for verification
    logger.info(f"Property photos - raw_type: {type(raw_primary_photo).__name__}, primary: {primary_photo is not None}, url_preview: {str(primary_photo)[:80] if primary_photo else 'None'}")
    
    return PropertyResponse(
        address=safe_str(prop.get("street")),
        city=safe_str(prop.get("city")),
        state=safe_str(prop.get("state")),
        zip_code=safe_str(prop.get("zip_code")),
        latitude=safe_float(prop.get("latitude")),
        longitude=safe_float(prop.get("longitude")),
        price=safe_float(prop.get("list_price")),
        bedrooms=safe_int(prop.get("beds")),
        bathrooms=bathrooms,
        sqft=safe_int(prop.get("sqft")),
        lot_sqft=safe_int(prop.get("lot_sqft")),
        year_built=safe_int(prop.get("year_built")),
        property_type=safe_str(prop.get("style")),
        price_per_sqft=safe_float(prop.get("price_per_sqft")),
        hoa_fee=safe_float(prop.get("hoa_fee")),
        days_on_mls=safe_int(prop.get("days_on_mls")),
        list_date=safe_str(prop.get("list_date")),
        sold_price=safe_float(prop.get("sold_price")),
        last_sold_date=safe_str(prop.get("last_sold_date")),
        assessed_value=safe_float(prop.get("tax_assessed_value")),
        estimated_value=safe_float(prop.get("estimated_value")),
        mls_id=safe_str(prop.get("mls_id")),
        listing_url=safe_str(prop.get("property_url")),
        primary_photo=primary_photo,
        alt_photos=alt_photos,
        source=safe_str(prop.get("mls")),
        status=safe_str(prop.get("status")),
        description=safe_str(prop.get("text")),
        agent_name=safe_str(prop.get("agent_name")),
        broker_name=safe_str(prop.get("broker_name")),
        stories=safe_int(prop.get("stories")),
        parking_garage=safe_int(prop.get("garage")),
    )


def safe_scrape(params: dict) -> Optional[List[Dict]]:
    """Safely scrape properties using raw return type to get photos, returning None on any error"""
    try:
        # IMPORTANT: Use return_type='raw' to get primary_photo and alt_photos
        # These fields are NOT available in the default pandas DataFrame!
        params["return_type"] = "raw"
        result = scrape_property(**params)
        
        # Raw return type returns a list of dicts
        if result and isinstance(result, list):
            logger.info(f"Scrape returned {len(result)} properties")
            return result
        return None
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
    # Log the raw primary_photo value to debug
    raw_photo = prop.get('primary_photo')
    logger.info(f"Found property: {address}")
    logger.info(f"Raw primary_photo type: {type(raw_photo).__name__}, value: {str(raw_photo)[:100] if raw_photo else 'None'}")
    logger.info(f"=== PROPERTY REQUEST END ===")
    return raw_property_to_response(prop)


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
