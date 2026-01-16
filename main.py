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


def get_nested(data, *keys, default=None):
    """Safely get nested dictionary/list values. Supports both dict keys and list indices."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, list) and isinstance(key, int):
            if 0 <= key < len(data):
                data = data[key]
            else:
                return default
        else:
            return default
        if data is None:
            return default
    return data


def first_non_empty(*values):
    """Return the first non-empty value (None/empty string are skipped)."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def get_first(data, *paths):
    """Get the first non-empty value from a list of keys or nested paths."""
    for path in paths:
        if isinstance(path, (list, tuple)):
            value = get_nested(data, *path)
        else:
            value = data.get(path) if isinstance(data, dict) else None
        value = first_non_empty(value)
        if value is not None:
            return value
    return None


def normalize_history(value):
    """Normalize history fields to a list of dicts if possible."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        history = value.get("history")
        if isinstance(history, list):
            return history
    return None


def extract_photo_url(photo_value, high_quality: bool = True) -> Optional[str]:
    """Extract URL from photo value - handles both string and dict formats
    
    Args:
        photo_value: The photo data (string URL or dict with 'href')
        high_quality: If True, attempt to get larger image by modifying URL suffix
    """
    import re
    
    if photo_value is None:
        return None
    
    url = None
    
    # HomeHarvest raw format returns photos as {'href': 'url'} dicts
    if isinstance(photo_value, dict):
        url = photo_value.get('href')
    elif isinstance(photo_value, str):
        url = photo_value if photo_value else None
    
    if not url:
        return None
    
    # Upgrade image quality for realtor.com images (rdcpix.com)
    # URL patterns vary but generally have size indicators:
    # - 's' = small (~140px), 'm' = medium (~280px), 'l' = large (~640px), 'o'/'od' = original
    # Examples:
    #   - https://ap.rdcpix.com/xxx/xxx-m1234567890s.jpg (small)
    #   - https://ap.rdcpix.com/xxx/xxx-c1234-m1234567890s.jpg (small with crop)
    #   - https://ap.rdcpix.com/xxx/xxx-w640_h480_x2-m1234567890s.jpg (small with dimensions)
    if high_quality and 'rdcpix.com' in url:
        # Pattern 1: URL ends with size indicator before .jpg
        # Match any character sequence ending in s.jpg or m.jpg and change to l.jpg (large)
        # The size indicator comes right before .jpg, e.g., "...123456s.jpg" or "...123456m.jpg"
        url = re.sub(r'([0-9]+)[sm]\.jpg(\?.*)?$', r'\1l.jpg\2', url)
        
        # Pattern 2: Handle URLs with -s.jpg or -m.jpg suffix
        url = re.sub(r'-[sm]\.jpg(\?.*)?$', r'-l.jpg\1', url)
        
        # Pattern 3: Handle width parameter in query string (e.g., ?w=140 -> ?w=1080)
        if '?w=' in url or '&w=' in url:
            url = re.sub(r'([?&])w=\d+', r'\1w=1080', url)
        
        # Pattern 4: Handle width in path (e.g., -w140_ -> -w1080_)
        url = re.sub(r'-w\d+_', '-w1080_', url)
        
        # Pattern 5: Handle some URLs with 'e-' prefix for size (e.g., e-s, e-m, e-l)
        url = re.sub(r'/e-[sm]/', '/e-l/', url)
    
    return url


def raw_property_to_response(prop: Dict) -> PropertyResponse:
    """Convert a raw property dict (from return_type='raw') to PropertyResponse
    
    The raw format from HomeHarvest follows the Realtor.com API structure with nested objects:
    - location: contains address info (address.line, address.city, etc.)
    - description: contains property details (beds, baths, sqft, year_built, etc.)
    - primary_photo: photo info
    - list_price: listing price
    """
    # Extract nested structures - realtor.com raw format uses 'description' for property details
    # and 'location' for address info
    location = prop.get("location") or {}
    address_info = location.get("address") or {} if isinstance(location, dict) else {}
    coordinate = address_info.get("coordinate") or {} if isinstance(address_info, dict) else {}
    
    # 'description' in raw format contains property details (beds, baths, sqft, etc.)
    # This is confusing but it's how the realtor.com API works
    raw_description = prop.get("description")
    details = raw_description if isinstance(raw_description, dict) else {}
    
    # Handle bathrooms - might be full_baths + half_baths or just baths
    # Try nested structure first, then fall back to flat structure
    full_baths = safe_float(details.get("baths_full")) or safe_float(prop.get("full_baths")) or 0
    half_baths = safe_float(details.get("baths_half")) or safe_float(prop.get("half_baths")) or 0
    total_baths = safe_float(details.get("baths")) or safe_float(prop.get("baths"))
    
    if total_baths:
        bathrooms = total_baths
    elif full_baths or half_baths:
        bathrooms = full_baths + (half_baths * 0.5)
    else:
        bathrooms = None
    
    # Get primary photo - extract URL from dict if needed
    raw_primary_photo = prop.get("primary_photo")
    primary_photo = extract_photo_url(raw_primary_photo)
    
    # Get alt photos from 'photos' array - each may be a dict with 'href'
    raw_alt_photos = prop.get("photos") or prop.get("alt_photos")
    alt_photos = None
    if raw_alt_photos and isinstance(raw_alt_photos, list):
        alt_photos = [url for p in raw_alt_photos if (url := extract_photo_url(p))]
        alt_photos = alt_photos if alt_photos else None
    
    # Log photo info for debugging (including original URL to verify transformation)
    raw_primary_url = raw_primary_photo.get('href') if isinstance(raw_primary_photo, dict) else raw_primary_photo
    logger.info(f"Property photos - original_url: {raw_primary_url[:100] if raw_primary_url else 'None'}")
    logger.info(f"Property photos - transformed_url: {primary_photo[:100] if primary_photo else 'None'}, alt_count: {len(alt_photos) if alt_photos else 0}")
    
    # Extract address - try nested location.address first, then flat fields
    street = safe_str(address_info.get("line")) or safe_str(prop.get("street"))
    city = safe_str(address_info.get("city")) or safe_str(prop.get("city"))
    state = safe_str(address_info.get("state_code")) or safe_str(address_info.get("state")) or safe_str(prop.get("state"))
    zip_code = safe_str(address_info.get("postal_code")) or safe_str(prop.get("zip_code"))
    
    # Extract coordinates
    latitude = safe_float(coordinate.get("lat")) or safe_float(prop.get("latitude"))
    longitude = safe_float(coordinate.get("lon")) or safe_float(prop.get("longitude"))
    
    # Extract property details - try nested 'description' first, then flat fields
    beds = safe_int(details.get("beds")) or safe_int(prop.get("beds"))
    sqft = safe_int(details.get("sqft")) or safe_int(prop.get("sqft"))
    lot_sqft = safe_int(details.get("lot_sqft")) or safe_int(prop.get("lot_sqft"))
    year_built = (
        safe_int(details.get("year_built"))
        or safe_int(details.get("yearBuilt"))
        or safe_int(prop.get("year_built"))
        or safe_int(prop.get("yearBuilt"))
        or safe_int(get_nested(prop, "description", "year_built"))
        or safe_int(get_nested(prop, "description", "yearBuilt"))
        or safe_int(get_nested(prop, "building", "year_built"))
        or safe_int(get_nested(prop, "building", "yearBuilt"))
        or safe_int(get_nested(prop, "property", "year_built"))
        or safe_int(get_nested(prop, "property", "yearBuilt"))
    )
    stories = safe_int(details.get("stories")) or safe_int(prop.get("stories"))
    garage = safe_int(details.get("garage")) or safe_int(prop.get("garage"))
    property_type = safe_str(details.get("type")) or safe_str(prop.get("style"))
    
    # Extract HOA fee - might be in nested 'hoa' object
    hoa_info = prop.get("hoa") or {}
    hoa_fee = safe_float(hoa_info.get("fee")) if isinstance(hoa_info, dict) else safe_float(prop.get("hoa_fee"))
    
    # Build listing URL from href or permalink
    listing_url = safe_str(prop.get("href")) or safe_str(prop.get("permalink")) or safe_str(prop.get("property_url"))
    if listing_url and not listing_url.startswith("http"):
        listing_url = f"https://www.realtor.com{listing_url}"
    
    # Get description text
    desc_text = (
        safe_str(details.get("text"))
        or safe_str(prop.get("text"))
        or safe_str(raw_description)  # Sometimes description is a plain string
        or safe_str(prop.get("remarks"))
        or safe_str(prop.get("public_remarks"))
    )

    list_date = safe_str(
        get_first(
            prop,
            "list_date",
            "list_date_utc",
            "listDate",
            ("listing", "list_date"),
            ("listing", "list_date_utc"),
            ("dates", "list_date"),
            ("dates", "listed"),
        )
    )

    sold_price = safe_float(
        get_first(
            prop,
            "sold_price",
            "last_sold_price",
            ("last_sold", "price"),
            ("sale", "price"),
        )
    )

    last_sold_date = safe_str(
        get_first(
            prop,
            "last_sold_date",
            "sold_date",
            ("last_sold", "date"),
            ("sale", "date"),
        )
    )

    assessed_value = safe_float(
        get_first(
            prop,
            "tax_assessed_value",
            "assessed_value",
            ("tax", "assessed_value"),
            ("tax", "total_assessed_value"),
            ("taxes", "assessed_value"),
            ("assessment", "total"),
        )
    )

    annual_taxes = safe_float(
        get_first(
            prop,
            "annual_taxes",
            "annual_tax",
            "tax_amount",
            ("tax", "amount"),
            ("tax", "tax_amount"),
            ("taxes", "amount"),
            ("tax", "annual"),
            ("taxes", "annual"),
        )
    )

    tax_rate = safe_float(
        get_first(
            prop,
            "tax_rate",
            ("tax", "rate"),
            ("taxes", "rate"),
        )
    )

    price_history = normalize_history(
        get_first(
            prop,
            "price_history",
            "property_history",
            ("property_history", "price_history"),
            ("price_history", "history"),
            ("property_history", "history"),
        )
    )

    tax_history = normalize_history(
        get_first(
            prop,
            "tax_history",
            ("tax_history", "history"),
            ("tax", "history"),
            ("taxes", "history"),
            ("tax", "tax_history"),
        )
    )
    
    return PropertyResponse(
        address=street,
        city=city,
        state=state,
        zip_code=zip_code,
        latitude=latitude,
        longitude=longitude,
        price=safe_float(prop.get("list_price")),
        bedrooms=beds,
        bathrooms=bathrooms,
        sqft=sqft,
        lot_sqft=lot_sqft,
        year_built=year_built,
        property_type=property_type,
        price_per_sqft=safe_float(prop.get("price_per_sqft")),
        hoa_fee=hoa_fee,
        days_on_mls=safe_int(prop.get("days_on_mls")),
        list_date=list_date,
        sold_price=sold_price,
        last_sold_date=last_sold_date,
        assessed_value=assessed_value,
        estimated_value=safe_float(get_nested(prop, "estimates", "estimate", "value")),
        annual_taxes=annual_taxes,
        tax_rate=tax_rate,
        tax_history=tax_history,
        price_history=price_history,
        mls_id=safe_str(prop.get("mls_id")) or safe_str(prop.get("listing_id")),
        listing_url=listing_url,
        primary_photo=primary_photo,
        alt_photos=alt_photos,
        source=safe_str(get_nested(prop, "source", "id")) or safe_str(prop.get("mls")),
        status=safe_str(prop.get("status")),
        description=desc_text,
        agent_name=safe_str(get_nested(prop, "advertisers", 0, "name")),
        broker_name=safe_str(get_nested(prop, "advertisers", 0, "broker", "name")),
        stories=stories,
        parking_garage=garage,
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
