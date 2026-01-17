"""Shared helper functions for scraping and mapping property data."""

from typing import Optional, List, Any, Dict
import logging
import re
from homeharvest import scrape_property

from schemas import PropertyResponse


logger = logging.getLogger(__name__)


def safe_int(value) -> Optional[int]:
    """Safely convert to int, handling None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_float(value) -> Optional[float]:
    """Safely convert to float, handling None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_str(value) -> Optional[str]:
    """Safely convert to string, handling None."""
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


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 0:
        return (values_sorted[mid - 1] + values_sorted[mid]) / 2
    return values_sorted[mid]


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


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
    """Extract URL from photo value - handles both string and dict formats.

    Args:
        photo_value: The photo data (string URL or dict with 'href')
        high_quality: If True, attempt to get larger image by modifying URL suffix
    """
    if photo_value is None:
        return None

    url = None

    # HomeHarvest raw format returns photos as {'href': 'url'} dicts
    if isinstance(photo_value, dict):
        url = photo_value.get("href")
    elif isinstance(photo_value, str):
        url = photo_value if photo_value else None

    if not url:
        return None

    # Upgrade image quality for realtor.com images (rdcpix.com)
    if high_quality and "rdcpix.com" in url:
        url = re.sub(r"([0-9]+)[sm]\.jpg(\?.*)?$", r"\1l.jpg\2", url)
        url = re.sub(r"-[sm]\.jpg(\?.*)?$", r"-l.jpg\1", url)
        if "?w=" in url or "&w=" in url:
            url = re.sub(r"([?&])w=\d+", r"\1w=1080", url)
        url = re.sub(r"-w\d+_", "-w1080_", url)
        url = re.sub(r"/e-[sm]/", "/e-l/", url)

    return url


def raw_property_to_response(prop: Dict) -> PropertyResponse:
    """Convert a raw property dict (from return_type='raw') to PropertyResponse."""
    location = prop.get("location") or {}
    address_info = location.get("address") or {} if isinstance(location, dict) else {}
    coordinate = address_info.get("coordinate") or {} if isinstance(address_info, dict) else {}

    raw_description = prop.get("description")
    details = raw_description if isinstance(raw_description, dict) else {}

    full_baths = safe_float(details.get("baths_full")) or safe_float(prop.get("full_baths")) or 0
    half_baths = safe_float(details.get("baths_half")) or safe_float(prop.get("half_baths")) or 0
    total_baths = safe_float(details.get("baths")) or safe_float(prop.get("baths"))

    if total_baths:
        bathrooms = total_baths
    elif full_baths or half_baths:
        bathrooms = full_baths + (half_baths * 0.5)
    else:
        bathrooms = None

    raw_primary_photo = prop.get("primary_photo")
    primary_photo = extract_photo_url(raw_primary_photo)

    raw_alt_photos = prop.get("photos") or prop.get("alt_photos")
    alt_photos = None
    if raw_alt_photos and isinstance(raw_alt_photos, list):
        alt_photos = [url for p in raw_alt_photos if (url := extract_photo_url(p))]
        alt_photos = alt_photos if alt_photos else None

    raw_primary_url = raw_primary_photo.get("href") if isinstance(raw_primary_photo, dict) else raw_primary_photo
    logger.info(
        "Property photos - original_url: %s",
        raw_primary_url[:100] if raw_primary_url else "None",
    )
    logger.info(
        "Property photos - transformed_url: %s, alt_count: %s",
        primary_photo[:100] if primary_photo else "None",
        len(alt_photos) if alt_photos else 0,
    )

    street = safe_str(address_info.get("line")) or safe_str(prop.get("street"))
    city = safe_str(address_info.get("city")) or safe_str(prop.get("city"))
    state = (
        safe_str(address_info.get("state_code"))
        or safe_str(address_info.get("state"))
        or safe_str(prop.get("state"))
    )
    zip_code = safe_str(address_info.get("postal_code")) or safe_str(prop.get("zip_code"))

    latitude = safe_float(coordinate.get("lat")) or safe_float(prop.get("latitude"))
    longitude = safe_float(coordinate.get("lon")) or safe_float(prop.get("longitude"))

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

    hoa_info = prop.get("hoa") or {}
    hoa_fee = safe_float(hoa_info.get("fee")) if isinstance(hoa_info, dict) else safe_float(prop.get("hoa_fee"))

    listing_url = safe_str(prop.get("href")) or safe_str(prop.get("permalink")) or safe_str(prop.get("property_url"))
    if listing_url and not listing_url.startswith("http"):
        listing_url = f"https://www.realtor.com{listing_url}"

    desc_text = (
        safe_str(details.get("text"))
        or safe_str(prop.get("text"))
        or safe_str(raw_description)
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
    """Safely scrape properties using raw return type to get photos, returning None on any error."""
    try:
        params["return_type"] = "raw"
        result = scrape_property(**params)

        if result and isinstance(result, list):
            logger.info("Scrape returned %s properties", len(result))
            return result
        return None
    except Exception as exc:
        logger.warning("Scrape failed for params %s: %s", params, str(exc))
        return None
