# Dock Property API

A FastAPI backend service that provides real estate property data using [HomeHarvest](https://github.com/ZacharyHampton/HomeHarvest).

## Features

- Search properties by location (city, state, or ZIP code)
- Filter by price, beds, baths, square footage
- Support for multiple listing types: for_sale, for_rent, sold, pending
- Data from Zillow, Redfin, and Realtor.com
- Get property details by address or listing URL

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# or
uvicorn main:app --reload
```

Server runs at http://localhost:8000

## API Endpoints

### GET /search
Search properties by location with filters.

Query Parameters:
- `location` (required): City, state or ZIP code
- `listing_type`: for_sale, for_rent, sold, pending (default: for_sale)
- `site`: zillow, redfin, realtor.com
- `min_price`, `max_price`: Price range
- `min_beds`, `max_beds`: Bedroom range
- `min_baths`: Minimum bathrooms
- `min_sqft`, `max_sqft`: Square footage range
- `past_days`: Filter to recent listings
- `limit`: Max results (default: 50)

### GET /property
Get property by address.

Query Parameters:
- `address` (required): Full address
- `site`: Preferred source

### GET /property/url
Get property from listing URL.

Query Parameters:
- `url` (required): Zillow, Redfin, or Realtor.com URL

## Deploy to Railway

1. Push this directory to a GitHub repo
2. Connect the repo to Railway
3. Deploy!

The `railway.toml` and `Procfile` are already configured.

## Environment Variables

No API keys required! HomeHarvest scrapes data directly.
# dock-api
