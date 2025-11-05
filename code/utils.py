import math
import re
from langchain_core.tools import tool
from datetime import datetime, timedelta

# ===========================================================================
# Utility functions and data for ReAct agent demo
# ===========================================================================

# Tiny city knowledge base (for demo purposes)
# lat, lon, currency, utc_offset_hours
CITY_DB = {
    "paris": {"lat": 48.8566, "lon": 2.3522, "currency": "EUR", "utc_offset": 1},
    "london": {"lat": 51.5074, "lon": -0.1278, "currency": "GBP", "utc_offset": 0},
    "new york": {"lat": 40.7128, "lon": -74.0060, "currency": "USD", "utc_offset": -5},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "currency": "JPY", "utc_offset": 9},
    "dubai": {"lat": 25.2048, "lon": 55.2708, "currency": "AED", "utc_offset": 4},
    "sydney": {"lat": -33.8688, "lon": 151.2093, "currency": "AUD", "utc_offset": 10},
}

# Static FX rates based USD (for demo purposes)
FX_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.78,
    "JPY": 134.5,
    "AED": 3.67,
    "AUD": 1.54,
}

_CITY_PATTERNS = [
    re.compile(r"^\s*from\s+(?P<a>.+?)\s+to\s+(?P<b>.+?)\s*$", re.IGNORECASE), # "from Paris to Tokyo"
    re.compile(r"^\s*(?P<a>.+?)\s+to\s+(?P<b>.+?)\s*$", re.IGNORECASE), # "Paris to Tokyo"
    re.compile(r"^\s*(?P<a>.+?)\s*[-–—>]+\s*(?P<b>.+?)\s*$", re.IGNORECASE),  # "Paris -> Tokyo", "Paris—Tokyo"
]

def _extract_number(text) -> float:
    """Return the first numeric value from strings like '9,712 km' or '9 712.4km'."""
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).replace("\u202f", "").replace("\xa0", " ")  # thin/nbsp
    s = s.replace("km", " ").replace("KM", " ")
    s = s.replace(",", " ").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Could not parse a number from '{text}'")
    return float(m.group(0).replace(" ", ""))

def _normalize_city(name: str) -> str:
    """Normalize city name for lookup."""
    # remove surrounding quotes/spaces and common punctuation
    cleaned = name.strip().strip("'\"`“”‘’.,;:()[]{}")
    return cleaned.lower()


def _parse_city_pair(query: str) -> tuple[str, str]:
    """Extract CityA, CityB from a variety of inputs."""
    q = query.strip()
    for pat in _CITY_PATTERNS:
        m = pat.match(q)
        if m:
            a = _normalize_city(m.group("a"))
            b = _normalize_city(m.group("b"))
            return a, b
    # last resort: try simple "to" split
    if "to" in q.lower():
        a, b = [s.strip() for s in re.split(r"\bto\b", q, flags=re.IGNORECASE, maxsplit=1)]
        return _normalize_city(a), _normalize_city(b)
    raise ValueError(f"Could not parse cities from '{query}'. Try 'CityA to CityB'.")


def _get_city_info(name: str) -> dict:
    """Get city info from the CITY_DB (after normalization)."""
    city_key = _normalize_city(name)
    city_info = CITY_DB.get(city_key)
    if not city_info:
        raise ValueError(f"Unknown city '{name}'. Known: {', '.join(CITY_DB)}")
    return city_info


def _get_city_info(name: str) -> dict:
    """Get city info from the CITY_DB."""
    city_info = CITY_DB.get(_normalize_city(name))
    if not city_info:
        raise ValueError(f"Unknown city '{name}'. Known: {', '.join(CITY_DB)}")
    return city_info

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371.0  # Earth radius in kilometers
    p1, p2 = math.radians(lat1), math.radians(lat2) # Convert degrees to radians
    dphi = math.radians(lat2 - lat1) # Delta phi
    dlambda = math.radians(lon2 - lon1) # Delta lambda
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2  # Haversine formula
    return R * 2 * math.asin(math.sqrt(a))

# ===========================================================================
# Tool definitions for ReAct agent demo
# ===========================================================================

@tool
def distance_km(query: str) -> float:
    """distance_km: CityA to CityB — Return great-circle distance (km).
    Accepts variants like: 'Paris to Tokyo', 'from Paris to Tokyo', 'Paris -> Tokyo'.
    """
    try:
        city_a_key, city_b_key = _parse_city_pair(query)
        a = _get_city_info(city_a_key)
        b = _get_city_info(city_b_key)
        return round(_haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 0)
    except Exception as e:
        raise ValueError(f"usage: 'CityA to CityB' — {e}")

@tool
def flight_emissions(km_str: str) -> float:
    """flight_emissions: <km> — Estimate CO2 emissions (kg) for a given flight distance.
    Uses a simple factor (short-haul vs long-haul).
    """
    km = _extract_number(km_str)
    factor = 0.12 if km < 3500 else 0.09  # demo factors only
    return round(km * factor, 2)

@tool
def convert_currency(spec: str) -> float:
    """convert_currency: <amount> <FROM> to <TO> — Convert amount using demo static FX rates.
    Example: '250 EUR to USD'
    """
    try:
        amount_str, rest = spec.strip().split(" ", 1)
        amount = float(amount_str)
        frm, _, to = rest.split()
        frm = frm.upper()
        to = to.upper()
        if frm not in FX_RATES or to not in FX_RATES:
            raise ValueError("Unknown currency")
        usd = amount / FX_RATES[frm]
        return round(usd * FX_RATES[to], 2)
    except Exception as e:
        raise ValueError(f"usage: '250 EUR to USD' — {e}")

@tool
def timezone_diff(spec: str) -> float:
    """timezone_diff: CityA vs CityB — Return time difference in hours (CityA − CityB).
    Example: 'Tokyo vs Paris'
    """
    try:
        left, right = [s.strip() for s in spec.split("vs")]
        a = _get_city_info(left)
        b = _get_city_info(right)
        return float(a["utc_offset"] - b["utc_offset"])
    except Exception as e:
        raise ValueError(f"usage: 'CityA vs CityB' — {e}")

@tool
def pack_suggestions(spec: str) -> str:
    """pack_suggestions: City in season — Return a comma-separated packing checklist (demo rules).
    Example: 'London in winter'
    """
    try:
        city, _, season = spec.partition(" in ")
        city = _normalize_city(city)
        season = season.strip().lower()
        base = ["passport", "charger", "toiletries"]
        cold = ["coat", "sweater", "gloves", "warm shoes"]
        warm = ["sunscreen", "hat", "light layers", "sunglasses"]
        lst = base + (cold if season in {"winter"} else warm if season in {"summer"} else ["light jacket"])
        return ", ".join(lst)
    except Exception as e:
        raise ValueError(f"usage: 'City in season' — {e}")

@tool
def calculate(expr: str) -> float:
    """calculate: <expression> — Evaluate a math expression (uses Python eval safely for demo)."""
    try:
        # Demo-safe eval: math namespace only
        import math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        return float(eval(expr, {"__builtins__": {}}, allowed))
    except Exception as e:
        raise ValueError(f"Bad expression: {e}")


# =======================================================   
# helper function to print llm responses
# =======================================================

def print_response_details(resp):
    import json

    # flatten various container shapes
    if isinstance(resp, dict):
        items = []
        for v in resp.values():
            if isinstance(v, list):
                items.extend(v)
            else:
                items.append(v)
    elif isinstance(resp, list):
        items = resp
    else:
        # fallback: try common attributes
        items = getattr(resp, "messages", getattr(resp, "generations", [resp]))

    for m in items:
        # dict-like message
        if isinstance(m, dict):
            content = m.get("content") or m.get("text") or ""
            print(f"DictMessage: {content}")
            if "tool_calls" in m:
                print("  tool_calls:", json.dumps(m["tool_calls"], indent=2))
            print("---")
            continue

        cls = m.__class__.__name__
        content = getattr(m, "content", None)
        print(f"{cls}: {content}")

        # ToolMessage specifics
        name = getattr(m, "name", None)
        if name:
            print(f"  tool name: {name}")

        # common places LangChain stores tool call info
        tool_calls = getattr(m, "tool_calls", None) or getattr(m, "tool_call", None)
        if tool_calls:
            print("  tool_calls:")
            try:
                print(json.dumps(tool_calls, default=str, indent=2))
            except Exception:
                print("   ", tool_calls)

        # some Versions put tool call info in response_metadata or additional_kwargs
        meta = getattr(m, "response_metadata", None) or getattr(m, "additional_kwargs", None)
        if isinstance(meta, dict):
            if "tool_calls" in meta:
                print("  response_metadata.tool_calls:", json.dumps(meta["tool_calls"], indent=2))
            if "finish_reason" in meta:
                print("  finish_reason:", meta["finish_reason"])

        print("---")