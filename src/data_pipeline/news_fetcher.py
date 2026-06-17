import requests
import urllib.parse
import xml.etree.ElementTree as ET
import time
import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Incident type → human-readable search term
_INCIDENT_TYPE_TERMS = {
    "accident":     "road accident crash",
    "congestion":   "traffic congestion jam",
    "road_closure": "road closed blocked",
    "hazard":       "road hazard obstruction",
    "event":        "public event rally gathering",
    "waterlogging": "waterlogging flood",
    "protest":      "protest demonstration",
    "vip_movement": "VIP convoy road block",
}

# Always appended so results stay Bengaluru-scoped
_BASE_TERMS = "Bengaluru traffic"


def _build_query(incident_data: dict) -> str:
    """
    Build a hyper-local Bing search query from available incident context.

    Priority order for location term:
      1. metadata['address']           — most specific (e.g. "Peenya Industrial Area")
      2. metadata['landmark']          — fallback landmark
      3. metadata['corridor']          — road corridor name
      4. metadata['locality'] / area   — broader locality
      5. None → omit location, fall back to base terms only

    The incident_type (or event_cause) is mapped to a more descriptive
    search phrase so Bing returns relevant articles.
    """
    metadata = incident_data.get("metadata", {}) or {}

    # 1. Location term
    location_term = (
        metadata.get("address")
        or metadata.get("landmark")
        or metadata.get("corridor")
        or metadata.get("locality")
        or metadata.get("area")
        or ""
    )
    # Strip "NULL", empty strings, etc.
    if not location_term or str(location_term).strip().upper() in ("", "NULL", "NONE"):
        location_term = ""
    else:
        # Take only the first meaningful part to avoid over-specificity
        location_term = str(location_term).split(",")[0].strip()

    # 2. Incident type term
    raw_type = (
        metadata.get("event_cause")
        or incident_data.get("incident_type")
        or ""
    )
    type_term = _INCIDENT_TYPE_TERMS.get(str(raw_type).lower(), str(raw_type).replace("_", " "))

    # 3. Compose query
    parts = [_BASE_TERMS]
    if location_term:
        parts.append(location_term)
    if type_term:
        parts.append(type_term)

    return " ".join(parts)


class NewsFetcher:
    """
    Fetches hyper-local traffic news from Bing RSS.

    Each call to `fetch_for_incident` builds a dynamic query from the
    incident's address, corridor, and type — so "Peenya accident" and
    "Silk Board waterlogging" get separate, relevant result sets.

    A per-query cache (TTL: 120 s) prevents hammering Bing when the same
    location appears in rapid-fire simulator events.
    """

    CACHE_TTL = 120  # seconds

    def __init__(self):
        # query_string → {"ts": float, "articles": list}
        self._cache: dict[str, dict] = {}
        # Kept for backwards-compatibility with callers that pass a static query
        self._static_query = _BASE_TERMS

    # ------------------------------------------------------------------
    # Primary API: incident-aware fetch
    # ------------------------------------------------------------------

    def fetch_for_incident(self, incident_data: dict, limit: int = 5) -> list[dict]:
        """
        Fetch news articles using a query derived from incident context.
        Returns a list of article dicts: {title, pub_date, source}.
        """
        query = _build_query(incident_data)
        logger.info(f"NewsFetcher dynamic query: '{query}'")
        return self._fetch(query, limit)

    def check_for_active_keywords(
        self,
        incident_data: Optional[dict] = None,
        keywords: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Return article titles that contain any of the alert keywords.
        If incident_data is provided, uses a hyper-local query; otherwise
        falls back to the base Bengaluru traffic query.
        """
        if keywords is None:
            keywords = ["storm", "rain", "rally", "protest", "waterlogging",
                        "accident", "flood", "VIP", "blockade", "strike"]

        if incident_data:
            articles = self.fetch_for_incident(incident_data)
        else:
            articles = self._fetch(self._static_query)

        matched = []
        for n in articles:
            title_lower = n["title"].lower()
            if any(k.lower() in title_lower for k in keywords):
                matched.append(n["title"])
        return matched

    # Backwards-compat alias used by callers without incident context
    def get_latest_news(self, limit: int = 5) -> list[dict]:
        return self._fetch(self._static_query, limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, query: str, limit: int = 5) -> list[dict]:
        """Fetch & cache news for a given query string."""
        now = time.time()
        cached = self._cache.get(query)
        if cached and (now - cached["ts"]) < self.CACHE_TTL:
            return cached["articles"]

        url = f"https://www.bing.com/news/search?q={urllib.parse.quote(query)}&format=rss"
        articles: list[dict] = []
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            r = requests.get(url, headers=headers, timeout=10)
            root = ET.fromstring(r.text)

            for item in root.findall(".//item")[: limit * 3]:
                title_elem = item.find("title")
                title = title_elem.text if title_elem is not None else "No Title"

                pub_elem = item.find("pubDate")
                pub_date = (
                    pub_elem.text
                    if pub_elem is not None and pub_elem.text
                    else "Unknown Date"
                )

                # Drop articles older than 60 days
                if pub_date != "Unknown Date":
                    try:
                        dt = parsedate_to_datetime(pub_date)
                        if datetime.now(timezone.utc) - dt > timedelta(days=60):
                            continue
                    except Exception:
                        pass

                source_elem = item.find("source")
                source = source_elem.text if source_elem is not None else "Bing News"

                articles.append({"title": title, "pub_date": pub_date, "source": source})
                if len(articles) >= limit:
                    break

            logger.info(f"NewsFetcher: fetched {len(articles)} articles for '{query}'")
        except Exception as e:
            logger.error(f"NewsFetcher: failed to fetch news for '{query}': {e}")

        self._cache[query] = {"ts": now, "articles": articles}
        return articles
