import requests
import xml.etree.ElementTree as ET
import time
import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self, query="Bengaluru traffic weather protest rally"):
        import urllib.parse
        q = urllib.parse.quote(query)
        self.url = f"https://www.bing.com/news/search?q={q}&format=rss"
        self.last_fetch = 0.0
        self.cache = []
        self.cache_ttl = 120 # 2 minutes

    def get_latest_news(self, limit=5):
        now = time.time()
        if now - self.last_fetch > self.cache_ttl or not self.cache:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                r = requests.get(self.url, headers=headers, timeout=10)
                root = ET.fromstring(r.text)
                news = []
                for item in root.findall('.//item')[:limit*3]:
                    title_elem = item.find('title')
                    title = title_elem.text if title_elem is not None else "No Title"
                    
                    pub_elem = item.find('pubDate')
                    pubDate = pub_elem.text if pub_elem is not None and pub_elem.text is not None else "Unknown Date"
                    
                    if pubDate != "Unknown Date":
                        try:
                            dt = parsedate_to_datetime(pubDate)
                            if datetime.now(timezone.utc) - dt > timedelta(days=60):
                                continue
                        except Exception:
                            pass
                    
                    source_elem = item.find('source')
                    source = source_elem.text if source_elem is not None else "Google News"
                    news.append({
                        "title": title,
                        "pub_date": pubDate,
                        "source": source
                    })
                    
                    if len(news) >= limit:
                        break
                self.cache = news
                logger.info(f"Fetched {len(news)} latest news articles.")
            except Exception as e:
                logger.error(f"Failed to fetch news: {e}")
            finally:
                self.last_fetch = now
        
        return self.cache

    def check_for_active_keywords(self, keywords=['storm', 'rain', 'rally', 'protest', 'waterlogging', 'accident']):
        news = self.get_latest_news()
        matched = []
        for n in news:
            title_lower = n['title'].lower()
            for k in keywords:
                if k in title_lower:
                    matched.append(n['title'])
                    break
        return matched
