from src.data_pipeline.news_fetcher import NewsFetcher
fetcher = NewsFetcher()
import logging
logging.basicConfig(level=logging.DEBUG)
print(fetcher.get_latest_news())
