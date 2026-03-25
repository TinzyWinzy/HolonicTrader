"""
SentimentHolon - The "Speculator" Brain (Phase 38 + AEGIS M-01 Fix)
Specialized in:
1. Fetching News RSS Feeds (Multiple redundant sources)
2. Analyzing Keyword Sentiment (Bullish/Bearish)
3. Providing a 'Sentiment Bias' (-1.0 to 1.0) to the Oracle

AEGIS M-01 FIX:
- Redundant RSS feed aggregator with 15+ sources
- Automatic health monitoring and failover
- Alternative aggregators when primary feeds fail
"""

from typing import Any, List, Dict
import time
import threading
from datetime import datetime, timezone
import re
from HolonicTrader.holon_core import Holon, Disposition
import config

# AEGIS M-01: Import redundant RSS aggregator
try:
    from HolonicTrader.rss_aggregator import (
        RedundantRSSAggregator,
        get_global_rss_aggregator,
        PRIMARY_SOURCES,
        AGGREGATOR_SOURCES,
        ALTERNATIVE_SOURCES
    )
    AEGIS_RSS_ENABLED = True
except ImportError:
    AEGIS_RSS_ENABLED = False
    print("[SentimentHolon] ℹ️ AEGIS RSS Aggregator not available, using fallback")

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

class SentimentHolon(Holon):
    def __init__(self, name: str = "SentimentHolon"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.5, integration=0.5))

        # AEGIS M-01: Use redundant RSS aggregator
        if AEGIS_RSS_ENABLED:
            self.rss_aggregator = get_global_rss_aggregator()
            print(f"[{self.name}] 🛡️ AEGIS Redundant RSS Aggregator enabled (15+ sources)")
        else:
            self.rss_aggregator = None
            # Fallback to legacy sources
            self.sources = getattr(config, 'SENTIMENT_SOURCES', [
                'https://cointelegraph.com/rss',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://www.theblockcrypto.com/rss',  # AEGIS addition
                'https://decrypt.co/feed',  # AEGIS addition
                'https://cryptoslate.com/feed/'  # AEGIS addition
            ])

        # State
        self.current_sentiment_score = 0.0 # -1.0 (Extreme Fear) to 1.0 (Extreme Greed)
        self.last_update = 0
        self.news_cache = [] # List of processed titles to avoid duplicates
        self._lock = threading.Lock()

        # Keywords (Heuristic Fallback)
        self.bull_words = ['soar', 'surge', 'bull', 'adoption', 'partner', 'record', 'high', 'etf', 'approve', 'gain', 'jump']
        self.bear_words = ['crash', 'drop', 'bear', 'ban', 'hack', 'lawsuit', 'sec', 'fraud', 'plummet', 'low', 'dump']
        # === MACRO STRATEGY: Crisis Lexicon ===
        self.crisis_words = ['war', 'invasion', 'missile', 'conflict', 'army', 'emergency', 'nuclear', 'sanction', 'oil price', 'shortage', 'tank', 'military']
        # === WHALE TRACKING ===
        self.whale_words = ['whale', 'large transfer', 'accumulating', 'wallet', 'millions', 'movement', 'alert', 'dormant']

        # === HYPE / PUMP TRACKING (The "Rocket" Fuel) ===
        self.hype_words = ['moon', 'gem', '100x', 'parabolic', 'pump', 'shill', 'breakout', 'alpha', 'lambo', 'ape in']

        self.crisis_score = 0.0 # 0.0 (Peace) to 1.0 (Global Conflict)
        self.latest_news = []   # List of {'title': str, 'link': str, 'source': str, 'sentiment': float, 'is_crisis': bool}

    def fetch_and_analyze(self) -> float:
        """
        Main loop hook. Fetches news using AEGIS redundant aggregator, updates score.
        Returns the current score.

        AEGIS M-01 FIX: Uses redundant RSS aggregator with automatic failover.
        """
        now = time.time()
        # Update every 5 minutes
        if now - self.last_update < 300:
            return self.current_sentiment_score

        print(f"[{self.name}] 📰 Fetching Market News...")

        # AEGIS M-01: Use redundant aggregator
        if AEGIS_RSS_ENABLED and self.rss_aggregator:
            items = self._fetch_via_aggregator()
        else:
            # Fallback to legacy method
            items = self._fetch_rss_items()

        if not items:
            print(f"[{self.name}] ⚠️ No news found or connection failed.")
            # AEGIS: Check aggregator health for diagnostics
            if AEGIS_RSS_ENABLED and self.rss_aggregator:
                health = self.rss_aggregator.get_health_report()
                print(f"[{self.name}] 📊 RSS Health: {health['healthy_feeds']}/{health['total_feeds']} feeds healthy")
            return self.current_sentiment_score

        score_sum = 0
        crisis_hits = 0
        count = 0
        new_feed_items = []

        for item in items:
            # Check cache based on title or hash
            item_title = item.title if hasattr(item, 'title') else item['title']
            if item_title in self.news_cache:
                continue

            sent_score, is_crisis, is_whale, is_hype = self._analyze_text(item_title)

            # Enrich item
            if hasattr(item, 'link'):
                item_dict = {
                    'title': item.title,
                    'link': item.link,
                    'source': item.source,
                    'sentiment': sent_score,
                    'is_crisis': is_crisis,
                    'is_whale': getattr(item, 'is_whale', is_whale),
                    'is_hype': getattr(item, 'is_hype', is_hype)
                }
            else:
                item['sentiment'] = sent_score
                item['is_crisis'] = is_crisis
                item['is_whale'] = is_whale
                item['is_hype'] = is_hype
                item_dict = item

            new_feed_items.append(item_dict)

            score_sum += sent_score
            if is_crisis: crisis_hits += 1
            count += 1

            # Update cache (keep size manageable)
            self.news_cache.append(item_title)
            if len(self.news_cache) > 200:
                self.news_cache.pop(0)

        # Update public feed (Prepend new items, keep max 50)
        if new_feed_items:
            self.latest_news = new_feed_items + self.latest_news
            self.latest_news = self.latest_news[:50]

        if count > 0:
            avg_batch_score = score_sum / count
            # Smoothing: Combine with previous score (Exponential Moving Series)
            # Alpha = 0.3 (New news affects 30% of global sentiment)
            alpha = 0.3
            self.current_sentiment_score = (self.current_sentiment_score * (1-alpha)) + (avg_batch_score * alpha)

            # FIX BUG-006: Crisis score decay mechanism
            # Crisis decays by 30% per update (faster decay to prevent stuck-at-1.0)
            # Each crisis hit adds 0.15 (reduced from 0.2) to prevent rapid escalation
            decay_rate = 0.70  # 30% decay per update
            crisis_impact_per_hit = 0.15

            # Calculate crisis impact from current batch
            current_crisis_impact = min(1.0, crisis_hits * crisis_impact_per_hit)

            # Apply decay first, then add new impact
            # This ensures crisis score naturally decays even with some hits
            self.crisis_score = min(1.0, max(0.0, (self.crisis_score * decay_rate) + current_crisis_impact))

            # FIX: Add time-based decay when no crisis news found
            # If no crisis hits, decay faster to return to normal
            if crisis_hits == 0:
                self.crisis_score = max(0.0, self.crisis_score * 0.5)  # 50% decay when no crisis

            print(f"[{self.name}] 🧠 Sentiment Updated. Batch: {avg_batch_score:.2f} -> Global: {self.current_sentiment_score:.2f} | ☢️ Crisis Score: {self.crisis_score:.2f}")

        self.last_update = now
        return self.current_sentiment_score

    def _fetch_via_aggregator(self) -> List[Any]:
        """
        AEGIS M-01: Fetch news via redundant RSS aggregator.

        Returns:
            List of NewsItem objects
        """
        try:
            items = self.rss_aggregator.fetch_news(force_refresh=True)

            # Log aggregator health
            health = self.rss_aggregator.get_health_report()
            if health['disabled_feeds'] > 0:
                print(f"[{self.name}] ⚠️ {health['disabled_feeds']} RSS feeds disabled (auto-failover active)")

            return items
        except Exception as e:
            print(f"[{self.name}] ❌ Aggregator fetch error: {e}")
            return []

    def _fetch_rss_items(self) -> List[dict]:
        if not feedparser:
            # Mock mode if lib missing
            return [
                {'title': "Bitcoin soars to new highs", 'link': 'http://google.com', 'source': 'MockFeed'}
            ] if self.current_sentiment_score < 0.5 else [
                {'title': "Hack detected in bridge", 'link': 'http://google.com', 'source': 'MockFeed'}
            ]
            
        all_items = []
        import requests 
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Helper for user-agent rotation
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        
        sources_to_try = list(self.sources)
        if 'https://cryptopanic.com/news/rss/' not in sources_to_try:
             sources_to_try.append('https://cryptopanic.com/news/rss/')
        
        def fetch_single_feed(url):
            try:
                headers = {
                    'User-Agent': user_agents[hash(url) % len(user_agents)],
                    'Accept': 'application/rss+xml, application/xml, text/xml'
                }
                # Fix: Increase to 10.0s timeout to avoid heavy blocking/latency from RSS feeds
                # while allowing resilient connections
                resp = requests.get(url, timeout=10.0, headers=headers)
                
                if resp.status_code == 502:
                     print(f"[{self.name}] ⚠️ Feed 502 Bad Gateway: {url}.")
                     return []
                elif resp.status_code != 200:
                    print(f"[{self.name}] ⚠️ Feed Error {resp.status_code}: {url}")
                    return []
                    
                feed = feedparser.parse(resp.content)
                source_name = feed.feed.get('title', 'Unknown Source')
                
                if not feed.entries:
                     return []
                     
                items = []
                for entry in feed.entries[:5]: # Top 5 per source
                    items.append({
                        'title': entry.title,
                        'link': entry.link,
                        'source': source_name
                    })
                return items
            except Exception as e:
                print(f"[{self.name}] RSS Error ({url}): {str(e)[:50]}...")
                return []

        # Use ThreadPoolExecutor to run requests in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(fetch_single_feed, url): url for url in sources_to_try}
            for future in as_completed(future_to_url):
                try:
                    all_items.extend(future.result())
                except Exception as exc:
                    url = future_to_url[future]
                    print(f"[{self.name}] Exception parallel fetching {url}: {exc}")
                    
        return all_items

    def _analyze_text(self, text: str) -> tuple:
        text_lower = text.lower()
        score = 0.0
        is_crisis = False
        is_whale = False
        
        # 1. TextBlob Polarity (-1 to 1)
        if TextBlob:
            blob = TextBlob(text)
            score += blob.sentiment.polarity
            
        # 2. Keyword Boaster
        for w in self.bull_words:
            if w in text_lower: score += 0.3
            
        for w in self.bear_words:
            if w in text_lower: score -= 0.3
            
        # 3. Crisis Check
        for w in self.crisis_words:
            if w in text_lower:
                is_crisis = True
                score -= 0.5 # Crisis is generally bearish for markets
        
        # 4. Whale Check
        for w in self.whale_words:
            if w in text_lower:
                is_whale = True
                score += (config.WHALE_SENTIMENT_WEIGHT if score > 0 else -config.WHALE_SENTIMENT_WEIGHT)

        # 5. Hype Check (The Rocket)
        is_hype = False
        for w in self.hype_words:
            if w in text_lower:
                score += 0.4 # Hype is very bullish short-term
                is_hype = True
            
        # Clamp
        return max(-1.0, min(1.0, score)), is_crisis, is_whale, is_hype

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
