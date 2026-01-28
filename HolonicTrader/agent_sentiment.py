"""
SentimentHolon - The "Speculator" Brain (Phase 38)
Specialized in:
1. Fetching News RSS Feeds (CryptoPanic, CoinDesk)
2. Analyzing Keyword Sentiment (Bullish/Bearish)
3. Providing a 'Sentiment Bias' (-1.0 to 1.0) to the Oracle
"""

from typing import Any, List, Dict
import time
import threading
from datetime import datetime, timezone
import re
from HolonicTrader.holon_core import Holon, Disposition
import config

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
        
        self.sources = getattr(config, 'SENTIMENT_SOURCES', [
            'https://cointelegraph.com/rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/'
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
        Main loop hook. Fetches news, updates score.
        Returns the current score.
        """
        now = time.time()
        # Update every 5 minutes
        if now - self.last_update < 300:
            return self.current_sentiment_score
            
        print(f"[{self.name}] 📰 Fetching Market News...")
        
        items = self._fetch_rss_items()
        if not items:
            print(f"[{self.name}] ⚠️ No news found or connection failed.")
            return self.current_sentiment_score
            
        score_sum = 0
        crisis_hits = 0
        count = 0
        new_feed_items = []
        
        for item in items:
            # Check cache based on title
            if item['title'] in self.news_cache:
                continue
                
            sent_score, is_crisis, is_whale, is_hype = self._analyze_text(item['title'])
            
            # Enrich item
            item['sentiment'] = sent_score
            item['is_crisis'] = is_crisis
            item['is_whale'] = is_whale
            item['is_hype'] = is_hype
            new_feed_items.append(item)
            
            score_sum += sent_score
            if is_crisis: crisis_hits += 1
            count += 1
            
            # Update cache (keep size manageable)
            self.news_cache.append(item['title'])
            if len(self.news_cache) > 200:
                self.news_cache.pop(0)
        
        # Update public feed (Prepend new items, keep max 50)
        if new_feed_items:
            self.latest_news = new_feed_items + self.latest_news
            self.latest_news = self.latest_news[:50]

        if count > 0:
            avg_batch_score = score_sum / count
            # Smoothing: Combine with previous score (Exponential Moving Average)
            # Alpha = 0.3 (New news affects 30% of global sentiment)
            alpha = 0.3
            self.current_sentiment_score = (self.current_sentiment_score * (1-alpha)) + (avg_batch_score * alpha)
            
            # Crisis Logic: Each hit adds 0.2 to crisis score, decays by 10% naturally if no news
            current_crisis_impact = min(1.0, crisis_hits * 0.2)
            # FIX: Clamp to 1.0 MAX to prevent hallucination
            self.crisis_score = min(1.0, max(0.0, (self.crisis_score * 0.9) + current_crisis_impact))
            
            print(f"[{self.name}] 🧠 Sentiment Updated. Batch: {avg_batch_score:.2f} -> Global: {self.current_sentiment_score:.2f} | ☢️ Crisis Score: {self.crisis_score:.2f}")
        
        self.last_update = now
        return self.current_sentiment_score

    def _fetch_rss_items(self) -> List[dict]:
        if not feedparser:
            # Mock mode if lib missing
            return [
                {'title': "Bitcoin soars to new highs", 'link': 'http://google.com', 'source': 'MockFeed'}
            ] if self.current_sentiment_score < 0.5 else [
                {'title': "Hack detected in bridge", 'link': 'http://google.com', 'source': 'MockFeed'}
            ]
            
        all_items = []
        import requests # Lazy import or ensure at top
        
        for url in self.sources:
            try:
                # FIX: Use requests with timeout & user-agent to prevent hanging/blocking (Reddit)
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                resp = requests.get(url, timeout=5.0, headers=headers)
                if resp.status_code != 200:
                    print(f"[{self.name}] ⚠️ Feed Error {resp.status_code}: {url}")
                    continue
                    
                feed = feedparser.parse(resp.content)
                source_name = feed.feed.get('title', 'Unknown Source')
                for entry in feed.entries[:5]: # Top 5 per source
                    all_items.append({
                        'title': entry.title,
                        'link': entry.link,
                        'source': source_name
                    })
            except Exception as e:
                print(f"[{self.name}] RSS Error ({url}): {str(e)[:50]}...")
                
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
