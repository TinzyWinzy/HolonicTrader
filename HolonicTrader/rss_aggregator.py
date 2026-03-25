"""
AEGIS QUANTSEC - Redundant RSS Feed Aggregator

Provides:
1. Multiple redundant RSS feed sources
2. Feed health monitoring and auto-failover
3. Alternative crypto news aggregators
4. Resilient fetching with circuit breakers

Addresses: M-01 Sentiment Feed Degradation

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import logging

logger = logging.getLogger("AEGIS.RSSAggregator")


# =============================================================================
# RSS FEED SOURCES - REDUNDANT CONFIGURATION
# =============================================================================

# Primary Sources (Major crypto news)
PRIMARY_SOURCES = [
    {
        'name': 'CoinDesk',
        'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'priority': 1,
        'category': 'news'
    },
    {
        'name': 'Cointelegraph',
        'url': 'https://cointelegraph.com/rss',
        'priority': 1,
        'category': 'news'
    },
    {
        'name': 'The Block',
        'url': 'https://www.theblockcrypto.com/rss',
        'priority': 1,
        'category': 'news'
    },
]

# Aggregator Sources (Multiple feeds in one)
AGGREGATOR_SOURCES = [
    {
        'name': 'CryptoPanic',
        'url': 'https://cryptopanic.com/news/rss/',
        'priority': 2,
        'category': 'aggregator'
    },
    {
        'name': 'CoinMarketCap News',
        'url': 'https://coinmarketcap.com/headlines/news/',
        'priority': 2,
        'category': 'aggregator'
    },
]

# Alternative/Specialized Sources
ALTERNATIVE_SOURCES = [
    {
        'name': 'Bitcoin Magazine',
        'url': 'https://bitcoinmagazine.com/.rss/all',
        'priority': 3,
        'category': 'specialized'
    },
    {
        'name': 'Decrypt',
        'url': 'https://decrypt.co/feed',
        'priority': 3,
        'category': 'news'
    },
    {
        'name': 'CoinJournal',
        'url': 'https://coinjournal.net/news/feed/',
        'priority': 3,
        'category': 'news'
    },
    {
        'name': 'NewsBTC',
        'url': 'https://www.newsbtc.com/feed/',
        'priority': 3,
        'category': 'news'
    },
    {
        'name': 'CryptoSlate',
        'url': 'https://cryptoslate.com/feed/',
        'priority': 3,
        'category': 'news'
    },
]

# Social/Community Sources
SOCIAL_SOURCES = [
    {
        'name': 'r/CryptoCurrency',
        'url': 'https://www.reddit.com/r/CryptoCurrency/top/.rss?t=hour',
        'priority': 4,
        'category': 'social'
    },
    {
        'name': 'r/Bitcoin',
        'url': 'https://www.reddit.com/r/Bitcoin/hot/.rss',
        'priority': 4,
        'category': 'social'
    },
    {
        'name': 'r/Ethereum',
        'url': 'https://www.reddit.com/r/ethereum/hot/.rss',
        'priority': 4,
        'category': 'social'
    },
]

# Macro/Traditional Finance (for crisis detection)
MACRO_SOURCES = [
    {
        'name': 'CNBC Crypto',
        'url': 'https://www.cnbc.com/id/100727362/device/rss/rss.html',
        'priority': 5,
        'category': 'macro'
    },
    {
        'name': 'Bloomberg Crypto',
        'url': 'https://www.bloomberg.com/crypto',
        'priority': 5,
        'category': 'macro'
    },
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeedHealth:
    """Health metrics for a single RSS feed."""
    url: str
    name: str
    success_count: int = 0
    failure_count: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    last_error: str = ""
    avg_response_time_ms: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    consecutive_failures: int = 0
    is_disabled: bool = False
    disabled_until: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0.0 - 1.0)."""
        if self.is_disabled:
            return 0.0
        
        # Factor 1: Success rate (60% weight)
        score = self.success_rate * 0.6
        
        # Factor 2: Recent failures (20% weight)
        if self.consecutive_failures >= 5:
            score *= 0.2
        elif self.consecutive_failures >= 3:
            score *= 0.5
        elif self.consecutive_failures >= 1:
            score *= 0.8
        
        # Factor 3: Response time (20% weight)
        if self.avg_response_time_ms < 2000:
            score += 0.2
        elif self.avg_response_time_ms < 5000:
            score += 0.1
        elif self.avg_response_time_ms < 10000:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def record_success(self, response_time_ms: float):
        """Record a successful fetch."""
        self.success_count += 1
        self.last_success = time.time()
        self.consecutive_failures = 0
        self.response_times.append(response_time_ms)
        
        # Update average
        if self.response_times:
            self.avg_response_time_ms = sum(self.response_times) / len(self.response_times)
        
        # Re-enable if was disabled
        if self.is_disabled and time.time() > self.disabled_until:
            self.is_disabled = False
            logger.info(f"Feed {self.name} re-enabled after recovery period")
    
    def record_failure(self, error: str):
        """Record a failed fetch."""
        self.failure_count += 1
        self.last_failure = time.time()
        self.consecutive_failures += 1
        self.last_error = error
        
        # Auto-disable after 5 consecutive failures
        if self.consecutive_failures >= 5:
            self.is_disabled = True
            self.disabled_until = time.time() + 300  # 5 minute cooldown
            logger.warning(f"Feed {self.name} disabled due to consecutive failures")


@dataclass
class NewsItem:
    """Parsed news item."""
    title: str
    link: str
    source: str
    published: float
    summary: str = ""
    sentiment_score: float = 0.0
    is_crisis: bool = False
    is_whale: bool = False
    is_hype: bool = False
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.title.encode()).hexdigest()


# =============================================================================
# REDUNDANT RSS AGGREGATOR
# =============================================================================

class RedundantRSSAggregator:
    """
    Redundant RSS feed aggregator with health monitoring.
    
    Features:
    - Multiple feed sources with priority levels
    - Automatic health tracking and failover
    - Duplicate detection
    - Rate limiting protection
    
    Usage:
        aggregator = RedundantRSSAggregator()
        items = aggregator.fetch_news()
        
        # Get health status
        health = aggregator.get_health_report()
    """
    
    def __init__(
        self,
        enable_primary: bool = True,
        enable_aggregators: bool = True,
        enable_alternative: bool = True,
        enable_social: bool = False,
        enable_macro: bool = True,
        max_items_per_source: int = 5,
        cache_ttl_seconds: int = 300
    ):
        self.max_items_per_source = max_items_per_source
        self.cache_ttl = cache_ttl_seconds
        
        # Build feed list based on enabled categories
        self.feeds: List[Dict] = []
        
        if enable_primary:
            self.feeds.extend(PRIMARY_SOURCES)
        if enable_aggregators:
            self.feeds.extend(AGGREGATOR_SOURCES)
        if enable_alternative:
            self.feeds.extend(ALTERNATIVE_SOURCES)
        if enable_social:
            self.feeds.extend(SOCIAL_SOURCES)
        if enable_macro:
            self.feeds.extend(MACRO_SOURCES)
        
        # Health tracking
        self._health: Dict[str, FeedHealth] = {}
        for feed in self.feeds:
            self._health[feed['url']] = FeedHealth(url=feed['url'], name=feed['name'])
        
        # Cache
        self._cache: List[NewsItem] = []
        self._cache_timestamp: float = 0.0
        self._seen_hashes: set = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'duplicates_filtered': 0,
            'last_fetch_time': 0.0
        }
        
        logger.info(f"RedundantRSSAggregator initialized with {len(self.feeds)} feeds")
    
    def fetch_news(self, force_refresh: bool = False) -> List[NewsItem]:
        """
        Fetch news from all healthy feeds.
        
        Args:
            force_refresh: Skip cache and fetch fresh
            
        Returns:
            List of NewsItem objects
        """
        now = time.time()
        
        # Check cache
        if not force_refresh and (now - self._cache_timestamp < self.cache_ttl):
            return self._cache
        
        with self._lock:
            self._stats['total_fetches'] += 1
            self._stats['last_fetch_time'] = now
            
            all_items = []
            
            # Fetch from all feeds in parallel
            import requests
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ]
            
            def fetch_single_feed(feed: Dict) -> List[NewsItem]:
                url = feed['url']
                health = self._health.get(url)
                
                # Skip disabled feeds
                if health and health.is_disabled:
                    return []
                
                try:
                    start_time = time.time()
                    headers = {
                        'User-Agent': user_agents[hash(url) % len(user_agents)],
                        'Accept': 'application/rss+xml, application/xml, text/xml'
                    }
                    
                    # AEGIS: Increased timeout to 15s for resilience
                    resp = requests.get(url, timeout=15.0, headers=headers)
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    if resp.status_code == 200:
                        # Parse RSS
                        try:
                            import feedparser
                            parsed = feedparser.parse(resp.content)
                            items = []
                            
                            for entry in parsed.entries[:self.max_items_per_source]:
                                item = NewsItem(
                                    title=entry.title,
                                    link=entry.link,
                                    source=feed['name'],
                                    published=time.time(),
                                    summary=getattr(entry, 'summary', '')
                                )
                                items.append(item)
                            
                            # Record success
                            if health:
                                health.record_success(response_time_ms)
                            
                            self._stats['successful_fetches'] += 1
                            return items
                            
                        except Exception as parse_error:
                            if health:
                                health.record_failure(f"Parse error: {parse_error}")
                            self._stats['failed_fetches'] += 1
                            return []
                    else:
                        if health:
                            health.record_failure(f"HTTP {resp.status_code}")
                        self._stats['failed_fetches'] += 1
                        return []
                        
                except Exception as e:
                    if health:
                        health.record_failure(str(e))
                    self._stats['failed_fetches'] += 1
                    return []
            
            # Parallel fetch
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(fetch_single_feed, feed): feed for feed in self.feeds}
                
                for future in as_completed(futures):
                    try:
                        items = future.result()
                        all_items.extend(items)
                    except Exception as exc:
                        feed = futures[future]
                        logger.error(f"Exception fetching {feed['name']}: {exc}")
            
            # Filter duplicates
            unique_items = []
            for item in all_items:
                if item.hash not in self._seen_hashes:
                    self._seen_hashes.add(item.hash)
                    unique_items.append(item)
                else:
                    self._stats['duplicates_filtered'] += 1
            
            # Sort by published time (newest first)
            unique_items.sort(key=lambda x: x.published, reverse=True)
            
            # Keep only last 100 items
            unique_items = unique_items[:100]
            
            # Update cache
            self._cache = unique_items
            self._cache_timestamp = now
            
            return unique_items
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        with self._lock:
            feeds_by_category = {}
            
            for feed in self.feeds:
                category = feed.get('category', 'unknown')
                if category not in feeds_by_category:
                    feeds_by_category[category] = []
                
                health = self._health.get(feed['url'])
                if health:
                    feeds_by_category[category].append({
                        'name': health.name,
                        'url': health.url,
                        'health_score': health.health_score,
                        'success_rate': health.success_rate,
                        'consecutive_failures': health.consecutive_failures,
                        'is_disabled': health.is_disabled,
                        'avg_response_time_ms': health.avg_response_time_ms
                    })
            
            # Calculate overall health
            total_feeds = len(self._health)
            healthy_feeds = sum(1 for h in self._health.values() if h.health_score > 0.7)
            disabled_feeds = sum(1 for h in self._health.values() if h.is_disabled)
            
            return {
                'timestamp': time.time(),
                'total_feeds': total_feeds,
                'healthy_feeds': healthy_feeds,
                'disabled_feeds': disabled_feeds,
                'overall_health': healthy_feeds / max(1, total_feeds),
                'categories': feeds_by_category,
                'statistics': self._stats.copy()
            }
    
    def get_healthy_feeds(self, min_health_score: float = 0.5) -> List[Dict]:
        """Get list of healthy feeds."""
        healthy = []
        for feed in self.feeds:
            health = self._health.get(feed['url'])
            if health and health.health_score >= min_health_score and not health.is_disabled:
                healthy.append(feed)
        return healthy
    
    def enable_feed(self, url: str):
        """Manually enable a feed."""
        if url in self._health:
            self._health[url].is_disabled = False
            logger.info(f"Feed {self._health[url].name} manually enabled")
    
    def disable_feed(self, url: str, duration_seconds: int = 300):
        """Manually disable a feed."""
        if url in self._health:
            self._health[url].is_disabled = True
            self._health[url].disabled_until = time.time() + duration_seconds
            logger.info(f"Feed {self._health[url].name} manually disabled")


# =============================================================================
# GLOBAL AGGREGATOR INSTANCE
# =============================================================================

_global_aggregator: Optional[RedundantRSSAggregator] = None


def get_global_rss_aggregator() -> RedundantRSSAggregator:
    """Get or create global RSS aggregator instance."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = RedundantRSSAggregator(
            enable_primary=True,
            enable_aggregators=True,
            enable_alternative=True,
            enable_social=False,
            enable_macro=True
        )
    return _global_aggregator


def initialize_rss_aggregator(
    enable_social: bool = False
) -> RedundantRSSAggregator:
    """Initialize global RSS aggregator with custom settings."""
    global _global_aggregator
    
    _global_aggregator = RedundantRSSAggregator(
        enable_primary=True,
        enable_aggregators=True,
        enable_alternative=True,
        enable_social=enable_social,
        enable_macro=True
    )
    
    logger.info("Global RSS Aggregator initialized")
    return _global_aggregator
