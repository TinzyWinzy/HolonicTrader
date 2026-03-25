import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.agent_sentiment import SentimentHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
import config

def main():
    print("=== SENTIMENT & BIAS DIAGNOSTICS ===")
    
    # 1. Initialize Sentiment Holon
    print("\n[1] Initializing Sentiment Holon...")
    sentiment = SentimentHolon()
    print(f"Sources: {len(config.SENTIMENT_SOURCES)}")
    
    # 2. Fetch Sentiment
    print("\n[2] Fetching Sentiment (This may take a moment)...")
    try:
        score = sentiment.fetch_and_analyze()
        print(f"✅ Sentiment Score: {score:.4f}")
        print(f"Crisis Score: {getattr(sentiment, 'crisis_score', 0.0):.4f}")
        
        # Print details if available
        if hasattr(sentiment, 'news_cache'):
            print(f"Cached Headlines: {len(sentiment.news_cache)}")
            for i, news in enumerate(sentiment.news_cache[:5]):
                 print(f"  - [{news.get('sentiment', 0):.2f}] {news.get('title', 'No Title')[:50]}...")
    except Exception as e:
        print(f"❌ Sentiment Fetch Error: {e}")
        score = 0.0

    # 3. Initialize Oracle
    print("\n[3] Initializing Oracle for Bias Calculation...")
    oracle = EntryOracleHolon()
    
    # 4. Calculate Bias
    print("\n[4] Bias Calculation Details:")
    bias = oracle.get_market_bias(sentiment_score=score)
    print(f"✅ Global Market Bias: {bias:.4f}")
    print(f"Thresholds -> FairWeather: {getattr(config, 'FAIR_WEATHER_MIN_BIAS', 'N/A')}, GMB: {config.GMB_THRESHOLD}")
    
    # Inspect internal Oracle state contributing to bias if possible
    # (Oracle might need market state data to fully calc bias if it relies on more than just sentiment)
    # get_market_bias usually combines Sentiment + Internal Trend?
    # Let's check get_market_bias implementation in oracle if needed, or just trust the output.

if __name__ == "__main__":
    main()
