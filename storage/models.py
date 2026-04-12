from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from storage.database import Base

class PriceData(Base):
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    
    direction = Column(String, nullable=False) # 'bullish', 'bearish', 'neutral'
    magnitude_pct = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    
    target_24h = Column(Float, nullable=True)
    target_72h = Column(Float, nullable=True)
    target_7d = Column(Float, nullable=True)
    
    # E.g., for quantile bands output from TimesFM, or model contribution weights
    metadata_json = Column(JSON, nullable=True)

class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    
    source = Column(String, nullable=False) # e.g. 'twitter', 'reddit', 'telegram', or 'aggregated'
    
    score_cryptobert = Column(Float, nullable=True)
    score_finbert = Column(Float, nullable=True)
    score_vader = Column(Float, nullable=True)
    
    # The final combined sentiment proxy metric inside [-1, 1]
    aggregated_score = Column(Float, nullable=True)
    
    # The total number of items processed to inform this score
    num_items = Column(Integer, default=0)
