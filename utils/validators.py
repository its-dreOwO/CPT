from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionResult(BaseModel):
    """Schema for outputted predictions to be broadcast to clients/DB."""
    model_config = ConfigDict(populate_by_name=True)

    coin: str
    timestamp: datetime
    direction: str = Field(description="'bullish', 'bearish', or 'neutral'")
    magnitude_pct: float = Field(description="Expected percentage move (positive or negative)")
    confidence: float = Field(ge=0.0, le=1.0)
    
    target_24h: Optional[float] = None
    target_72h: Optional[float] = None
    target_7d: Optional[float] = None
    
    ensemble_weights_used: Optional[Dict[str, float]] = None
    
class NotificationConfig(BaseModel):
    """Schema representing toggles for notifications."""
    discord_enabled: bool = True
    whatsapp_enabled: bool = False
    zalo_enabled: bool = False
