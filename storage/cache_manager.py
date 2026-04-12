import redis
import json
import logging
from typing import Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        try:
            self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            # ping to ensure connected
            self.client.ping()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {settings.REDIS_URL}: {e}")
            self.client = None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        if not self.client:
            return False

        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        try:
            return self.client.set(key, value, ex=ttl_seconds)
        except Exception as e:
            logger.error(f"Redis hit an error on SET: {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        if not self.client:
            return None

        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis hit an error on GET: {e}")
            return None

    def get_json(self, key: str) -> Optional[Any]:
        val = self.get(key)
        if val:
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
        return None

    def exists(self, key: str) -> bool:
        if not self.client:
            return False
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis hit an error on EXISTS: {e}")
            return False

    def delete(self, key: str) -> int:
        if not self.client:
            return 0
        try:
            return self.client.delete(key)
        except Exception:
            return 0


# Global instance
cache = CacheManager()
