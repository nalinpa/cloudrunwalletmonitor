import logging
from typing import Optional
import httpx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PriceConverter:
    """Convert crypto values to USD"""
    
    def __init__(self):
        self.eth_price_cache = None
        self.cache_time = None
        self.cache_duration = timedelta(minutes=5)
    
    async def get_eth_price_usd(self) -> float:
        """Get current ETH price in USD with caching"""
        now = datetime.utcnow()
        
        # Return cached price if fresh
        if self.eth_price_cache and self.cache_time:
            if now - self.cache_time < self.cache_duration:
                return self.eth_price_cache
        
        try:
            # Try CoinGecko (free, no API key)
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": "ethereum", "vs_currencies": "usd"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    price = float(data["ethereum"]["usd"])
                    
                    # Cache the price
                    self.eth_price_cache = price
                    self.cache_time = now
                    
                    logger.info(f"ETH price updated: ${price:,.2f}")
                    return price
        
        except Exception as e:
            logger.warning(f"Failed to fetch ETH price: {e}")
        
        # Fallback to approximate price if API fails
        fallback = 3500.0
        logger.warning(f"Using fallback ETH price: ${fallback}")
        return fallback
    
    async def eth_to_usd(self, eth_amount: float) -> float:
        """Convert ETH amount to USD"""
        if eth_amount == 0:
            return 0.0
        
        eth_price = await self.get_eth_price_usd()
        return eth_amount * eth_price
    
    async def get_token_price_usd(self, token_address: str, network: str = "ethereum") -> Optional[float]:
        """Get token price in USD from DexScreener"""
        try:
            chain_id = "ethereum" if network == "ethereum" else "base"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("pairs"):
                        # Get the pair with highest liquidity
                        pairs = sorted(
                            data["pairs"],
                            key=lambda p: float(p.get("liquidity", {}).get("usd", 0)),
                            reverse=True
                        )
                        
                        if pairs:
                            price_usd = pairs[0].get("priceUsd")
                            if price_usd:
                                return float(price_usd)
        
        except Exception as e:
            logger.debug(f"Failed to get token price for {token_address}: {e}")
        
        return None

# Global instance
price_converter = PriceConverter()