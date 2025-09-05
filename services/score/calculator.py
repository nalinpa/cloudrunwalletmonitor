# Enhanced Alpha Score Calculator using Web3.py and AI Logic
# Add this as a new file: services/scoring/alpha_calculator.py

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from web3 import Web3
from dataclasses import dataclass
import statistics
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class TokenMetrics:
    """Comprehensive token metrics for AI scoring"""
    # Basic info
    symbol: str
    contract_address: str
    network: str
    
    # Trading metrics
    total_eth_volume: float
    unique_wallets: int
    transaction_count: int
    avg_wallet_score: float
    
    # Web3 metrics
    token_age_hours: Optional[float] = None
    holder_count: Optional[int] = None
    liquidity_eth: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    
    # Smart money metrics
    smart_money_percentage: Optional[float] = None
    whale_activity: Optional[float] = None
    recent_momentum: Optional[float] = None
    
    # Risk factors
    honeypot_risk: Optional[float] = None
    rug_risk: Optional[float] = None
    contract_verified: Optional[bool] = None

class EnhancedAlphaCalculator:
    """Enhanced alpha score calculator with Web3 and AI logic"""
    
    def __init__(self, config):
        self.config = config
        self.w3_ethereum = None
        self.w3_base = None
        self.session = None
        
        # AI Scoring weights (tunable)
        self.weights = {
            'volume_score': 0.25,      # ETH volume importance
            'wallet_quality': 0.20,    # Smart wallet quality
            'momentum': 0.15,          # Recent activity momentum
            'liquidity': 0.15,         # Liquidity depth
            'holder_distribution': 0.10, # Holder diversity
            'age_factor': 0.08,        # Token age consideration
            'risk_factor': 0.07        # Risk assessment
        }
    
    async def initialize(self):
        """Initialize Web3 connections and HTTP session"""
        try:
            # Initialize Web3 connections
            self.w3_ethereum = Web3(Web3.HTTPProvider(
                f"https://eth-mainnet.g.alchemy.com/v2/{self.config.alchemy_api_key}"
            ))
            
            self.w3_base = Web3(Web3.HTTPProvider(
                f"https://base-mainnet.g.alchemy.com/v2/{self.config.alchemy_api_key}"
            ))
            
            # HTTP session for API calls
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            
            logger.info("Enhanced Alpha Calculator initialized with Web3 connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Alpha Calculator: {e}")
            raise
    
    async def calculate_enhanced_alpha_score(self, token_metrics: TokenMetrics) -> Tuple[float, Dict]:
        """Calculate enhanced alpha score using AI logic and Web3 data"""
        try:
            logger.info(f"Calculating enhanced alpha score for {token_metrics.symbol}")
            
            # Enrich metrics with Web3 data
            enriched_metrics = await self._enrich_with_web3_data(token_metrics)
            
            # Calculate component scores using AI logic
            scores = await self._calculate_component_scores(enriched_metrics)
            
            # Apply AI weighting and normalization
            final_score = self._apply_ai_weighting(scores)
            
            # Add momentum and risk adjustments
            adjusted_score = self._apply_adjustments(final_score, enriched_metrics, scores)
            
            # Detailed breakdown for transparency
            breakdown = {
                'base_scores': scores,
                'weighted_score': final_score,
                'final_score': adjusted_score,
                'risk_factors': self._assess_risk_factors(enriched_metrics),
                'confidence_level': self._calculate_confidence(enriched_metrics)
            }
            
            logger.info(f"Enhanced alpha score for {token_metrics.symbol}: {adjusted_score:.2f}")
            return adjusted_score, breakdown
            
        except Exception as e:
            logger.error(f"Error calculating enhanced alpha score: {e}")
            # Fallback to basic scoring
            return self._fallback_score(token_metrics), {}
    
    async def _enrich_with_web3_data(self, metrics: TokenMetrics) -> TokenMetrics:
        """Enrich token metrics with Web3 blockchain data"""
        try:
            w3 = self.w3_ethereum if metrics.network.lower() == 'ethereum' else self.w3_base
            
            if not w3 or not w3.is_connected():
                logger.warning(f"Web3 not connected for {metrics.network}")
                return metrics
            
            # Get contract creation time (token age)
            metrics.token_age_hours = await self._get_token_age(w3, metrics.contract_address)
            
            # Get holder count and distribution
            holder_data = await self._get_holder_data(metrics.contract_address, metrics.network)
            metrics.holder_count = holder_data.get('holder_count')
            
            # Get liquidity information
            metrics.liquidity_eth = await self._get_liquidity_data(metrics.contract_address, metrics.network)
            
            # Get price and volume data
            price_data = await self._get_price_data(metrics.contract_address, metrics.network)
            metrics.price_change_24h = price_data.get('price_change_24h')
            metrics.volume_24h_usd = price_data.get('volume_24h')
            
            # Calculate smart money metrics
            metrics.smart_money_percentage = self._calculate_smart_money_percentage(metrics)
            metrics.whale_activity = await self._analyze_whale_activity(metrics)
            metrics.recent_momentum = self._calculate_momentum(metrics)
            
            # Risk assessment
            risk_data = await self._assess_contract_risk(w3, metrics.contract_address)
            metrics.honeypot_risk = risk_data.get('honeypot_risk', 0.0)
            metrics.contract_verified = risk_data.get('verified', False)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error enriching metrics with Web3 data: {e}")
            return metrics
    
    async def _get_token_age(self, w3: Web3, contract_address: str) -> Optional[float]:
        """Get token age in hours using Web3"""
        try:
            if not contract_address or len(contract_address) < 40:
                return None
            
            # Get contract creation transaction (simplified)
            current_block = w3.eth.block_number
            
            # Binary search for contract creation (approximate)
            creation_block = await self._find_creation_block(w3, contract_address, current_block)
            
            if creation_block:
                creation_timestamp = w3.eth.get_block(creation_block)['timestamp']
                age_seconds = datetime.now().timestamp() - creation_timestamp
                return age_seconds / 3600  # Convert to hours
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting token age: {e}")
            return None
    
    async def _find_creation_block(self, w3: Web3, contract_address: str, current_block: int) -> Optional[int]:
        """Find approximate contract creation block"""
        try:
            # Check if contract exists at current block
            code = w3.eth.get_code(Web3.to_checksum_address(contract_address))
            if not code or code == b'':
                return None
            
            # Simple approximation: assume contract is relatively new (last 100k blocks)
            search_start = max(0, current_block - 100000)
            
            # Quick check at midpoint
            mid_block = (search_start + current_block) // 2
            try:
                mid_code = w3.eth.get_code(Web3.to_checksum_address(contract_address), mid_block)
                if mid_code and mid_code != b'':
                    return mid_block  # Approximate creation
            except:
                pass
            
            return search_start  # Fallback
            
        except Exception as e:
            logger.debug(f"Error finding creation block: {e}")
            return None
    
    async def _get_holder_data(self, contract_address: str, network: str) -> Dict:
        """Get holder count and distribution data"""
        try:
            if not self.session:
                return {}
            
            # Try multiple APIs for holder data
            apis_to_try = [
                f"https://api.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={contract_address}&page=1&offset=100",
                # Add more APIs as needed
            ]
            
            for api_url in apis_to_try:
                try:
                    async with self.session.get(api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('status') == '1':
                                holders = data.get('result', [])
                                return {
                                    'holder_count': len(holders),
                                    'top_holder_percentage': self._calculate_concentration(holders)
                                }
                except:
                    continue
            
            return {}
            
        except Exception as e:
            logger.debug(f"Error getting holder data: {e}")
            return {}
    
    async def _get_liquidity_data(self, contract_address: str, network: str) -> Optional[float]:
        """Get liquidity data from DEX APIs"""
        try:
            if not self.session:
                return None
            
            # DexScreener API
            api_url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        # Get largest liquidity pool
                        largest_pool = max(pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                        liquidity_usd = float(largest_pool.get('liquidity', {}).get('usd', 0))
                        
                        # Convert to ETH (approximate)
                        eth_price = 2400  # Approximate ETH price
                        return liquidity_usd / eth_price
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting liquidity data: {e}")
            return None
    
    async def _get_price_data(self, contract_address: str, network: str) -> Dict:
        """Get price and volume data"""
        try:
            if not self.session:
                return {}
            
            # DexScreener API for price data
            api_url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        largest_pool = max(pairs, key=lambda x: float(x.get('volume', {}).get('h24', 0)))
                        
                        return {
                            'price_change_24h': float(largest_pool.get('priceChange', {}).get('h24', 0)),
                            'volume_24h': float(largest_pool.get('volume', {}).get('h24', 0))
                        }
            
            return {}
            
        except Exception as e:
            logger.debug(f"Error getting price data: {e}")
            return {}
    
    async def _calculate_component_scores(self, metrics: TokenMetrics) -> Dict[str, float]:
        """Calculate individual component scores using AI logic"""
        scores = {}
        
        # Volume Score (0-100)
        volume_score = min(metrics.total_eth_volume * 15, 100)  # 15 points per ETH
        scores['volume_score'] = volume_score
        
        # Wallet Quality Score (0-100)
        if metrics.avg_wallet_score > 0:
            wallet_quality = min((metrics.avg_wallet_score / 300) * 100, 100)  # Normalize to 100
        else:
            wallet_quality = 0
        scores['wallet_quality'] = wallet_quality
        
        # Momentum Score (0-100) - Based on unique wallets and recent activity
        momentum_base = min(metrics.unique_wallets * 8, 80)
        momentum_boost = 0
        
        if metrics.price_change_24h and metrics.price_change_24h > 0:
            momentum_boost += min(metrics.price_change_24h / 10, 20)  # Up to 20 points for price increase
        
        scores['momentum'] = min(momentum_base + momentum_boost, 100)
        
        # Liquidity Score (0-100)
        if metrics.liquidity_eth:
            liquidity_score = min(metrics.liquidity_eth * 2, 100)  # 2 points per ETH liquidity
        else:
            liquidity_score = 30  # Default moderate score
        scores['liquidity'] = liquidity_score
        
        # Holder Distribution Score (0-100)
        if metrics.holder_count:
            distribution_score = min(metrics.holder_count / 10, 100)  # 0.1 point per holder
        else:
            distribution_score = 50  # Default
        scores['holder_distribution'] = distribution_score
        
        # Age Factor (0-100) - Newer tokens get higher scores initially
        if metrics.token_age_hours:
            if metrics.token_age_hours < 24:  # Less than 1 day
                age_score = 90
            elif metrics.token_age_hours < 168:  # Less than 1 week
                age_score = 70
            elif metrics.token_age_hours < 720:  # Less than 1 month
                age_score = 50
            else:
                age_score = 30
        else:
            age_score = 60  # Default
        scores['age_factor'] = age_score
        
        # Risk Factor (0-100) - Higher is better (lower risk)
        risk_score = 80  # Default good score
        
        if metrics.honeypot_risk and metrics.honeypot_risk > 0.5:
            risk_score -= 40
        
        if metrics.contract_verified is False:
            risk_score -= 20
        
        scores['risk_factor'] = max(risk_score, 0)
        
        return scores
    
    def _apply_ai_weighting(self, scores: Dict[str, float]) -> float:
        """Apply AI-based weighting to component scores"""
        weighted_score = 0.0
        
        for component, score in scores.items():
            weight = self.weights.get(component, 0)
            weighted_score += score * weight
        
        return min(weighted_score, 100)
    
    def _apply_adjustments(self, base_score: float, metrics: TokenMetrics, scores: Dict) -> float:
        """Apply final AI adjustments based on patterns"""
        adjusted_score = base_score
        
        # Smart money concentration bonus
        if metrics.smart_money_percentage and metrics.smart_money_percentage > 0.8:
            adjusted_score += 10  # Bonus for high smart money participation
        
        # Volume momentum bonus
        if metrics.volume_24h_usd and metrics.volume_24h_usd > 100000:  # $100k+ volume
            adjusted_score += 5
        
        # Whale activity adjustment
        if metrics.whale_activity and metrics.whale_activity > 0.7:
            adjusted_score += 8  # Big players involved
        
        # Risk penalty
        if metrics.honeypot_risk and metrics.honeypot_risk > 0.3:
            adjusted_score -= 15  # Significant risk penalty
        
        # Quality floor and ceiling
        return max(min(adjusted_score, 100), 0)
    
    def _calculate_smart_money_percentage(self, metrics: TokenMetrics) -> float:
        """Calculate what percentage of activity comes from smart money"""
        if metrics.unique_wallets == 0:
            return 0.0
        
        # Simple heuristic: higher avg wallet score = more smart money
        if metrics.avg_wallet_score > 250:
            return 0.9  # Very high smart money
        elif metrics.avg_wallet_score > 200:
            return 0.7  # High smart money
        elif metrics.avg_wallet_score > 150:
            return 0.5  # Moderate smart money
        else:
            return 0.3  # Lower smart money
    
    async def _analyze_whale_activity(self, metrics: TokenMetrics) -> float:
        """Analyze whale activity patterns"""
        # Simplified whale detection based on ETH volume per wallet
        if metrics.unique_wallets > 0:
            avg_eth_per_wallet = metrics.total_eth_volume / metrics.unique_wallets
            
            if avg_eth_per_wallet > 5:  # 5+ ETH per wallet on average
                return 0.9  # High whale activity
            elif avg_eth_per_wallet > 2:
                return 0.6  # Moderate whale activity
            else:
                return 0.3  # Lower whale activity
        
        return 0.0
    
    def _calculate_momentum(self, metrics: TokenMetrics) -> float:
        """Calculate recent momentum score"""
        momentum = 0.0
        
        # Transaction density momentum
        if metrics.transaction_count > 20:
            momentum += 0.4
        elif metrics.transaction_count > 10:
            momentum += 0.2
        
        # Wallet diversity momentum
        if metrics.unique_wallets > 15:
            momentum += 0.3
        elif metrics.unique_wallets > 8:
            momentum += 0.2
        
        # Volume momentum
        if metrics.total_eth_volume > 10:
            momentum += 0.3
        elif metrics.total_eth_volume > 5:
            momentum += 0.2
        
        return min(momentum, 1.0)
    
    async def _assess_contract_risk(self, w3: Web3, contract_address: str) -> Dict:
        """Assess smart contract risk factors"""
        try:
            if not contract_address or len(contract_address) < 40:
                return {'honeypot_risk': 0.5, 'verified': False}
            
            # Get contract code
            code = w3.eth.get_code(Web3.to_checksum_address(contract_address))
            
            risk_factors = {
                'honeypot_risk': 0.0,
                'verified': len(code) > 100  # Simple check
            }
            
            # Simple heuristics for risk assessment
            if code:
                code_str = code.hex()
                
                # Check for suspicious patterns (simplified)
                if 'selfdestruct' in code_str.lower():
                    risk_factors['honeypot_risk'] += 0.3
                
                if len(code) < 1000:  # Very simple contracts might be risky
                    risk_factors['honeypot_risk'] += 0.2
            
            return risk_factors
            
        except Exception as e:
            logger.debug(f"Error assessing contract risk: {e}")
            return {'honeypot_risk': 0.3, 'verified': False}
    
    def _assess_risk_factors(self, metrics: TokenMetrics) -> Dict:
        """Assess various risk factors"""
        return {
            'honeypot_risk': metrics.honeypot_risk or 0.0,
            'age_risk': 'high' if (metrics.token_age_hours and metrics.token_age_hours < 1) else 'low',
            'liquidity_risk': 'high' if (metrics.liquidity_eth and metrics.liquidity_eth < 1) else 'low',
            'verification_risk': 'high' if not metrics.contract_verified else 'low'
        }
    
    def _calculate_confidence(self, metrics: TokenMetrics) -> float:
        """Calculate confidence level in the alpha score"""
        confidence = 0.5  # Base confidence
        
        # More data = higher confidence
        if metrics.token_age_hours is not None:
            confidence += 0.1
        if metrics.holder_count is not None:
            confidence += 0.1
        if metrics.liquidity_eth is not None:
            confidence += 0.1
        if metrics.price_change_24h is not None:
            confidence += 0.1
        if metrics.contract_verified is not None:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_concentration(self, holders: List) -> float:
        """Calculate holder concentration"""
        if not holders or len(holders) < 2:
            return 1.0
        
        # Simple concentration: top holder percentage
        total_supply = sum(float(h.get('TokenHolderQuantity', 0)) for h in holders)
        if total_supply > 0:
            top_holder = max(float(h.get('TokenHolderQuantity', 0)) for h in holders)
            return top_holder / total_supply
        
        return 0.5
    
    def _fallback_score(self, metrics: TokenMetrics) -> float:
        """Fallback scoring when Web3 data unavailable"""
        # Your existing scoring logic as fallback
        volume_score = min(metrics.total_eth_volume * 50, 50)
        diversity_score = min(metrics.unique_wallets * 8, 30)
        quality_score = min((metrics.avg_wallet_score / 100) * 20, 20)
        
        return volume_score + diversity_score + quality_score
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        logger.info("Enhanced Alpha Calculator cleaned up")