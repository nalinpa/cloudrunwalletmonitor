import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.config import Config
from core.data.models import AnalysisResult, Purchase

# Import consolidated web3 utilities
from utils.web3_utils import (
    extract_unique_tokens_info,
    create_default_web3_intelligence,
    apply_token_heuristics,
    safe_float_conversion,
    safe_int_conversion,
    safe_bool_conversion,
    get_web3_service,
    cleanup_web3_service
)

logger = logging.getLogger(__name__)

class AdvancedCryptoAI:
    """Enhanced AI system with integrated batch Web3 intelligence processing - Using web3_utils"""
    
    def __init__(self):
        self.config = Config()
        try:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
            self.scaler = StandardScaler()
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"AI model initialization failed: {e}")
            raise
        
        # Simplified thresholds - easier to tune
        self.thresholds = {
            'whale_threshold': 0.7,
            'pump_threshold': 0.65,
            'anomaly_threshold': 0.3,
            'smart_money_threshold': 200,
            'min_liquidity_eth': 5.0
        }
        
        # Scoring weights - easily adjustable
        self.weights = {
            'volume_score': 0.30,
            'wallet_quality': 0.25,
            'momentum': 0.20,
            'liquidity': 0.15,
            'age_factor': 0.05,
            'risk_factor': 0.05
        }
        
        logger.info("Enhanced AI initialized with web3_utils integration")
    
    # ============================================================================
    # MAIN METHOD: AI Analysis with Integrated Web3 Intelligence
    # ============================================================================
    
    async def complete_ai_analysis_with_web3(self, purchases: List, analysis_type: str) -> Dict:
        """
        ENHANCED: AI analysis with integrated batch Web3 intelligence processing using web3_utils
        """
        try:
            logger.info(f"AI Analysis with web3_utils integration: {len(purchases)} {analysis_type}")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Extract unique tokens using web3_utils
            unique_tokens = extract_unique_tokens_info(purchases)
            logger.info(f"Found {len(unique_tokens)} unique tokens for batch Web3 analysis")
            
            # Step 2: Batch process Web3 intelligence using web3_utils
            try:
                web3_service = get_web3_service(self.config)
                web3_intelligence = await web3_service.batch_process_tokens(unique_tokens)
            except Exception as e:
                logger.warning(f"Web3 intelligence processing failed: {e}, using heuristics")
                web3_intelligence = {token: apply_token_heuristics(token, token_info['network']) 
                                   for token, token_info in unique_tokens.items()}
            
            # Step 3: Apply Web3 intelligence to all purchases
            self._apply_web3_intelligence_to_purchases(purchases, web3_intelligence)
            
            # Step 4: Run enhanced AI analysis with Web3 data
            try:
                result = await self._run_enhanced_ai_analysis(purchases, analysis_type)
            except KeyError as e:
                logger.error(f"KeyError in AI analysis: {e}")
                logger.info("Falling back to basic analysis")
                result = await self._run_basic_ai_analysis(purchases, analysis_type)
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                result = self._create_empty_result(analysis_type)
            
            # Step 5: Add Web3 statistics
            result['web3_enriched_count'] = len(web3_intelligence)
            result['web3_enhanced'] = True
            
            logger.info(f"AI with Web3 complete: {len(result.get('scores', {}))} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"AI analysis with Web3 failed: {e}")
            logger.error(f"Full error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_result(analysis_type)
    
    def _apply_web3_intelligence_to_purchases(self, purchases: List, web3_intelligence: Dict):
        """Apply Web3 intelligence to all purchases using web3_utils data"""
        for purchase in purchases:
            token = purchase.token_bought
            if token in web3_intelligence:
                intelligence = web3_intelligence[token]
                
                # Log Web3 data when found
                token_age = intelligence.get('token_age_hours')
                
                if token_age is not None:
                    age_days = token_age / 24
                    age_category = ("BRAND NEW" if age_days < 1 else "FRESH" if age_days < 7 else 
                                "RECENT" if age_days < 30 else "ESTABLISHED" if age_days < 365 else "MATURE")
                    logger.info(f"Token age data available for {token}: {token_age:.1f} hours ({age_days:.1f}d) - {age_category}")
                
                # Update purchase web3_analysis
                if purchase.web3_analysis:
                    existing_data = purchase.web3_analysis.copy()
                    existing_data.update(intelligence)
                    purchase.web3_analysis = existing_data
                else:
                    purchase.web3_analysis = intelligence.copy()
                
                # Ensure critical fields are properly stored using web3_utils safe conversions
                for field, field_type in [
                    ('token_age_hours', float),
                    ('is_verified', bool),
                    ('has_liquidity', bool),
                    ('liquidity_usd', float),
                    ('honeypot_risk', float)
                ]:
                    if field in intelligence and intelligence[field] is not None:
                        try:
                            if field_type == float:
                                purchase.web3_analysis[field] = safe_float_conversion(intelligence[field])
                            elif field_type == bool:
                                purchase.web3_analysis[field] = safe_bool_conversion(intelligence[field])
                            elif field_type == int:
                                purchase.web3_analysis[field] = safe_int_conversion(intelligence[field])
                        except Exception as e:
                            logger.error(f"Error converting {field} for {token}: {e}")
                
                logger.info(f"Web3 intelligence applied to {token}: {len(purchase.web3_analysis)} fields")
           
    async def _run_enhanced_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Run the enhanced AI analysis with Web3-enriched data"""
        try:
            # Step 1: Create enhanced DataFrame with Web3 data
            df = self._create_enhanced_dataframe_with_web3(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Step 2: Run core analyses
            analyses = await self._run_core_analyses(df)
            
            # Step 3: Create enhanced scores with Web3 bonuses
            enhanced_scores = self._create_enhanced_scores_with_web3(analyses, df)
            
            # Step 4: Build result
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe_with_web3(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame with Web3 intelligence data using web3_utils safe conversions"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = safe_float_conversion(getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0)))
                wallet_score = safe_float_conversion(getattr(purchase, 'sophistication_score', 0))
                
                # Extract Web3 intelligence with safe defaults using web3_utils
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # Safe extraction of Web3 fields with defaults using web3_utils functions
                is_verified = safe_bool_conversion(web3_data.get('is_verified', False))
                has_liquidity = safe_bool_conversion(web3_data.get('has_liquidity', False))
                liquidity_usd = safe_float_conversion(web3_data.get('liquidity_usd', 0))
                honeypot_risk = safe_float_conversion(web3_data.get('honeypot_risk', 0.3))
                
                # Safe token age extraction
                token_age_hours = safe_float_conversion(web3_data.get('token_age_hours'), 999999)  # Default to old token
                
                # Safe smart money extraction with proper defaults
                smart_money_buying = safe_bool_conversion(web3_data.get('smart_money_buying', False))
                whale_accumulation = safe_bool_conversion(web3_data.get('whale_accumulation', False))
                
                # Core data with Web3 enhancements
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': eth_value,
                    'amount': safe_float_conversion(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': wallet_score,
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 intelligence with SAFE defaults
                    'is_verified': is_verified,
                    'has_liquidity': has_liquidity,
                    'liquidity_usd': liquidity_usd,
                    'honeypot_risk': honeypot_risk,
                    'smart_money_buying': smart_money_buying,
                    'whale_accumulation': whale_accumulation,
                    'has_coingecko_listing': safe_bool_conversion(web3_data.get('has_coingecko_listing', False)),
                    'data_sources_count': len(web3_data.get('data_sources', [])),                    
                    'token_age_hours': token_age_hours
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase for AI: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features with SAFE operations
        df['log_eth'] = np.log1p(df['eth_value'])
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > 200
        df['has_api_data'] = df['data_sources_count'] > 0
        df['is_high_liquidity'] = df['liquidity_usd'] > 10000
        
        # Safe new token detection
        df['is_new_token'] = df['token_age_hours'] < 168  # Less than 1 week
        
        # Create has_smart_money column safely
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        # Log Web3 intelligence statistics
        verified_count = df['is_verified'].sum()
        liquidity_count = df['has_liquidity'].sum()
        api_data_count = df['has_api_data'].sum()
        new_token_count = df['is_new_token'].sum()
        smart_money_count = df['has_smart_money'].sum()
        
        logger.info(f"AI DataFrame with Web3: {len(df)} purchases")
        logger.info(f"Verified: {verified_count}/{len(df)} ({verified_count/len(df)*100:.1f}%)")
        logger.info(f"With liquidity: {liquidity_count}/{len(df)} ({liquidity_count/len(df)*100:.1f}%)")
        logger.info(f"API data: {api_data_count}/{len(df)} ({api_data_count/len(df)*100:.1f}%)")
        logger.info(f"New tokens: {new_token_count}/{len(df)} ({new_token_count/len(df)*100:.1f}%)")
        logger.info(f"Smart money: {smart_money_count}/{len(df)} ({smart_money_count/len(df)*100:.1f}%)")
        
        return df

    def _create_enhanced_scores_with_web3(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores with Web3 intelligence bonuses"""
        enhanced_scores = {}
        
        # Group by token for scoring
        for token, token_df in df.groupby('token'):
            try:
                # Basic metrics
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['wallet_score'].mean()
                
                # Component scores
                volume_score = min(total_eth * 40, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 200) * 20, 20)
                
                # WEB3 INTELLIGENCE BONUSES
                web3_bonus = 0
                
                # Verification bonus (major impact)
                if token_df['is_verified'].any():
                    web3_bonus += 15
                    logger.debug(f"{token}: +15 points for verified contract")
                
                # Liquidity bonus
                if token_df['has_liquidity'].any():
                    web3_bonus += 10
                    logger.debug(f"{token}: +10 points for liquidity")
                
                # High liquidity bonus
                if token_df['is_high_liquidity'].any():
                    web3_bonus += 5
                    logger.debug(f"{token}: +5 points for high liquidity")
                
                # API data bonus
                if token_df['has_api_data'].any():
                    web3_bonus += 3
                    logger.debug(f"{token}: +3 points for API data")
                
                # CoinGecko listing bonus
                if token_df['has_coingecko_listing'].any():
                    web3_bonus += 7
                    logger.debug(f"{token}: +7 points for CoinGecko listing")
                
                # Risk penalty
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
                risk_penalty = avg_honeypot_risk * 20
                
                if risk_penalty > 3:
                    logger.debug(f"{token}: -{risk_penalty:.1f} points for risk")
                
                # AI multiplier from analyses
                ai_multiplier = 1.0
                if analyses.get('whale_coordination', {}).get('detected'):
                    ai_multiplier += analyses['whale_coordination']['score'] * 0.3
                if analyses.get('pump_signals', {}).get('detected'):
                    ai_multiplier += analyses['pump_signals']['score'] * 0.4
                
                # Final score calculation
                base_score = volume_score + diversity_score + quality_score + web3_bonus - risk_penalty
                final_score = base_score * ai_multiplier
                
                # Store enhanced score data
                enhanced_scores[token] = {
                    'total_score': final_score,
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'web3_bonus': web3_bonus,
                    'risk_penalty': risk_penalty,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'confidence': min(0.95, 0.6 + (ai_multiplier - 1) * 0.2),
                    
                    # Web3 intelligence metadata
                    'is_verified': token_df['is_verified'].any(),
                    'has_liquidity': token_df['has_liquidity'].any(),
                    'liquidity_usd': token_df['liquidity_usd'].max(),
                    'honeypot_risk': avg_honeypot_risk,
                    'has_api_data': token_df['has_api_data'].any(),
                    'has_coingecko_listing': token_df['has_coingecko_listing'].any()
                }
                
                if web3_bonus > 0:
                    logger.info(f"{token}: Total={final_score:.1f} (Web3 bonus: +{web3_bonus})")
                
            except Exception as e:
                logger.debug(f"Error scoring token {token}: {e}")
                continue
        
        return enhanced_scores
    
    # ============================================================================
    # CORE AI ANALYSIS METHODS
    # ============================================================================
    
    async def complete_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Core AI analysis method"""
        try:
            logger.info(f"AI ANALYSIS: {len(purchases)} {analysis_type} transactions")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Create enhanced DataFrame
            df = self._create_enhanced_dataframe(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Step 2: Run core AI analyses
            analyses = await self._run_core_analyses(df)
            
            # Step 3: Create enhanced scores
            enhanced_scores = self._create_enhanced_scores(analyses, df)
            
            # Step 4: Build comprehensive result
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            logger.info(f"AI SUCCESS: {len(enhanced_scores)} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame - basic version using web3_utils safe conversions"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = safe_float_conversion(getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0)))
                wallet_score = safe_float_conversion(getattr(purchase, 'sophistication_score', 0))
                
                # Extract Web3 data efficiently
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # Core data with Web3 enhancements using web3_utils safe conversions
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': eth_value,
                    'amount': safe_float_conversion(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': wallet_score,
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 data (with defaults)
                    'token_age_hours': safe_float_conversion(web3_data.get('token_age_hours'), 999999),
                    'is_verified': safe_bool_conversion(web3_data.get('is_verified', False)),
                    'honeypot_risk': safe_float_conversion(web3_data.get('honeypot_risk', 0)),
                    'smart_money_buying': safe_bool_conversion(web3_data.get('smart_money_buying', False)),
                    'whale_accumulation': safe_bool_conversion(web3_data.get('whale_accumulation', False)),
                    'has_liquidity': safe_bool_conversion(web3_data.get('has_liquidity', False))
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features efficiently
        df['log_eth'] = np.log1p(df['eth_value'])
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > self.thresholds['smart_money_threshold']
        df['is_new_token'] = df['token_age_hours'] < 168  # 1 week
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        return df
    
    async def _run_core_analyses(self, df: pd.DataFrame) -> Dict:
        """Run core AI analyses - streamlined but comprehensive"""
        analyses = {}
        
        # Whale coordination detection
        analyses['whale_coordination'] = self._detect_whale_coordination(df)
        
        # Pump signal detection
        analyses['pump_signals'] = self._detect_pump_signals(df)
        
        # Smart money flow analysis
        analyses['smart_money_flow'] = self._analyze_smart_money_flow(df)
        
        # Momentum analysis
        analyses['momentum_analysis'] = self._analyze_momentum(df)
        
        # Risk assessment
        analyses['risk_assessment'] = self._assess_risks(df)
        
        # Anomaly detection (if enough data)
        if len(df) > 10:
            analyses['anomaly_detection'] = self._detect_anomalies(df)
        else:
            analyses['anomaly_detection'] = {'detected': False, 'score': 0}
        
        return analyses
    
    def _detect_whale_coordination(self, df: pd.DataFrame) -> Dict:
        """Detect whale coordination - simplified but effective"""
        if df.empty or len(df) < 3:
            return {'detected': False, 'score': 0}
        
        # Group by token for analysis
        coordination_scores = []
        
        for token, token_df in df.groupby('token'):
            if len(token_df) < 2:
                continue
            
            # Whale metrics
            whale_transactions = token_df[token_df['is_whale']]
            whale_ratio = len(whale_transactions) / len(token_df)
            
            # Timing coordination (transactions close in time)
            if len(token_df) > 1:
                time_diffs = token_df['unix_time'].diff().dropna()
                avg_time_diff = time_diffs.mean() / 3600  # Convert to hours
                timing_score = 1.0 / (1.0 + avg_time_diff) if avg_time_diff > 0 else 0
            else:
                timing_score = 0
            
            # Wallet quality correlation
            high_score_ratio = len(token_df[token_df['is_smart_wallet']]) / len(token_df)
            
            # Combined coordination score
            coord_score = (whale_ratio * 0.4 + timing_score * 0.3 + high_score_ratio * 0.3)
            coordination_scores.append(coord_score)
        
        avg_coordination = np.mean(coordination_scores) if coordination_scores else 0
        detected = avg_coordination >= self.thresholds['whale_threshold']
        
        return {
            'detected': detected,
            'score': avg_coordination,
            'evidence_strength': 'STRONG' if avg_coordination > 0.8 else 'MODERATE' if avg_coordination > 0.5 else 'WEAK'
        }
    
    def _detect_pump_signals(self, df: pd.DataFrame) -> Dict:
        """Detect pump signals"""
        if df.empty or len(df) < 5:
            return {'detected': False, 'score': 0}
        
        df_sorted = df.sort_values('unix_time')
        
        # Volume acceleration
        mid_point = len(df_sorted) // 2
        early_volume = df_sorted.iloc[:mid_point]['eth_value'].sum()
        recent_volume = df_sorted.iloc[mid_point:]['eth_value'].sum()
        volume_acceleration = recent_volume / early_volume if early_volume > 0 else 1
        
        # Wallet growth
        early_wallets = df_sorted.iloc[:mid_point]['wallet'].nunique()
        recent_wallets = df_sorted.iloc[mid_point:]['wallet'].nunique()
        wallet_growth = recent_wallets / early_wallets if early_wallets > 0 else 1
        
        # Smart money involvement
        smart_ratio = len(df[df['is_smart_wallet']]) / len(df)
        
        # New token ratio
        new_token_ratio = len(df[df['is_new_token']]) / len(df) if 'is_new_token' in df.columns else 0
        
        # Smart money activity
        smart_money_active = 0.15 if df['has_smart_money'].any() else 0
        
        # Calculate pump score
        pump_score = (
            min((volume_acceleration - 1) * 0.3, 0.3) +
            min((wallet_growth - 1) * 0.2, 0.2) +
            min(smart_ratio * 0.2, 0.2) +
            (0.15 if new_token_ratio > 0.5 else 0) +
            smart_money_active
        )
        
        detected = pump_score >= 0.65
        
        return {
            'detected': detected,
            'score': pump_score,
            'volume_acceleration': volume_acceleration,
            'wallet_growth': wallet_growth,
            'phase': self._get_pump_phase(pump_score),
            'confidence': min(pump_score * 1.2, 1.0)
        }
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze smart money flow"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_trades = df[df['is_smart_wallet']]
        smart_volume = smart_trades['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        # Smart money signals check
        smart_money_buying = df['smart_money_buying'].sum() if 'smart_money_buying' in df.columns else 0
        whale_accumulation = df['whale_accumulation'].sum() if 'whale_accumulation' in df.columns else 0
        
        # Determine flow direction
        if smart_ratio > 0.4 or smart_money_buying > 0:
            direction = 'STRONG_BULLISH' if smart_ratio > 0.6 else 'BULLISH'
        elif smart_ratio < 0.05:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'flow_direction': direction,
            'confidence': min(smart_ratio * 2, 1.0),
            'smart_money_ratio': smart_ratio,
            'smart_wallet_count': len(smart_trades),
            'whale_accumulation_detected': whale_accumulation > 0
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum - simplified but effective"""
        if df.empty or len(df) < 3:
            return {'momentum_detected': False, 'strength': 0}
        
        momentum_signals = {}
        
        for token, token_df in df.groupby('token'):
            if len(token_df) < 2:
                continue
            
            token_df = token_df.sort_values('unix_time').reset_index(drop=True)
            
            # Volume momentum
            if len(token_df) >= 4:
                mid = len(token_df) // 2
                early_vol = token_df.iloc[:mid]['eth_value'].mean()
                recent_vol = token_df.iloc[mid:]['eth_value'].mean()
                vol_momentum = recent_vol / early_vol if early_vol > 0 else 1
            else:
                vol_momentum = 1
            
            # Wallet momentum
            wallet_momentum = token_df['wallet'].nunique() / len(token_df)
            
            # Time momentum (frequency)
            time_span = (token_df['unix_time'].max() - token_df['unix_time'].min()) / 3600  # hours
            frequency = len(token_df) / max(time_span, 1)
            
            # Combined momentum
            momentum_score = (
                min((vol_momentum - 1) * 0.4, 0.4) +
                min(wallet_momentum * 0.3, 0.3) +
                min(frequency * 0.1, 0.3)
            )
            
            momentum_signals[token] = {
                'overall_score': momentum_score,
                'volume_momentum': vol_momentum,
                'wallet_momentum': wallet_momentum,
                'frequency': frequency
            }
        
        if momentum_signals:
            avg_momentum = np.mean([s['overall_score'] for s in momentum_signals.values()])
            momentum_detected = avg_momentum > 0.3
        else:
            avg_momentum = 0
            momentum_detected = False
        
        return {
            'momentum_detected': momentum_detected,
            'strength': avg_momentum,
            'token_momentum': momentum_signals
        }
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assess risks - comprehensive but efficient"""
        if df.empty:
            return {'overall_risk': 0.5, 'risk_level': 'MEDIUM'}
        
        risk_factors = {}
        
        # Concentration risk
        wallet_volumes = df.groupby('wallet')['eth_value'].sum()
        risk_factors['concentration'] = wallet_volumes.max() / wallet_volumes.sum() if wallet_volumes.sum() > 0 else 0
        
        # Timing risk (unusual hours)
        unusual_hours = [0, 1, 2, 3, 4, 5]
        risk_factors['timing'] = len(df[df['hour'].isin(unusual_hours)]) / len(df)
        
        # Honeypot risk
        risk_factors['honeypot'] = df['honeypot_risk'].mean()
        
        # New token risk
        risk_factors['new_token'] = len(df[df['is_new_token']]) / len(df)
        
        # Liquidity risk
        risk_factors['no_liquidity'] = len(df[~df['has_liquidity']]) / len(df)
        
        # Calculate overall risk
        overall_risk = (
            risk_factors['concentration'] * 0.25 +
            risk_factors['timing'] * 0.15 +
            risk_factors['honeypot'] * 0.25 +
            risk_factors['new_token'] * 0.15 +
            risk_factors['no_liquidity'] * 0.20
        )
        
        risk_level = 'HIGH' if overall_risk >= 0.7 else 'MEDIUM' if overall_risk >= 0.4 else 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies using ML - simplified approach"""
        try:
            # Prepare features for anomaly detection
            features = df[['eth_value', 'wallet_score', 'hour']].copy()
            features['log_eth'] = np.log1p(features['eth_value'])
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features[['log_eth', 'wallet_score', 'hour']])
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
            anomaly_ratio = (anomaly_scores == -1).sum() / len(anomaly_scores)
            
            detected = anomaly_ratio > self.thresholds['anomaly_threshold']
            
            return {
                'detected': detected,
                'anomaly_ratio': anomaly_ratio,
                'anomaly_count': (anomaly_scores == -1).sum()
            }
            
        except Exception as e:
            logger.debug(f"Anomaly detection failed: {e}")
            return {'detected': False, 'anomaly_ratio': 0}
    
    def _create_enhanced_scores(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores - streamlined approach using web3_utils safe conversions"""
        enhanced_scores = {}
        
        # Group by token for scoring
        for token, token_df in df.groupby('token'):
            try:
                # Basic metrics using safe conversions
                total_eth = safe_float_conversion(token_df['eth_value'].sum())
                unique_wallets = safe_int_conversion(token_df['wallet'].nunique())
                avg_score = safe_float_conversion(token_df['wallet_score'].mean())
                
                # Component scores
                volume_score = min(total_eth * 40, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 200) * 20, 20)
                
                # Web3 enhancements using safe conversions
                web3_bonus = 0
                if safe_bool_conversion(token_df['is_verified'].any()):
                    web3_bonus += 5
                if safe_bool_conversion(token_df['has_liquidity'].any()):
                    web3_bonus += 5
                if safe_bool_conversion(token_df['has_smart_money'].any()):
                    web3_bonus += 10
                
                # Risk penalty using safe conversion
                avg_honeypot_risk = safe_float_conversion(token_df['honeypot_risk'].mean())
                risk_penalty = avg_honeypot_risk * 15
                
                # AI multiplier
                ai_multiplier = 1.0
                if analyses['whale_coordination']['detected']:
                    ai_multiplier += analyses['whale_coordination']['score'] * 0.3
                if analyses['pump_signals']['detected']:
                    ai_multiplier += analyses['pump_signals']['score'] * 0.4
                
                # Final score
                base_score = volume_score + diversity_score + quality_score + web3_bonus - risk_penalty
                final_score = base_score * ai_multiplier
                
                enhanced_scores[token] = {
                    'total_score': final_score,
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'web3_bonus': web3_bonus,
                    'risk_penalty': risk_penalty,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'confidence': min(0.9, 0.6 + (ai_multiplier - 1) * 0.2),
                    'honeypot_risk': avg_honeypot_risk,
                    'is_verified': safe_bool_conversion(token_df['is_verified'].any()),
                    'has_liquidity': safe_bool_conversion(token_df['has_liquidity'].any()),
                    'has_smart_money': safe_bool_conversion(token_df['has_smart_money'].any())
                }
                
            except Exception as e:
                logger.debug(f"Error scoring token {token}: {e}")
                continue
        
        return enhanced_scores
    
    def _build_enhanced_result(self, analyses: Dict, enhanced_scores: Dict, analysis_type: str) -> Dict:
        """Build enhanced result - comprehensive"""
        return {
            'scores': enhanced_scores,
            'analysis_type': analysis_type,
            'enhanced': True,
            'web3_enhanced': True,
            'ai_analyses': analyses,
            'analysis_summary': {
                'total_tokens': len(enhanced_scores),
                'ai_patterns_detected': sum(1 for analysis in analyses.values() 
                                          if isinstance(analysis, dict) and analysis.get('detected', False)),
                'avg_confidence': np.mean([s['confidence'] for s in enhanced_scores.values()]) if enhanced_scores else 0,
                'high_confidence_tokens': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.85),
                'verified_tokens': sum(1 for s in enhanced_scores.values() if s.get('is_verified')),
                'tokens_with_liquidity': sum(1 for s in enhanced_scores.values() if s.get('has_liquidity')),
                'honeypot_warnings': sum(1 for s in enhanced_scores.values() if s.get('honeypot_risk', 0) > 0.5),
                'smart_money_tokens': sum(1 for s in enhanced_scores.values() if s.get('has_smart_money'))
            }
        }
    
    def _get_pump_phase(self, score: float) -> str:
        """Determine pump phase"""
        if score >= 0.8:
            return "PUMP_IMMINENT"
        elif score >= 0.6:
            return "LATE_ACCUMULATION" 
        elif score >= 0.4:
            return "EARLY_ACCUMULATION"
        else:
            return "NORMAL"
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Create empty result"""
        return {
            'scores': {},
            'analysis_type': analysis_type,
            'enhanced': False,
            'web3_enhanced': False,
            'error': 'No data to analyze'
        }
    
    # ============================================================================
    # FALLBACK BASIC AI ANALYSIS (when enhanced fails)
    # ============================================================================
    
    async def _run_basic_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Fallback basic analysis using web3_utils heuristics"""
        try:
            logger.info(f"Running basic AI analysis with web3_utils heuristics")
            
            # Create basic DataFrame
            df = self._create_enhanced_dataframe(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Basic scoring using web3_utils safe conversions
            scores = {}
            for token in df['token'].unique():
                token_df = df[df['token'] == token]
                
                # Basic metrics using safe conversions
                total_eth = safe_float_conversion(token_df['eth_value'].sum())
                unique_wallets = safe_int_conversion(token_df['wallet'].nunique())
                avg_score = safe_float_conversion(token_df['wallet_score'].mean())
                
                # Simple scoring
                volume_score = min(total_eth * 50, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 100) * 20, 20)
                
                # Web3 heuristic bonus using safe conversions
                heuristic_bonus = 0
                if safe_bool_conversion(token_df['is_verified'].any()):
                    heuristic_bonus += 10
                if safe_bool_conversion(token_df['has_liquidity'].any()):
                    heuristic_bonus += 8
                
                total_score = volume_score + diversity_score + quality_score + heuristic_bonus
                
                scores[token] = {
                    'total_score': total_score,
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'web3_bonus': heuristic_bonus,
                    'ai_enhanced': False,
                    'confidence': 0.7,
                    'is_verified': safe_bool_conversion(token_df['is_verified'].any()),
                    'has_liquidity': safe_bool_conversion(token_df['has_liquidity'].any())
                }
            
            return {
                'scores': scores,
                'analysis_type': analysis_type,
                'enhanced': False,
                'web3_enhanced': True,
                'fallback_mode': True
            }
            
        except Exception as e:
            logger.error(f"Basic AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    # ============================================================================
    # CLEANUP
    # ============================================================================
    
    async def cleanup(self):
        """Cleanup AI system resources including web3_utils service"""
        try:
            # Cleanup web3_utils global service
            await cleanup_web3_service()
            logger.info("AI system cleanup completed with web3_utils service cleanup")
        except Exception as e:
            logger.error(f"Error during AI system cleanup: {e}")

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS (for backward compatibility)
# =============================================================================

# These functions are kept for backward compatibility but now use web3_utils internally

def extract_contract_address_legacy(transfer: Dict) -> str:
    """Legacy function - now uses web3_utils"""
    from utils.web3_utils import extract_contract_address
    return extract_contract_address(transfer)

def is_excluded_token_legacy(asset: str, contract_address: str = None) -> bool:
    """Legacy function - now uses web3_utils"""
    from utils.web3_utils import is_excluded_token
    return is_excluded_token(asset, contract_address)

def calculate_eth_spent_legacy(outgoing_transfers: List[Dict], target_tx: str, target_block: str) -> float:
    """Legacy function - now uses web3_utils"""
    from utils.web3_utils import calculate_eth_spent
    return calculate_eth_spent(outgoing_transfers, target_tx, target_block)

def calculate_eth_received_legacy(incoming_transfers: List[Dict], target_tx: str, target_block: str) -> float:
    """Legacy function - now uses web3_utils"""
    from utils.web3_utils import calculate_eth_received
    return calculate_eth_received(incoming_transfers, target_tx, target_block)

def parse_timestamp_legacy(transfer: Dict, block_number: int = None) -> datetime:
    """Legacy function - now uses web3_utils"""
    from utils.web3_utils import parse_timestamp
    return parse_timestamp(transfer, block_number)