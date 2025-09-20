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

logger = logging.getLogger(__name__)

class AdvancedCryptoAI:
    """Simple AI system using only available data - no complex dependencies"""
    
    def __init__(self):
        try:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
            self.scaler = StandardScaler()
            logger.info("🤖 Simple AI models initialized successfully")
        except Exception as e:
            logger.error(f"❌ AI model initialization failed: {e}")
            raise
        
        # Simple thresholds
        self.thresholds = {
            'whale_threshold': 0.4,
            'pump_threshold': 0.5,
            'anomaly_threshold': 0.2,
            'smart_money_threshold': 200
        }
        
        # Scoring weights
        self.weights = {
            'volume_score': 0.35,
            'wallet_quality': 0.25,
            'momentum': 0.20,
            'web3_bonus': 0.15,
            'risk_penalty': 0.05
        }
        
        # Web3 intelligence session
        self._web3_session = None
        
        logger.info("🚀 Simple AI initialized - using available data only")
    
    async def complete_ai_analysis_with_web3(self, purchases: List, analysis_type: str) -> Dict:
        """
        Simple AI analysis using only available data
        """
        try:
            logger.info(f"🤖 Simple AI Analysis: {len(purchases)} {analysis_type}")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Extract unique tokens for batch Web3 processing
            unique_tokens = self._extract_unique_tokens_info(purchases)
            logger.info(f"🔍 Found {len(unique_tokens)} unique tokens")
            
            # Step 2: Batch process Web3 intelligence with error handling
            try:
                web3_intelligence = await self._batch_process_web3_intelligence(unique_tokens)
            except Exception as e:
                logger.warning(f"Web3 intelligence failed: {e}, using defaults")
                web3_intelligence = {token: self._default_web3_intelligence(token, 'ethereum') 
                                for token in unique_tokens.keys()}
            
            # Step 3: Apply Web3 intelligence to all purchases
            self._apply_web3_intelligence_to_purchases(purchases, web3_intelligence)
            
            # Step 4: Run simple AI analysis
            try:
                result = await self._run_enhanced_ai_analysis(purchases, analysis_type)
            except Exception as e:
                logger.error(f"Enhanced AI analysis failed: {e}")
                result = await self._run_basic_ai_analysis(purchases, analysis_type)
            
            # Step 5: Add Web3 statistics
            result['web3_enriched_count'] = len(web3_intelligence)
            result['web3_enhanced'] = True
            
            logger.info(f"✅ Simple AI complete: {len(result.get('scores', {}))} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Simple AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _extract_unique_tokens_info(self, purchases: List) -> Dict:
        """Extract unique token information"""
        unique_tokens = {}
        
        for purchase in purchases:
            token = purchase.token_bought
            if token not in unique_tokens:
                contract_address = ""
                network = "ethereum"
                
                if purchase.web3_analysis:
                    contract_address = purchase.web3_analysis.get('contract_address', '') or purchase.web3_analysis.get('ca', '')
                    network = purchase.web3_analysis.get('network', 'ethereum')
                
                unique_tokens[token] = {
                    'contract_address': contract_address,
                    'network': network,
                    'token_symbol': token,
                    'purchase_count': 0,
                    'total_eth_value': 0
                }
            
            unique_tokens[token]['purchase_count'] += 1
            if hasattr(purchase, 'eth_spent'):
                unique_tokens[token]['total_eth_value'] += purchase.eth_spent
            elif hasattr(purchase, 'amount_received'):
                unique_tokens[token]['total_eth_value'] += purchase.amount_received
        
        return unique_tokens
    
    async def _batch_process_web3_intelligence(self, unique_tokens: Dict) -> Dict:
        """Simple Web3 intelligence processing"""
        web3_results = {}
        session = await self._get_web3_session()
        
        logger.info(f"🔍 Processing Web3 intelligence for {len(unique_tokens)} tokens")
        
        # Process in small batches
        batch_size = 3
        token_items = list(unique_tokens.items())
        
        for i in range(0, len(token_items), batch_size):
            batch = token_items[i:i + batch_size]
            
            tasks = []
            for token, token_info in batch:
                task = self._analyze_single_token_web3(session, token, token_info)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (token, token_info), result in zip(batch, batch_results):
                if isinstance(result, dict) and result:
                    web3_results[token] = result
                else:
                    web3_results[token] = self._default_web3_intelligence(token, token_info['network'])
            
            if i + batch_size < len(token_items):
                await asyncio.sleep(0.8)
        
        return web3_results
    
    async def _analyze_single_token_web3(self, session, token_symbol: str, token_info: Dict) -> Dict:
        """Simple Web3 analysis for a single token"""
        try:
            contract_address = token_info['contract_address']
            network = token_info['network']
            
            if not contract_address or len(contract_address) != 42:
                return self._apply_heuristic_intelligence(token_symbol, network)
            
            intelligence = {
                'contract_address': contract_address.lower(),
                'ca': contract_address.lower(),
                'token_symbol': token_symbol,
                'network': network,
                'is_verified': False,
                'has_liquidity': False,
                'liquidity_usd': 0,
                'honeypot_risk': 0.3,
                'data_sources': []
            }
            
            # Simple intelligence checks
            tasks = [
                self._check_contract_verification(session, contract_address, network),
                self._check_dexscreener_liquidity(session, contract_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result:
                    intelligence.update(result)
                    if result.get('source'):
                        intelligence['data_sources'].append(result['source'])
            
            if not intelligence['data_sources']:
                intelligence = self._apply_heuristic_intelligence(token_symbol, network)
            
            return intelligence
            
        except Exception as e:
            logger.debug(f"Web3 analysis failed for {token_symbol}: {e}")
            return self._apply_heuristic_intelligence(token_symbol, network)
    
    async def _check_contract_verification(self, session, contract_address: str, network: str) -> Dict:
        """Simple contract verification check"""
        try:
            # Placeholder for contract verification
            # You can implement Etherscan API calls here if you have the API key
            return {'is_verified': False, 'source': 'basic_check'}
        except Exception as e:
            return {}
    
    async def _check_dexscreener_liquidity(self, session, contract_address: str) -> Dict:
        """Simple DexScreener liquidity check"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        valid_pairs = [p for p in pairs if p.get('liquidity', {}).get('usd', 0) > 500]
                        
                        if valid_pairs:
                            best_pair = max(valid_pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                            liquidity_usd = float(best_pair.get('liquidity', {}).get('usd', 0))
                            
                            return {
                                'has_liquidity': liquidity_usd > 1000,
                                'liquidity_usd': liquidity_usd,
                                'source': 'dexscreener'
                            }
            
            return {}
        except Exception as e:
            return {}
    
    def _apply_heuristic_intelligence(self, token_symbol: str, network: str) -> Dict:
        """Apply simple heuristics for known tokens"""
        symbol_upper = token_symbol.upper()
        
        intelligence = {
            'contract_address': '',
            'ca': '',
            'token_symbol': token_symbol,
            'network': network,
            'is_verified': False,
            'has_liquidity': False,
            'liquidity_usd': 0,
            'honeypot_risk': 0.4,
            'data_sources': ['heuristic']
        }
        
        # Major tokens
        if symbol_upper in ['WETH', 'USDC', 'USDT', 'DAI', 'ETH']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.1
            })
        # DeFi tokens
        elif symbol_upper in ['UNI', 'AAVE', 'COMP', 'MKR', 'AERO']:
            intelligence.update({
                'is_verified': True,
                'has_liquidity': True,
                'honeypot_risk': 0.2
            })
        
        return intelligence
    
    def _apply_web3_intelligence_to_purchases(self, purchases: List, web3_intelligence: Dict):
        """Apply Web3 intelligence to all purchases"""
        for purchase in purchases:
            token = purchase.token_bought
            if token in web3_intelligence:
                if purchase.web3_analysis:
                    purchase.web3_analysis.update(web3_intelligence[token])
                else:
                    purchase.web3_analysis = web3_intelligence[token]
    
    async def _run_enhanced_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Run simple AI analysis"""
        try:
            df = self._create_enhanced_dataframe_with_web3(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            analyses = await self._run_core_analyses(df)
            enhanced_scores = self._create_enhanced_scores_with_web3(analyses, df)
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe_with_web3(self, purchases: List) -> pd.DataFrame:
        """Create DataFrame using only available data"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                row = {
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(getattr(purchase, 'amount_received', 0)),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    'is_verified': bool(web3_data.get('is_verified', False)),
                    'has_liquidity': bool(web3_data.get('has_liquidity', False)),
                    'liquidity_usd': float(web3_data.get('liquidity_usd', 0)),
                    'honeypot_risk': float(web3_data.get('honeypot_risk', 0.3)),
                    'token_age_hours': float(web3_data.get('token_age_hours', 999999)),
                    'data_sources_count': len(web3_data.get('data_sources', [])),
                }
                
                data.append(row)
                
            except Exception as e:
                logger.debug(f"Error processing purchase: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Add calculated features
        df['log_eth'] = np.log1p(df['eth_value'])
        
        if len(df) > 1:
            df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.8)
        else:
            df['is_whale'] = False
        
        df['is_smart_wallet'] = df['wallet_score'] > 200
        df['has_api_data'] = df['data_sources_count'] > 0
        df['is_high_liquidity'] = df['liquidity_usd'] > 10000
        df['is_new_token'] = df['token_age_hours'] < 168
        
        logger.info(f"🤖 Simple AI DataFrame: {len(df)} purchases")
        return df
    
    async def _run_core_analyses(self, df: pd.DataFrame) -> Dict:
        """Run simple core analyses"""
        analyses = {}
        
        analyses['whale_coordination'] = self._detect_whale_coordination(df)
        analyses['pump_signals'] = self._detect_pump_signals(df)
        analyses['smart_money_flow'] = self._analyze_smart_money_flow(df)
        analyses['momentum_analysis'] = self._analyze_momentum(df)
        analyses['risk_assessment'] = self._assess_risks(df)
        
        if len(df) > 10:
            analyses['anomaly_detection'] = self._detect_anomalies(df)
        else:
            analyses['anomaly_detection'] = {'detected': False, 'score': 0}
        
        return analyses
    
    def _detect_whale_coordination(self, df: pd.DataFrame) -> Dict:
        """Simple whale detection using ETH values and wallet scores"""
        if df.empty or len(df) < 2:
            return {'detected': False, 'score': 0}
        
        high_value_threshold = df['eth_value'].quantile(0.8)
        high_value_txs = len(df[df['eth_value'] > high_value_threshold])
        high_score_wallets = len(df[df['wallet_score'] > 200])
        
        whale_ratio = high_value_txs / len(df)
        quality_ratio = high_score_wallets / len(df)
        
        coordination_score = (whale_ratio + quality_ratio) / 2
        
        return {
            'detected': coordination_score > 0.4,
            'score': coordination_score
        }
    
    def _detect_pump_signals(self, df: pd.DataFrame) -> Dict:
        """Simple pump detection using volume, wallets, and token age"""
        if df.empty or len(df) < 3:
            return {'detected': False, 'score': 0}
        
        unique_wallets = df['wallet'].nunique()
        total_eth = df['eth_value'].sum()
        avg_wallet_score = df['wallet_score'].mean()
        
        new_token_bonus = 0
        if 'token_age_hours' in df.columns:
            new_tokens = len(df[df['token_age_hours'] < 168]) / len(df)
            new_token_bonus = 0.2 if new_tokens > 0.5 else 0
        
        volume_score = min(total_eth * 0.1, 0.3)
        wallet_score = min(unique_wallets * 0.05, 0.3)
        quality_score = min(avg_wallet_score / 1000, 0.2)
        
        pump_score = volume_score + wallet_score + quality_score + new_token_bonus
        
        return {
            'detected': pump_score > 0.5,
            'score': pump_score,
            'confidence': min(pump_score, 1.0)
        }
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Simple smart money analysis using wallet scores"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_wallets = df[df['wallet_score'] > 200]
        smart_volume = smart_wallets['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        if smart_ratio > 0.6:
            direction = 'STRONG_BULLISH'
        elif smart_ratio > 0.3:
            direction = 'BULLISH'
        elif smart_ratio < 0.1:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'flow_direction': direction,
            'confidence': smart_ratio,
            'smart_money_ratio': smart_ratio
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Simple momentum using transaction timing and frequency"""
        if df.empty or len(df) < 2:
            return {'momentum_detected': False, 'strength': 0}
        
        momentum_signals = {}
        
        for token, token_df in df.groupby('token'):
            if len(token_df) < 2:
                continue
            
            unique_wallets = token_df['wallet'].nunique()
            total_eth = token_df['eth_value'].sum()
            transaction_count = len(token_df)
            
            time_span_hours = (token_df['unix_time'].max() - token_df['unix_time'].min()) / 3600
            frequency = transaction_count / max(time_span_hours, 1)
            
            momentum_score = (
                min(total_eth * 0.1, 0.4) +
                min(unique_wallets * 0.05, 0.3) +
                min(frequency * 0.1, 0.2)
            )
            
            momentum_signals[token] = {
                'overall_score': momentum_score,
                'volume': total_eth,
                'wallets': unique_wallets,
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
            'strength': avg_momentum
        }
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Simple risk assessment using available data"""
        if df.empty:
            return {'overall_risk': 0.5, 'risk_level': 'MEDIUM'}
        
        risk_factors = {}
        
        # Concentration risk
        wallet_counts = df['wallet'].value_counts()
        max_wallet_txs = wallet_counts.max() if len(wallet_counts) > 0 else 0
        risk_factors['concentration'] = max_wallet_txs / len(df)
        
        # Low volume risk
        total_eth = df['eth_value'].sum()
        risk_factors['low_volume'] = 1.0 if total_eth < 0.1 else max(0, (1.0 - total_eth))
        
        # New token risk
        if 'token_age_hours' in df.columns:
            very_new_tokens = len(df[df['token_age_hours'] < 24]) / len(df)
            risk_factors['new_token'] = very_new_tokens
        else:
            risk_factors['new_token'] = 0.3
        
        # Timing risk
        unusual_hours = [0, 1, 2, 3, 4, 5]
        risk_factors['timing'] = len(df[df['hour'].isin(unusual_hours)]) / len(df)
        
        # Honeypot risk
        if 'honeypot_risk' in df.columns:
            risk_factors['honeypot'] = df['honeypot_risk'].mean()
        else:
            risk_factors['honeypot'] = 0.3
        
        overall_risk = (
            risk_factors['concentration'] * 0.25 +
            risk_factors['low_volume'] * 0.20 +
            risk_factors['new_token'] * 0.25 +
            risk_factors['timing'] * 0.15 +
            risk_factors['honeypot'] * 0.15
        )
        
        risk_level = 'HIGH' if overall_risk >= 0.7 else 'MEDIUM' if overall_risk >= 0.4 else 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Simple anomaly detection without complex ML"""
        if df.empty or len(df) < 5:
            return {'detected': False, 'anomaly_ratio': 0}
        
        try:
            eth_values = df['eth_value']
            
            q75_eth, q25_eth = np.percentile(eth_values, [75, 25])
            iqr_eth = q75_eth - q25_eth
            eth_outliers = len(eth_values[(eth_values < (q25_eth - 1.5 * iqr_eth)) | 
                                         (eth_values > (q75_eth + 1.5 * iqr_eth))])
            
            anomaly_ratio = eth_outliers / len(df)
            
            return {
                'detected': anomaly_ratio > 0.2,
                'anomaly_ratio': anomaly_ratio,
                'anomaly_count': eth_outliers
            }
            
        except Exception as e:
            return {'detected': False, 'anomaly_ratio': 0}
    
    def _create_enhanced_scores_with_web3(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores with Web3 bonuses"""
        enhanced_scores = {}
        
        for token, token_df in df.groupby('token'):
            try:
                total_eth = token_df['eth_value'].sum()
                unique_wallets = token_df['wallet'].nunique()
                avg_score = token_df['wallet_score'].mean()
                
                # Component scores
                volume_score = min(total_eth * 40, 50)
                diversity_score = min(unique_wallets * 8, 30)
                quality_score = min((avg_score / 200) * 20, 20)
                
                # Web3 bonuses
                web3_bonus = 0
                if token_df['is_verified'].any():
                    web3_bonus += 15
                if token_df['has_liquidity'].any():
                    web3_bonus += 10
                if token_df['is_high_liquidity'].any():
                    web3_bonus += 5
                if token_df['has_api_data'].any():
                    web3_bonus += 3
                
                # Risk penalty
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
                risk_penalty = avg_honeypot_risk * 15
                
                # AI multiplier
                ai_multiplier = 1.0
                if analyses['whale_coordination']['detected']:
                    ai_multiplier += analyses['whale_coordination']['score'] * 0.3
                if analyses['pump_signals']['detected']:
                    ai_multiplier += analyses['pump_signals']['score'] * 0.4
                
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
                    'is_verified': token_df['is_verified'].any(),
                    'has_liquidity': token_df['has_liquidity'].any(),
                    'liquidity_usd': token_df['liquidity_usd'].max(),
                    'honeypot_risk': avg_honeypot_risk
                }
                
                if web3_bonus > 0:
                    logger.info(f"🚀 {token}: Score={final_score:.1f} (Web3 bonus: +{web3_bonus})")
                
            except Exception as e:
                logger.debug(f"Error scoring token {token}: {e}")
                continue
        
        return enhanced_scores
    
    def _build_enhanced_result(self, analyses: Dict, enhanced_scores: Dict, analysis_type: str) -> Dict:
        """Build enhanced result"""
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
                'verified_tokens': sum(1 for s in enhanced_scores.values() if s.get('is_verified')),
                'tokens_with_liquidity': sum(1 for s in enhanced_scores.values() if s.get('has_liquidity'))
            }
        }
    
    async def _run_basic_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Basic AI analysis fallback"""
        try:
            scores = {}
            for purchase in purchases:
                token = purchase.token_bought
                if token not in scores:
                    eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                    score = min(eth_value * 20, 50)
                    scores[token] = {
                        'total_score': float(score),
                        'ai_enhanced': False,
                        'confidence': 0.7
                    }
            
            return {
                'scores': scores,
                'analysis_type': analysis_type,
                'enhanced': False,
                'web3_enhanced': False
            }
        except:
            return self._create_empty_result(analysis_type)
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Create empty result"""
        return {
            'scores': {},
            'analysis_type': analysis_type,
            'enhanced': False,
            'web3_enhanced': False,
            'error': 'No data to analyze'
        }
    
    async def _get_web3_session(self):
        """Get HTTP session for Web3 API calls"""
        if not self._web3_session:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            headers = {
                'User-Agent': 'CryptoAnalysis-AI/1.0',
                'Accept': 'application/json'
            }
            self._web3_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=8)
            )
        return self._web3_session
    
    def _default_web3_intelligence(self, token_symbol: str, network: str) -> Dict:
        """Default Web3 intelligence when APIs fail"""
        return {
            'contract_address': '',
            'ca': '',
            'token_symbol': token_symbol,
            'network': network,
            'is_verified': False,
            'has_liquidity': False,
            'liquidity_usd': 0,
            'honeypot_risk': 0.4,
            'data_sources': [],
            'error': 'no_data_available'
        }
    
    async def cleanup(self):
        """Cleanup Web3 session"""
        if self._web3_session:
            await self._web3_session.close()
            self._web3_session = None