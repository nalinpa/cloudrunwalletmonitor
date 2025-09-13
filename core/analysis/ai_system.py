import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedCryptoAI:
    """Enhanced AI system with Web3 data integration"""
    
    def __init__(self):
        # Core ML models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        
        # Memory cache
        self.cache = {}
        
        # Enhanced AI thresholds with Web3 data
        self.thresholds = {
            'whale_coordination': 0.7,
            'pump_probability': 0.65,
            'anomaly_threshold': 0.3,
            'sentiment_threshold': 0.6,
            'smart_money_threshold': 200,
            'honeypot_risk_threshold': 0.5,
            'min_holder_count': 50,
            'min_liquidity_eth': 5.0,
            'max_token_age_hours': 168  # 1 week for "new" tokens
        }
        
        # Enhanced sentiment with Web3 terms
        self.bullish_keywords = [
            'moon', 'pump', 'bullish', 'rocket', 'gem', 'alpha', 'breakout',
            'rally', 'surge', 'explosion', 'parabolic', 'rocket', 'massive', 'huge'
        ]
        
        self.bearish_keywords = [
            'dump', 'crash', 'bearish', 'fall', 'drop', 'sell', 'exit',
            'rug', 'scam', 'dead', 'rekt', 'liquidated', 'down', 'honeypot',
            'trap', 'fake', 'avoid', 'warning', 'danger'
        ]
        
        logger.info("🚀 AI initialized with Web3 Integration")
    
    async def complete_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Complete AI analysis with Web3 data integration"""
        try:
            logger.info(f"🤖 AI ANALYSIS with Web3: {len(purchases)} {analysis_type} transactions")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Enhanced DataFrame with Web3 data
            df = self._create_web3_enhanced_dataframe(purchases)
            
            # AI ANALYSES with Web3 data
            analyses = {
                'whale_coordination': self._detect_whale_coordination(df),
                'pump_signals': self._detect_pump_signals_web3(df),
                'anomaly_detection': self._detect_anomalies(df),
                'sentiment_analysis': await self._analyze_sentiment(df),
                'smart_money_flow': self._analyze_smart_money_flow_web3(df),
                'momentum_analysis': self._analyze_momentum_with_web3(df),
                'risk_assessment': self._assess_risks_web3(df),
                'honeypot_detection': self._detect_honeypots_web3(df),
                'liquidity_analysis': self._analyze_liquidity_web3(df),
                'holder_analysis': self._analyze_holders_web3(df)
            }
            
            # Enhanced scoring with Web3 factors
            enhanced_scores = self._create_web3_enhanced_scores(analyses, df)
            
            # Build result with Web3 insights
            result = self._build_web3_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            logger.info(f"✅ AI+Web3 SUCCESS: {len(enhanced_scores)} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI+Web3 analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_result(analysis_type)
    
    def _create_web3_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create DataFrame with Web3 enriched data"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                # Extract Web3 data from purchase
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                data.append({
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'minute': timestamp.minute,
                    'day_of_week': timestamp.weekday(),
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 enriched data
                    'token_age_hours': web3_data.get('token_age_hours', 999999),
                    'holder_count': web3_data.get('holder_count', 0),
                    'total_supply': web3_data.get('total_supply', 0),
                    'is_verified': web3_data.get('is_verified', False),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0),
                    'smart_money_buying': web3_data.get('smart_money_buying', False),
                    'smart_money_selling': web3_data.get('smart_money_selling', False),
                    'whale_accumulation': web3_data.get('whale_accumulation', False),
                    'liquidity_pools_count': len(web3_data.get('liquidity_pools', [])),
                    'has_liquidity': len(web3_data.get('liquidity_pools', [])) > 0,
                    'contract_verified': web3_data.get('is_verified', False),
                    'decimals': web3_data.get('decimals', 18)
                })
            except Exception as e:
                logger.warning(f"Error processing purchase with Web3 data: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Enhanced features with Web3 data
        df['log_eth'] = np.log1p(df['eth_value'])
        df['eth_rank'] = df['eth_value'].rank(pct=True)
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > self.thresholds['smart_money_threshold']
        df['time_rank'] = df['unix_time'].rank(method='dense')
        
        # Web3-based features
        df['is_new_token'] = df['token_age_hours'] < self.thresholds['max_token_age_hours']
        df['is_low_holder'] = df['holder_count'] < self.thresholds['min_holder_count']
        df['is_high_risk'] = df['honeypot_risk'] > self.thresholds['honeypot_risk_threshold']
        df['has_smart_money'] = df['smart_money_buying'] | df['whale_accumulation']
        
        return df
    
    def _detect_pump_signals_web3(self, df: pd.DataFrame) -> Dict:
        """Enhanced pump detection with Web3 data"""
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
        
        # Web3 factors
        new_token_ratio = len(df[df['is_new_token']]) / len(df)
        verified_ratio = len(df[df['contract_verified']]) / len(df)
        smart_money_signal = df['has_smart_money'].any()
        avg_honeypot_risk = df['honeypot_risk'].mean()
        
        # Calculate enhanced pump score
        pump_score = (
            min((volume_acceleration - 1) * 0.3, 0.3) +
            min((wallet_growth - 1) * 0.2, 0.2) +
            min(smart_ratio * 0.2, 0.2) +
            (0.15 if new_token_ratio > 0.5 else 0) +  # New token bonus
            (0.1 if smart_money_signal else 0) +      # Smart money bonus
            (0.05 if verified_ratio > 0.5 else 0) -    # Verification bonus
            (avg_honeypot_risk * 0.2)                  # Risk penalty
        )
        
        detected = pump_score >= self.thresholds['pump_probability']
        
        return {
            'detected': detected,
            'score': pump_score,
            'volume_acceleration': volume_acceleration,
            'wallet_growth': wallet_growth,
            'smart_money_ratio': smart_ratio,
            'new_token_pump': new_token_ratio > 0.7,
            'smart_money_pump': smart_money_signal,
            'honeypot_risk': avg_honeypot_risk,
            'phase': self._get_pump_phase_web3(pump_score, df),
            'confidence': min(pump_score * 1.3, 1.0)
        }
    
    def _analyze_smart_money_flow_web3(self, df: pd.DataFrame) -> Dict:
        """Enhanced smart money analysis with Web3 data"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_trades = df[df['is_smart_wallet']]
        web3_smart = df[df['has_smart_money']]
        
        smart_volume = smart_trades['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        # Web3 smart money signals
        smart_money_buying = df['smart_money_buying'].sum()
        smart_money_selling = df['smart_money_selling'].sum()
        whale_accumulation = df['whale_accumulation'].sum()
        
        # Combined analysis
        web3_signal = 'NEUTRAL'
        if smart_money_buying > smart_money_selling * 2:
            web3_signal = 'BULLISH'
        elif smart_money_selling > smart_money_buying * 2:
            web3_signal = 'BEARISH'
        
        # Flow direction with Web3 data
        if smart_ratio > 0.4 or web3_signal == 'BULLISH':
            direction = 'STRONG_BULLISH'
        elif smart_ratio > 0.25:
            direction = 'BULLISH'
        elif smart_ratio < 0.05 or web3_signal == 'BEARISH':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'flow_direction': direction,
            'confidence': min(smart_ratio * 2, 1.0),
            'smart_money_ratio': smart_ratio,
            'smart_wallet_count': len(smart_trades),
            'web3_smart_money_signal': web3_signal,
            'whale_accumulation_detected': whale_accumulation > 0,
            'smart_money_buying_count': smart_money_buying,
            'smart_money_selling_count': smart_money_selling
        }
    
    def _detect_honeypots_web3(self, df: pd.DataFrame) -> Dict:
        """Detect potential honeypots using Web3 data"""
        if df.empty:
            return {'detected': False, 'risk_level': 'LOW', 'confidence': 0}
        
        # Aggregate honeypot risk scores
        avg_risk = df['honeypot_risk'].mean()
        max_risk = df['honeypot_risk'].max()
        high_risk_count = len(df[df['is_high_risk']])
        
        # Check other risk factors
        low_holders = df['is_low_holder'].any()
        no_liquidity = ~df['has_liquidity'].any()
        not_verified = ~df['contract_verified'].any()
        
        # Calculate overall honeypot probability
        risk_score = (
            avg_risk * 0.4 +
            (0.2 if low_holders else 0) +
            (0.2 if no_liquidity else 0) +
            (0.1 if not_verified else 0) +
            (0.1 if high_risk_count > len(df) * 0.3 else 0)
        )
        
        if risk_score >= 0.7:
            risk_level = 'HIGH'
        elif risk_score >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        detected = risk_score >= self.thresholds['honeypot_risk_threshold']
        
        return {
            'detected': detected,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': min(0.5 + avg_risk, 1.0),
            'avg_honeypot_risk': avg_risk,
            'max_honeypot_risk': max_risk,
            'risk_factors': {
                'low_holders': low_holders,
                'no_liquidity': no_liquidity,
                'not_verified': not_verified,
                'high_risk_transactions': high_risk_count
            }
        }
    
    def _analyze_liquidity_web3(self, df: pd.DataFrame) -> Dict:
        """Analyze liquidity using Web3 data"""
        if df.empty:
            return {'liquidity_status': 'UNKNOWN', 'score': 0}
        
        # Liquidity metrics
        has_liquidity_ratio = df['has_liquidity'].sum() / len(df)
        avg_pools = df['liquidity_pools_count'].mean()
        
        # Token age vs liquidity correlation
        new_tokens_with_liquidity = df[df['is_new_token'] & df['has_liquidity']]
        new_token_liquidity_ratio = len(new_tokens_with_liquidity) / max(len(df[df['is_new_token']]), 1)
        
        # Calculate liquidity score
        liquidity_score = (
            has_liquidity_ratio * 0.5 +
            min(avg_pools / 3, 1) * 0.3 +
            new_token_liquidity_ratio * 0.2
        )
        
        if liquidity_score >= 0.7:
            status = 'STRONG'
        elif liquidity_score >= 0.4:
            status = 'MODERATE'
        elif liquidity_score >= 0.1:
            status = 'WEAK'
        else:
            status = 'NONE'
        
        return {
            'liquidity_status': status,
            'score': liquidity_score,
            'has_liquidity_ratio': has_liquidity_ratio,
            'avg_pool_count': avg_pools,
            'new_tokens_with_liquidity': new_token_liquidity_ratio,
            'confidence': min(0.5 + has_liquidity_ratio * 0.5, 1.0)
        }
    
    def _analyze_holders_web3(self, df: pd.DataFrame) -> Dict:
        """Analyze holder distribution using Web3 data"""
        if df.empty:
            return {'holder_status': 'UNKNOWN', 'score': 0}
        
        # Holder metrics
        avg_holders = df['holder_count'].mean()
        min_holders = df['holder_count'].min()
        max_holders = df['holder_count'].max()
        
        # Calculate holder score
        holder_score = 0
        if avg_holders >= 1000:
            holder_score = 1.0
            status = 'EXCELLENT'
        elif avg_holders >= 500:
            holder_score = 0.8
            status = 'GOOD'
        elif avg_holders >= 100:
            holder_score = 0.6
            status = 'MODERATE'
        elif avg_holders >= 50:
            holder_score = 0.4
            status = 'LOW'
        else:
            holder_score = 0.2
            status = 'VERY_LOW'
        
        # Check for holder growth (if we have time series)
        holder_growth = 'UNKNOWN'
        if len(df) > 1:
            early_holders = df.iloc[:len(df)//2]['holder_count'].mean()
            recent_holders = df.iloc[len(df)//2:]['holder_count'].mean()
            if recent_holders > early_holders * 1.2:
                holder_growth = 'GROWING'
            elif recent_holders < early_holders * 0.8:
                holder_growth = 'DECLINING'
            else:
                holder_growth = 'STABLE'
        
        return {
            'holder_status': status,
            'score': holder_score,
            'avg_holders': avg_holders,
            'min_holders': min_holders,
            'max_holders': max_holders,
            'holder_growth': holder_growth,
            'low_holder_warning': min_holders < self.thresholds['min_holder_count']
        }
    
    def _analyze_momentum_with_web3(self, df: pd.DataFrame) -> Dict:
        """Enhanced momentum analysis with Web3 data"""
        if df.empty or len(df) < 5:
            return {'momentum_detected': False, 'strength': 0}
        
        try:
            momentum_signals = {}
            
            for token, token_df in df.groupby('token'):
                if len(token_df) < 3:
                    continue
                
                # Sort by time
                token_df = token_df.sort_values('unix_time').reset_index(drop=True)
                
                # Basic momentum
                token_momentum = {}
                
                # Volume acceleration
                mid_point = len(token_df) // 2
                early_volume = token_df.iloc[:mid_point]['eth_value'].sum()
                recent_volume = token_df.iloc[mid_point:]['eth_value'].sum()
                volume_acceleration = recent_volume / early_volume if early_volume > 0 else 1
                token_momentum['volume_acceleration'] = volume_acceleration
                
                # Transaction frequency
                time_span = token_df['unix_time'].max() - token_df['unix_time'].min()
                tx_frequency = len(token_df) / max(time_span / 3600, 1)
                token_momentum['tx_frequency'] = tx_frequency
                
                # Wallet growth
                early_wallets = token_df.iloc[:mid_point]['wallet'].nunique()
                recent_wallets = token_df.iloc[mid_point:]['wallet'].nunique()
                wallet_growth = recent_wallets / early_wallets if early_wallets > 0 else 1
                token_momentum['wallet_growth'] = wallet_growth
                
                # Web3 momentum factors
                has_smart_money = token_df['has_smart_money'].any()
                is_new_token = token_df['is_new_token'].iloc[-1]
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
                holder_momentum = 0
                
                if len(token_df) > 1:
                    holder_change = token_df['holder_count'].iloc[-1] - token_df['holder_count'].iloc[0]
                    holder_momentum = holder_change / max(token_df['holder_count'].iloc[0], 1)
                
                # Enhanced momentum score with Web3 factors
                momentum_score = (
                    min((volume_acceleration - 1) * 0.25, 0.25) +
                    min(tx_frequency * 0.1, 0.15) +
                    min((wallet_growth - 1) * 0.2, 0.2) +
                    (0.15 if has_smart_money else 0) +
                    (0.1 if is_new_token else 0) +
                    min(holder_momentum * 0.15, 0.15) -
                    (avg_honeypot_risk * 0.1)
                )
                
                token_momentum['overall_score'] = momentum_score
                token_momentum['has_smart_money'] = has_smart_money
                token_momentum['holder_momentum'] = holder_momentum
                token_momentum['risk_adjusted'] = momentum_score * (1 - avg_honeypot_risk)
                
                momentum_signals[token] = token_momentum
            
            # Calculate overall momentum
            if momentum_signals:
                avg_momentum = np.mean([s['overall_score'] for s in momentum_signals.values()])
                momentum_detected = avg_momentum > 0.5
            else:
                avg_momentum = 0
                momentum_detected = False
            
            return {
                'momentum_detected': momentum_detected,
                'strength': avg_momentum,
                'token_momentum': momentum_signals,
                'total_tokens_analyzed': len(momentum_signals),
                'smart_money_momentum': sum(1 for s in momentum_signals.values() if s.get('has_smart_money', False))
            }
            
        except Exception as e:
            logger.error(f"Momentum analysis with Web3 failed: {e}")
            return {'momentum_detected': False, 'strength': 0}
    
    def _assess_risks_web3(self, df: pd.DataFrame) -> Dict:
        """Comprehensive risk assessment with Web3 data"""
        if df.empty:
            return {'overall_risk': 0.5, 'risk_level': 'MEDIUM'}
        
        risk_factors = {}
        
        # Concentration risk
        wallet_volumes = df.groupby('wallet')['eth_value'].sum()
        top_wallet_share = wallet_volumes.max() / wallet_volumes.sum() if wallet_volumes.sum() > 0 else 0
        risk_factors['concentration'] = top_wallet_share
        
        # Timing risk
        unusual_hours = [0, 1, 2, 3, 4, 5]
        unusual_count = len(df[df['hour'].isin(unusual_hours)])
        timing_risk = unusual_count / len(df)
        risk_factors['timing'] = timing_risk
        
        # Volume risk
        median_volume = df['eth_value'].median()
        tiny_trades = len(df[df['eth_value'] < median_volume * 0.01])
        volume_risk = tiny_trades / len(df)
        risk_factors['volume'] = volume_risk
        
        # Web3 risks
        risk_factors['honeypot'] = df['honeypot_risk'].mean()
        risk_factors['low_holders'] = len(df[df['is_low_holder']]) / len(df)
        risk_factors['no_liquidity'] = len(df[~df['has_liquidity']]) / len(df)
        risk_factors['not_verified'] = len(df[~df['contract_verified']]) / len(df)
        risk_factors['new_token'] = len(df[df['is_new_token']]) / len(df)
        
        # Calculate weighted overall risk
        overall_risk = (
            risk_factors['concentration'] * 0.15 +
            risk_factors['timing'] * 0.1 +
            risk_factors['volume'] * 0.1 +
            risk_factors['honeypot'] * 0.25 +
            risk_factors['low_holders'] * 0.15 +
            risk_factors['no_liquidity'] * 0.15 +
            risk_factors['not_verified'] * 0.05 +
            risk_factors['new_token'] * 0.05
        )
        
        if overall_risk >= 0.7:
            risk_level = 'HIGH'
        elif overall_risk >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'top_risks': sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _create_web3_enhanced_scores(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Create enhanced scores with Web3 data weighting"""
        enhanced_scores = {}
        
        # Basic token stats
        token_stats = df.groupby('token').agg({
            'eth_value': ['sum', 'mean', 'count'],
            'wallet': 'nunique',
            'wallet_score': 'mean',
            'holder_count': 'mean',
            'honeypot_risk': 'mean',
            'token_age_hours': 'min',
            'has_liquidity': 'any',
            'contract_verified': 'any',
            'has_smart_money': 'any'
        }).round(4)
        
        token_stats.columns = ['total_value', 'mean_value', 'tx_count', 'unique_wallets', 
                              'avg_score', 'avg_holders', 'avg_honeypot_risk', 
                              'token_age', 'has_liquidity', 'is_verified', 'has_smart_money']
        
        for token in token_stats.index:
            try:
                stats = token_stats.loc[token]
                
                # Basic score components
                volume_score = min(stats['total_value'] * 40, 40)
                diversity_score = min(stats['unique_wallets'] * 6, 20)
                quality_score = min((stats['avg_score'] / 300) * 20, 20)
                
                # Web3 score components
                holder_score = min(stats['avg_holders'] / 100, 1) * 10
                liquidity_score = 10 if stats['has_liquidity'] else 0
                verification_score = 5 if stats['is_verified'] else 0
                age_penalty = 0 if stats['token_age'] < 168 else -5  # Penalty for old tokens
                risk_penalty = stats['avg_honeypot_risk'] * -20
                
                basic_score = volume_score + diversity_score + quality_score
                web3_score = holder_score + liquidity_score + verification_score + age_penalty + risk_penalty
                
                # AI MULTIPLIER with Web3 factors
                ai_multiplier = 1.0
                
                # Standard AI boosts
                whale_coord = analyses['whale_coordination']
                if whale_coord['detected']:
                    ai_multiplier += whale_coord['score'] * 0.3
                
                pump_signals = analyses['pump_signals']
                if pump_signals['detected']:
                    ai_multiplier += pump_signals['score'] * 0.4
                
                # Web3 AI boosts
                honeypot = analyses['honeypot_detection']
                if honeypot['detected']:
                    ai_multiplier *= (1 - honeypot['risk_score'] * 0.5)  # Major penalty
                
                liquidity = analyses['liquidity_analysis']
                if liquidity['liquidity_status'] == 'STRONG':
                    ai_multiplier += 0.2
                elif liquidity['liquidity_status'] == 'NONE':
                    ai_multiplier *= 0.7  # Penalty
                
                holders = analyses['holder_analysis']
                if holders['holder_status'] == 'EXCELLENT':
                    ai_multiplier += 0.15
                elif holders['holder_status'] in ['LOW', 'VERY_LOW']:
                    ai_multiplier *= 0.8  # Penalty
                
                smart_money = analyses['smart_money_flow']
                if smart_money['flow_direction'] in ['BULLISH', 'STRONG_BULLISH']:
                    if smart_money.get('whale_accumulation_detected'):
                        ai_multiplier += 0.3
                
                # Final enhanced score
                ultimate_score = (basic_score + web3_score) * ai_multiplier
                confidence = min(0.95, 0.6 + (ai_multiplier - 1) * 0.2 + (0.1 if stats['has_liquidity'] else 0))
                
                enhanced_scores[token] = {
                    'total_score': ultimate_score,
                    'basic_score': basic_score,
                    'web3_score': web3_score,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'web3_enhanced': True,
                    'confidence': confidence,
                    
                    # Component scores
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    'holder_score': holder_score,
                    'liquidity_score': liquidity_score,
                    'verification_score': verification_score,
                    
                    # Web3 data
                    'holder_count': stats['avg_holders'],
                    'honeypot_risk': stats['avg_honeypot_risk'],
                    'token_age_hours': stats['token_age'],
                    'has_liquidity': stats['has_liquidity'],
                    'is_verified': stats['is_verified'],
                    'has_smart_money': stats['has_smart_money'],
                    
                    # AI analysis results
                    'whale_coordination': whale_coord['score'],
                    'pump_probability': pump_signals['score'],
                    'momentum_score': analyses['momentum_analysis'].get('token_momentum', {}).get(token, {}).get('overall_score', 0),
                    'sentiment_score': analyses['sentiment_analysis'].get('token_sentiments', {}).get(token, {}).get('combined_score', 0),
                    'smart_money_score': smart_money['confidence'],
                    'risk_score': 1 - analyses['risk_assessment']['overall_risk'],
                    
                    # Detailed AI results
                    'whale_evidence': whale_coord.get('evidence_strength', 'NONE'),
                    'pump_phase': pump_signals.get('phase', 'NORMAL'),
                    'liquidity_status': liquidity.get('liquidity_status', 'UNKNOWN'),
                    'holder_status': holders.get('holder_status', 'UNKNOWN'),
                    'honeypot_risk_level': honeypot.get('risk_level', 'UNKNOWN'),
                    'smart_money_direction': smart_money.get('flow_direction', 'NEUTRAL'),
                    'risk_level': analyses['risk_assessment'].get('risk_level', 'MEDIUM'),
                    
                    # AI metadata
                    'ai_features_detected': self._count_ai_features_web3(analyses, token, stats)
                }
                
            except Exception as e:
                logger.error(f"Error creating enhanced score for {token}: {e}")
                continue
        
        return enhanced_scores
    
    def _count_ai_features_web3(self, analyses: Dict, token: str, stats) -> int:
        """Count AI and Web3 features detected for this token"""
        features = 0
        
        # Standard AI features
        if analyses['whale_coordination']['detected']:
            features += 1
        if analyses['pump_signals']['detected']:
            features += 1
        if analyses['momentum_analysis']['momentum_detected']:
            features += 1
        if analyses['sentiment_analysis'].get('sentiment_detected'):
            features += 1
        if analyses['smart_money_flow']['flow_direction'] != 'NEUTRAL':
            features += 1
        if analyses['anomaly_detection']['detected']:
            features += 1
        
        # Web3 features
        if analyses['honeypot_detection']['detected']:
            features += 1
        if analyses['liquidity_analysis']['liquidity_status'] in ['STRONG', 'MODERATE']:
            features += 1
        if analyses['holder_analysis']['holder_status'] in ['EXCELLENT', 'GOOD']:
            features += 1
        if stats['has_smart_money']:
            features += 1
        if stats['is_verified']:
            features += 1
        if stats['has_liquidity']:
            features += 1
        
        return features
    
    def _get_pump_phase_web3(self, score: float, df: pd.DataFrame) -> str:
        """Get pump phase with Web3 considerations"""
        # Check Web3 indicators
        avg_honeypot = df['honeypot_risk'].mean() if not df.empty else 0
        has_liquidity = df['has_liquidity'].any() if not df.empty else False
        new_token_ratio = len(df[df['is_new_token']]) / len(df) if not df.empty else 0
        
        # Adjust phase based on Web3 data
        if avg_honeypot > 0.7:
            return "HONEYPOT_WARNING"
        elif not has_liquidity:
            return "NO_LIQUIDITY"
        elif score >= 0.7 and new_token_ratio > 0.8:
            return "NEW_TOKEN_PUMP"
        elif score >= 0.7:
            return "PUMP_IMMINENT"
        elif score >= 0.5:
            return "LATE_ACCUMULATION"
        elif score >= 0.3:
            return "EARLY_ACCUMULATION"
        else:
            return "NORMAL"
    
    def _build_web3_enhanced_result(self, analyses: Dict, enhanced_scores: Dict, analysis_type: str) -> Dict:
        """Build result with Web3 enhancements"""
        return {
            'token_stats': None,
            'scores': enhanced_scores,
            'analysis_type': analysis_type,
            'enhanced': True,
            'web3_enhanced': True,
            
            # AI analysis with Web3
            'ai_analyses': analyses,
            
            # Enhanced summary statistics
            'analysis_summary': {
                'total_tokens': len(enhanced_scores),
                'ai_patterns_detected': sum(1 for analysis in analyses.values() 
                                           if isinstance(analysis, dict) and analysis.get('detected', False)),
                'avg_confidence': np.mean([s['confidence'] for s in enhanced_scores.values()]) if enhanced_scores else 0,
                'high_confidence_tokens': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.85),
                'ultra_high_confidence': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.9),
                
                # Web3 specific stats
                'verified_tokens': sum(1 for s in enhanced_scores.values() if s.get('is_verified')),
                'tokens_with_liquidity': sum(1 for s in enhanced_scores.values() if s.get('has_liquidity')),
                'honeypot_warnings': sum(1 for s in enhanced_scores.values() if s.get('honeypot_risk', 0) > 0.5),
                'smart_money_tokens': sum(1 for s in enhanced_scores.values() if s.get('has_smart_money')),
                'new_tokens': sum(1 for s in enhanced_scores.values() if s.get('token_age_hours', 999) < 168),
                
                # Risk stats
                'high_risk_tokens': sum(1 for s in enhanced_scores.values() if s.get('risk_level') == 'HIGH'),
                'low_holder_warnings': sum(1 for s in enhanced_scores.values() if s.get('holder_count', 0) < 50),
                
                # AI detections
                'pump_signals': sum(1 for s in enhanced_scores.values() if s.get('pump_phase') not in ['NORMAL', None]),
                'whale_coordination_detected': analyses['whale_coordination']['detected'],
                'honeypot_detected': analyses['honeypot_detection']['detected'],
                'strong_liquidity_count': sum(1 for s in enhanced_scores.values() if s.get('liquidity_status') == 'STRONG')
            },
            
            # Web3 insights
            'web3_insights': {
                'honeypot_analysis': analyses.get('honeypot_detection', {}),
                'liquidity_analysis': analyses.get('liquidity_analysis', {}),
                'holder_analysis': analyses.get('holder_analysis', {}),
                'smart_money_flow': analyses.get('smart_money_flow', {})
            },
            
            # Enhancement info
            'enhancement_info': {
                'ai_engine': 'Advanced Crypto AI with Web3',
                'features_enabled': [
                    'Web3 Token Metadata',
                    'Honeypot Detection',
                    'Liquidity Analysis',
                    'Holder Distribution Analysis',
                    'Smart Contract Verification',
                    'Smart Money Tracking',
                    'Momentum Analysis',
                    'Sentiment Analysis',
                    'ML Anomaly Detection',
                    'Whale Coordination Detection',
                    'Risk Assessment'
                ],
                'web3_features': [
                    'Contract Age Detection',
                    'Total Supply Analysis',
                    'Holder Count Tracking',
                    'Liquidity Pool Discovery',
                    'Contract Verification Check',
                    'Transfer Event Analysis'
                ],
                'reliability': 'professional_grade',
                'cloud_compatible': True
            }
        }
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Empty result"""
        return {
            'token_stats': None,
            'scores': {},
            'analysis_type': analysis_type,
            'enhanced': False,
            'web3_enhanced': False,
            'error': 'No data to analyze'
        }