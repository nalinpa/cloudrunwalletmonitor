import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedCryptoAI:
    """Streamlined AI system - same power, much cleaner code"""
    
    def __init__(self):
        # Core ML models - keep the essentials
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        
        # Simplified thresholds - easier to tune
        self.thresholds = {
            'whale_threshold': 0.7,
            'pump_threshold': 0.65,
            'anomaly_threshold': 0.3,
            'smart_money_threshold': 200,
            'min_holder_count': 50,
            'min_liquidity_eth': 5.0
        }
        
        # Scoring weights - easily adjustable
        self.weights = {
            'volume_score': 0.25,
            'wallet_quality': 0.20,
            'momentum': 0.15,
            'liquidity': 0.15,
            'holder_distribution': 0.10,
            'age_factor': 0.08,
            'risk_factor': 0.07
        }
        
        logger.info("🚀 Streamlined AI initialized - same power, cleaner code")
    
    async def complete_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Main AI analysis - streamlined but comprehensive"""
        try:
            logger.info(f"🤖 AI ANALYSIS: {len(purchases)} {analysis_type} transactions")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Step 1: Create enhanced DataFrame (efficient)
            df = self._create_enhanced_dataframe(purchases)
            if df.empty:
                return self._create_empty_result(analysis_type)
            
            # Step 2: Run core AI analyses (parallel where possible)
            analyses = await self._run_core_analyses(df)
            
            # Step 3: Create enhanced scores
            enhanced_scores = self._create_enhanced_scores(analyses, df)
            
            # Step 4: Build comprehensive result
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            logger.info(f"✅ AI SUCCESS: {len(enhanced_scores)} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return self._create_empty_result(analysis_type)
    
    def _create_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame - more efficient approach"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                # Extract Web3 data efficiently
                web3_data = getattr(purchase, 'web3_analysis', {}) or {}
                
                # Core data with Web3 enhancements
                row = {
                    # Basic data
                    'token': purchase.token_bought,
                    'eth_value': float(eth_value),
                    'amount': float(purchase.amount_received),
                    'wallet': purchase.wallet_address,
                    'wallet_score': float(wallet_score),
                    'timestamp': timestamp,
                    'tx_hash': purchase.transaction_hash,
                    'hour': timestamp.hour,
                    'unix_time': timestamp.timestamp(),
                    
                    # Web3 data (with defaults)
                    'token_age_hours': web3_data.get('token_age_hours', 999999),
                    'holder_count': web3_data.get('holder_count', 0),
                    'is_verified': web3_data.get('is_verified', False),
                    'honeypot_risk': web3_data.get('honeypot_risk', 0),
                    'smart_money_buying': web3_data.get('smart_money_buying', False),
                    'whale_accumulation': web3_data.get('whale_accumulation', False),
                    'has_liquidity': len(web3_data.get('liquidity_pools', [])) > 0
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
        """Detect pump signals - streamlined approach"""
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
        
        # New token factor
        new_token_ratio = len(df[df['is_new_token']]) / len(df)
        
        # Calculate pump score
        pump_score = (
            min((volume_acceleration - 1) * 0.3, 0.3) +
            min((wallet_growth - 1) * 0.2, 0.2) +
            min(smart_ratio * 0.2, 0.2) +
            (0.15 if new_token_ratio > 0.5 else 0) +
            (0.15 if df['has_smart_money'].any() else 0)
        )
        
        detected = pump_score >= self.thresholds['pump_threshold']
        
        return {
            'detected': detected,
            'score': pump_score,
            'volume_acceleration': volume_acceleration,
            'wallet_growth': wallet_growth,
            'phase': self._get_pump_phase(pump_score),
            'confidence': min(pump_score * 1.2, 1.0)
        }
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze smart money flow - efficient approach"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_trades = df[df['is_smart_wallet']]
        smart_volume = smart_trades['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        # Smart money signals
        smart_money_buying = df['smart_money_buying'].sum()
        whale_accumulation = df['whale_accumulation'].sum()
        
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
        """Create enhanced scores - streamlined approach"""
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
                
                # Web3 enhancements
                web3_bonus = 0
                if token_df['is_verified'].any():
                    web3_bonus += 5
                if token_df['has_liquidity'].any():
                    web3_bonus += 5
                if token_df['has_smart_money'].any():
                    web3_bonus += 10
                
                # Risk penalty
                avg_honeypot_risk = token_df['honeypot_risk'].mean()
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
                    'is_verified': token_df['is_verified'].any(),
                    'has_liquidity': token_df['has_liquidity'].any(),
                    'has_smart_money': token_df['has_smart_money'].any()
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