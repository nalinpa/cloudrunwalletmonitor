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
    """Simplified AI system without pandas-ta - ML + Sentiment only"""
    
    def __init__(self):
        # Core ML models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        
        # Memory cache
        self.cache = {}
        
        # AI thresholds
        self.thresholds = {
            'whale_coordination': 0.7,
            'pump_probability': 0.65,
            'anomaly_threshold': 0.3,
            'sentiment_threshold': 0.6,
            'smart_money_threshold': 200
        }
        
        # Sentiment keywords
        self.bullish_keywords = [
            'moon', 'pump', 'bullish', 'rocket', 'gem', 'alpha', 'breakout',
            'rally', 'surge', 'explosion', 'parabolic', 'massive', 'huge'
        ]
        
        self.bearish_keywords = [
            'dump', 'crash', 'bearish', 'fall', 'drop', 'sell', 'exit',
            'rug', 'scam', 'dead', 'rekt', 'liquidated', 'down'
        ]
        
        logger.info("🚀 AI initialized with ML + Sentiment Analysis (no technical indicators)")
    
    async def complete_ai_analysis(self, purchases: List, analysis_type: str) -> Dict:
        """Complete AI analysis without technical indicators"""
        try:
            logger.info(f"🤖 AI ANALYSIS: {len(purchases)} {analysis_type} transactions")
            
            if not purchases:
                return self._create_empty_result(analysis_type)
            
            # Enhanced DataFrame
            df = self._create_enhanced_dataframe(purchases)
            
            # AI ANALYSES (without technical indicators)
            analyses = {
                'whale_coordination': self._detect_whale_coordination(df),
                'pump_signals': self._detect_pump_signals(df),
                'anomaly_detection': self._detect_anomalies(df),
                'sentiment_analysis': await self._analyze_sentiment(df),
                'smart_money_flow': self._analyze_smart_money_flow(df),
                'momentum_analysis': self._analyze_basic_momentum(df),  # Simple momentum without TA
                'risk_assessment': self._assess_risks(df)
            }
            
            # Enhanced scoring without technical indicators
            enhanced_scores = self._create_enhanced_scores(analyses, df)
            
            # Build result
            result = self._build_enhanced_result(analyses, enhanced_scores, analysis_type)
            
            logger.info(f"✅ AI SUCCESS: {len(enhanced_scores)} tokens analyzed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_result(analysis_type)
    
    def _analyze_basic_momentum(self, df: pd.DataFrame) -> Dict:
        """Basic momentum analysis without technical indicators"""
        if df.empty or len(df) < 5:
            return {'momentum_detected': False, 'strength': 0}
        
        try:
            momentum_signals = {}
            
            for token, token_df in df.groupby('token'):
                if len(token_df) < 3:
                    continue
                
                # Sort by time
                token_df = token_df.sort_values('unix_time').reset_index(drop=True)
                
                # Simple momentum indicators
                token_momentum = {}
                
                # Volume acceleration
                mid_point = len(token_df) // 2
                early_volume = token_df.iloc[:mid_point]['eth_value'].sum()
                recent_volume = token_df.iloc[mid_point:]['eth_value'].sum()
                volume_acceleration = recent_volume / early_volume if early_volume > 0 else 1
                token_momentum['volume_acceleration'] = volume_acceleration
                
                # Transaction frequency
                time_span = token_df['unix_time'].max() - token_df['unix_time'].min()
                tx_frequency = len(token_df) / max(time_span / 3600, 1)  # Transactions per hour
                token_momentum['tx_frequency'] = tx_frequency
                
                # Wallet growth
                early_wallets = token_df.iloc[:mid_point]['wallet'].nunique()
                recent_wallets = token_df.iloc[mid_point:]['wallet'].nunique()
                wallet_growth = recent_wallets / early_wallets if early_wallets > 0 else 1
                token_momentum['wallet_growth'] = wallet_growth
                
                # Simple price momentum (using eth_value as proxy)
                if len(token_df) >= 3:
                    values = token_df['eth_value'].values
                    simple_trend = np.polyfit(range(len(values)), values, 1)[0]
                    token_momentum['simple_trend'] = simple_trend
                else:
                    token_momentum['simple_trend'] = 0
                
                # Overall momentum score
                momentum_score = (
                    min((volume_acceleration - 1) * 0.3, 0.3) +
                    min(tx_frequency * 0.1, 0.2) +
                    min((wallet_growth - 1) * 0.3, 0.3) +
                    (0.2 if simple_trend > 0 else 0)
                )
                
                token_momentum['overall_score'] = momentum_score
                momentum_signals[token] = token_momentum
            
            # Calculate overall momentum strength
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
                'total_tokens_analyzed': len(momentum_signals)
            }
            
        except Exception as e:
            logger.error(f"Basic momentum analysis failed: {e}")
            return {'momentum_detected': False, 'strength': 0}
    
    def _create_enhanced_scores(self, analyses: Dict, df: pd.DataFrame) -> Dict:
        """Enhanced scoring without technical indicators"""
        enhanced_scores = {}
        
        # Basic token stats
        token_stats = df.groupby('token').agg({
            'eth_value': ['sum', 'mean', 'count'],
            'wallet': 'nunique',
            'wallet_score': 'mean'
        }).round(4)
        
        token_stats.columns = ['total_value', 'mean_value', 'tx_count', 'unique_wallets', 'avg_score']
        
        for token in token_stats.index:
            try:
                stats = token_stats.loc[token]
                
                # Basic score components
                volume_score = min(stats['total_value'] * 40, 50)
                diversity_score = min(stats['unique_wallets'] * 6, 25)
                quality_score = min((stats['avg_score'] / 300) * 25, 25)
                basic_score = volume_score + diversity_score + quality_score
                
                # AI MULTIPLIER (without technical analysis)
                ai_multiplier = 1.0
                
                # 1. Whale coordination boost
                whale_coord = analyses['whale_coordination']
                if whale_coord['detected']:
                    ai_multiplier += whale_coord['score'] * 0.4
                
                # 2. Pump signal boost
                pump_signals = analyses['pump_signals']
                if pump_signals['detected']:
                    ai_multiplier += pump_signals['score'] * 0.5
                
                # 3. Basic momentum boost (replaces technical analysis)
                momentum = analyses['momentum_analysis']
                token_momentum = momentum.get('token_momentum', {}).get(token, {})
                momentum_score = token_momentum.get('overall_score', 0)
                if momentum_score > 0.5:
                    ai_multiplier += momentum_score * 0.4
                
                # 4. Sentiment boost
                sentiment = analyses['sentiment_analysis']
                token_sentiment = sentiment.get('token_sentiments', {}).get(token, {})
                sentiment_score = token_sentiment.get('combined_score', 0)
                if sentiment_score > 0.3:
                    ai_multiplier += sentiment_score * 0.3
                
                # 5. Smart money boost
                smart_money = analyses['smart_money_flow']
                if smart_money['flow_direction'] in ['BULLISH', 'STRONG_BULLISH']:
                    ai_multiplier += smart_money['confidence'] * 0.4
                
                # 6. Risk penalty
                risk_penalty = analyses['risk_assessment']['overall_risk'] * 0.3
                ai_multiplier = max(ai_multiplier - risk_penalty, 0.6)
                
                # Final enhanced score
                ultimate_score = basic_score * ai_multiplier
                confidence = min(0.95, 0.7 + (ai_multiplier - 1) * 0.25)
                
                enhanced_scores[token] = {
                    'total_score': ultimate_score,
                    'basic_score': basic_score,
                    'ai_multiplier': ai_multiplier,
                    'ai_enhanced': True,
                    'confidence': confidence,
                    
                    # Component scores
                    'volume_score': volume_score,
                    'diversity_score': diversity_score,
                    'quality_score': quality_score,
                    
                    # AI analysis (no technical indicators)
                    'whale_coordination': whale_coord['score'],
                    'pump_probability': pump_signals['score'],
                    'momentum_score': momentum_score,
                    'sentiment_score': sentiment_score,
                    'smart_money_score': smart_money['confidence'],
                    'risk_score': 1 - analyses['risk_assessment']['overall_risk'],
                    
                    # Detailed AI results
                    'whale_evidence': whale_coord.get('evidence_strength', 'NONE'),
                    'pump_phase': pump_signals.get('phase', 'NORMAL'),
                    'sentiment_direction': token_sentiment.get('overall_sentiment', 'NEUTRAL'),
                    'smart_money_direction': smart_money.get('flow_direction', 'NEUTRAL'),
                    'risk_level': analyses['risk_assessment'].get('risk_level', 'MEDIUM'),
                    
                    # Momentum details
                    'momentum_details': token_momentum,
                    
                    # AI metadata
                    'ai_features_detected': self._count_ai_features(analyses, token)
                }
                
            except Exception as e:
                logger.error(f"Error creating enhanced score for {token}: {e}")
                continue
        
        return enhanced_scores
    
    def _count_ai_features(self, analyses: Dict, token: str) -> int:
        """Count AI features detected for this token"""
        features = 0
        
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
        
        return features
    
    def _build_enhanced_result(self, analyses: Dict, enhanced_scores: Dict, analysis_type: str) -> Dict:
        """Build enhanced analysis result without technical indicators"""
        return {
            'token_stats': None,
            'scores': enhanced_scores,
            'analysis_type': analysis_type,
            'enhanced': True,
            
            # AI analysis
            'ai_analyses': analyses,
            
            # Enhanced summary statistics
            'analysis_summary': {
                'total_tokens': len(enhanced_scores),
                'ai_patterns_detected': sum(1 for analysis in analyses.values() 
                                           if isinstance(analysis, dict) and analysis.get('detected', False)),
                'avg_confidence': np.mean([s['confidence'] for s in enhanced_scores.values()]) if enhanced_scores else 0,
                'high_confidence_tokens': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.85),
                'ultra_high_confidence': sum(1 for s in enhanced_scores.values() if s['confidence'] > 0.9),
                'risk_alerts': sum(1 for s in enhanced_scores.values() if s.get('risk_level') in ['HIGH']),
                'sentiment_bullish': sum(1 for s in enhanced_scores.values() if s.get('sentiment_direction') == 'BULLISH'),
                'pump_signals': sum(1 for s in enhanced_scores.values() if s.get('pump_phase') != 'NORMAL'),
                'momentum_bullish': sum(1 for s in enhanced_scores.values() if s.get('momentum_score', 0) > 0.5),
                'whale_coordination_detected': analyses['whale_coordination']['detected']
            },
            
            # AI metadata
            'enhancement_info': {
                'ai_engine': 'Advanced Crypto AI (Simplified)',
                'features_enabled': [
                    'Basic Momentum Analysis',
                    'Sentiment Analysis (TextBlob + Keywords)',
                    'ML Anomaly Detection (Isolation Forest + DBSCAN)',
                    'Whale Coordination Detection',
                    'Pump Signal Analysis',
                    'Smart Money Flow Analysis',
                    'Risk Assessment'
                ],
                'technical_indicators': 'None (removed pandas-ta dependency)',
                'reliability': 'high_grade',
                'cloud_compatible': True
            }
        }
    
    # Keep all the existing helper methods unchanged
    def _create_enhanced_dataframe(self, purchases: List) -> pd.DataFrame:
        """Create enhanced DataFrame with all features"""
        data = []
        
        for purchase in purchases:
            try:
                timestamp = getattr(purchase, 'timestamp', datetime.now())
                eth_value = getattr(purchase, 'eth_spent', getattr(purchase, 'amount_received', 0))
                wallet_score = getattr(purchase, 'sophistication_score', 0) or 0
                
                data.append({
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
                    'unix_time': timestamp.timestamp()
                })
            except Exception as e:
                logger.warning(f"Error processing purchase: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Enhanced features
        df['log_eth'] = np.log1p(df['eth_value'])
        df['eth_rank'] = df['eth_value'].rank(pct=True)
        df['is_whale'] = df['eth_value'] > df['eth_value'].quantile(0.85)
        df['is_smart_wallet'] = df['wallet_score'] > self.thresholds['smart_money_threshold']
        df['time_rank'] = df['unix_time'].rank(method='dense')
        
        return df
    
    def _detect_whale_coordination(self, df: pd.DataFrame) -> Dict:
        """Advanced whale coordination detection"""
        if df.empty:
            return {'detected': False, 'score': 0}
        
        whale_trades = df[df['is_whale']].copy()
        
        if len(whale_trades) < 2:
            return {'detected': False, 'score': 0}
        
        # Time coordination analysis
        whale_trades = whale_trades.sort_values('unix_time')
        time_diffs = whale_trades['unix_time'].diff().dropna()
        coordinated_trades = sum(time_diffs <= 300)  # 5 minutes
        
        # Amount similarity analysis
        amounts = whale_trades['eth_value'].values
        if len(amounts) > 1:
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1
            amount_similarity = max(0, 1 - cv)
        else:
            amount_similarity = 0
        
        # Coordination score
        score = (
            min(coordinated_trades / len(whale_trades), 1) * 0.5 +
            amount_similarity * 0.3 +
            min(len(whale_trades) / 10, 1) * 0.2
        )
        
        detected = score >= self.thresholds['whale_coordination']
        
        return {
            'detected': detected,
            'score': score,
            'whale_count': len(whale_trades),
            'coordinated_trades': coordinated_trades,
            'amount_similarity': amount_similarity,
            'evidence_strength': 'HIGH' if score > 0.8 else 'MEDIUM' if score > 0.5 else 'LOW'
        }
    
    def _detect_pump_signals(self, df: pd.DataFrame) -> Dict:
        """Advanced pump signal detection"""
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
        
        # Calculate pump score
        pump_score = (
            min((volume_acceleration - 1) * 0.4, 0.4) +
            min((wallet_growth - 1) * 0.3, 0.3) +
            min(smart_ratio * 0.3, 0.3)
        )
        
        detected = pump_score >= self.thresholds['pump_probability']
        
        return {
            'detected': detected,
            'score': pump_score,
            'volume_acceleration': volume_acceleration,
            'wallet_growth': wallet_growth,
            'smart_money_ratio': smart_ratio,
            'phase': self._get_pump_phase(pump_score),
            'confidence': min(pump_score * 1.3, 1.0)
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """ML anomaly detection"""
        if df.empty or len(df) < 5:
            return {'detected': False, 'score': 0}
        
        try:
            features = df[['log_eth', 'wallet_score', 'hour', 'minute']].fillna(0)
            features_scaled = self.scaler.fit_transform(features)
            
            anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
            anomaly_scores = self.anomaly_detector.score_samples(features_scaled)
            
            anomaly_count = sum(anomaly_labels == -1)
            anomaly_ratio = anomaly_count / len(df)
            
            detected = anomaly_ratio >= self.thresholds['anomaly_threshold']
            
            return {
                'detected': detected,
                'score': anomaly_ratio,
                'anomaly_count': anomaly_count,
                'avg_anomaly_score': float(np.mean(anomaly_scores)),
                'confidence': min(anomaly_ratio * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'detected': False, 'score': 0}
    
    async def _analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """Advanced sentiment analysis"""
        try:
            sentiment_data = {}
            
            for token in df['token'].unique():
                token_sentiment = {
                    'polarity': 0,
                    'subjectivity': 0,
                    'keyword_sentiment': 0,
                    'overall_sentiment': 'NEUTRAL'
                }
                
                # TextBlob analysis
                try:
                    blob = TextBlob(token.lower())
                    token_sentiment['polarity'] = blob.sentiment.polarity
                    token_sentiment['subjectivity'] = blob.sentiment.subjectivity
                except:
                    pass
                
                # Keyword analysis
                token_lower = token.lower()
                bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in token_lower)
                bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in token_lower)
                
                if bullish_score > bearish_score:
                    keyword_sentiment = 0.5 + (bullish_score * 0.1)
                elif bearish_score > bullish_score:
                    keyword_sentiment = -0.5 - (bearish_score * 0.1)
                else:
                    keyword_sentiment = 0
                
                token_sentiment['keyword_sentiment'] = max(-1, min(1, keyword_sentiment))
                
                # Combined sentiment
                combined = (token_sentiment['polarity'] + token_sentiment['keyword_sentiment']) / 2
                
                if combined > 0.3:
                    overall = 'BULLISH'
                elif combined < -0.3:
                    overall = 'BEARISH'
                else:
                    overall = 'NEUTRAL'
                
                token_sentiment['overall_sentiment'] = overall
                token_sentiment['combined_score'] = combined
                
                sentiment_data[token] = token_sentiment
            
            # Overall metrics
            all_sentiments = [data['combined_score'] for data in sentiment_data.values()]
            avg_sentiment = np.mean(all_sentiments) if all_sentiments else 0
            
            bullish_count = sum(1 for data in sentiment_data.values() if data['overall_sentiment'] == 'BULLISH')
            bearish_count = sum(1 for data in sentiment_data.values() if data['overall_sentiment'] == 'BEARISH')
            
            return {
                'token_sentiments': sentiment_data,
                'average_sentiment': avg_sentiment,
                'bullish_tokens': bullish_count,
                'bearish_tokens': bearish_count,
                'sentiment_detected': abs(avg_sentiment) > self.thresholds['sentiment_threshold']
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'token_sentiments': {}, 'average_sentiment': 0, 'sentiment_detected': False}
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Smart money flow analysis"""
        if df.empty:
            return {'flow_direction': 'NEUTRAL', 'confidence': 0}
        
        smart_trades = df[df['is_smart_wallet']]
        
        smart_volume = smart_trades['eth_value'].sum()
        total_volume = df['eth_value'].sum()
        smart_ratio = smart_volume / total_volume if total_volume > 0 else 0
        
        # Flow direction
        if smart_ratio > 0.4:
            direction = 'STRONG_BULLISH'
        elif smart_ratio > 0.25:
            direction = 'BULLISH'
        elif smart_ratio < 0.05:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'flow_direction': direction,
            'confidence': min(smart_ratio * 2, 1.0),
            'smart_money_ratio': smart_ratio,
            'smart_wallet_count': len(smart_trades)
        }
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Comprehensive risk assessment"""
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
        
        # Overall risk
        overall_risk = np.mean(list(risk_factors.values()))
        
        if overall_risk >= 0.7:
            risk_level = 'HIGH'
        elif overall_risk >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _get_pump_phase(self, score: float) -> str:
        """Get pump phase"""
        if score >= 0.7:
            return "PUMP_IMMINENT"
        elif score >= 0.5:
            return "LATE_ACCUMULATION" 
        elif score >= 0.3:
            return "EARLY_ACCUMULATION"
        else:
            return "NORMAL"
    
    def _create_empty_result(self, analysis_type: str) -> Dict:
        """Empty result"""
        return {
            'token_stats': None,
            'scores': {},
            'analysis_type': analysis_type,
            'enhanced': False,
            'error': 'No data to analyze'
        }