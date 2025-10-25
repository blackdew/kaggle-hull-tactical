"""
EXP-019: Aggressive Ensemble Strategy for 10+ Public Score

Strategy: 5-model ensemble with Kelly Criterion and regime-based weighting
CV Sharpe: 3.54 ± 0.77
Expected Public Score: 20-30+
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None


class EXP019Server(InferenceServer):
    """EXP-019: Aggressive Ensemble for 10+ Public Score"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
        from itertools import combinations
        from scipy.stats import skew, kurtosis

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.combinations = combinations
        self.skew = skew
        self.kurtosis = kurtosis
        self.ready = False

        # Base features
        self.top_20_base = [
            'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
            'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
        ]

        # Top 30 features (best performing)
        self.top_30_features = [
            'V13*I2*S5', 'E19*P7', 'M4*I2*S5', 'V13²', 'M4*V7',
            'V7*P7', 'V13³', 'V13*P7', 'M4*V13*V7', 'M4*V13',
            'M4*V13*S5', 'V13*E19', 'V13*V7', 'V13*V7*E19', 'M4*V13*E19',
            'market_turbulence', 'M4*E19', 'M4*V7*E19', 'M4*V13*S2*I2', 'V13⁴',
            'M4*V13*V7*S2', 'vol_mean_V', 'vol_composite_V_all', 'M4*V13*S2', 'M4*S2*I2',
            'M4*V13*V7*I2', 'V13*S2*E19', 'M_min', 'vol_composite_V', 'V_max'
        ]

        # Model configurations (top 5 models)
        self.model_configs = [
            {'name': 'k250_top30', 'k': 250, 'features': self.top_30_features,
             'n_estimators': 150, 'max_depth': 7, 'lr': 0.025, 'weight': 3.0},
            {'name': 'k100_conservative', 'k': 100, 'features': self.top_30_features,
             'n_estimators': 100, 'max_depth': 5, 'lr': 0.03, 'weight': 2.0},
            {'name': 'k250_top30_v2', 'k': 250, 'features': self.top_30_features,
             'n_estimators': 150, 'max_depth': 7, 'lr': 0.025, 'weight': 2.0},
            {'name': 'k150_top30', 'k': 150, 'features': self.top_30_features,
             'n_estimators': 150, 'max_depth': 7, 'lr': 0.025, 'weight': 1.5},
            {'name': 'k100_top30_v2', 'k': 100, 'features': self.top_30_features,
             'n_estimators': 150, 'max_depth': 7, 'lr': 0.025, 'weight': 1.0},
        ]

        def predict(batch):
            return EXP019Server.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def create_all_features(self, df):
        """Create all 284 features (1-row calculable)"""
        X_base = df[self.top_20_base].fillna(0).replace([np.inf, -np.inf], 0)
        features = {}
        eps = 1e-8

        top_10 = self.top_20_base[:10]
        top_5 = self.top_20_base[:5]
        top_8 = self.top_20_base[:8]
        top_6 = self.top_20_base[:6]

        # 2-way interactions
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i+1:]:
                features[f'{feat1}*{feat2}'] = X_base[feat1] * X_base[feat2]
                features[f'{feat1}/{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)
        for feat in top_5:
            features[f'{feat}²'] = X_base[feat] ** 2
            features[f'{feat}³'] = X_base[feat] ** 3
            features[f'{feat}⁴'] = X_base[feat] ** 4

        # 3-way interactions
        for feat1, feat2, feat3 in self.combinations(top_8, 3):
            features[f'{feat1}*{feat2}*{feat3}'] = X_base[feat1] * X_base[feat2] * X_base[feat3]
            if feat1 in top_5 and feat2 in top_5:
                features[f'({feat1}*{feat2})/{feat3}'] = (X_base[feat1] * X_base[feat2]) / (X_base[feat3].abs() + eps)

        # 4-way interactions
        for feat1, feat2, feat3, feat4 in self.combinations(top_6, 4):
            features[f'{feat1}*{feat2}*{feat3}*{feat4}'] = X_base[feat1] * X_base[feat2] * X_base[feat3] * X_base[feat4]

        # Meta-features
        M_features = ['M4', 'M2', 'M3', 'M12']
        V_features = ['V13', 'V7', 'V9', 'V10']
        P_features = ['P8', 'P7', 'P5', 'P12', 'P10', 'P11']
        S_features = ['S2', 'S5', 'S8']

        for cat_name, cat_feats in [('M', M_features), ('V', V_features), ('P', P_features), ('S', S_features)]:
            cat_data = X_base[cat_feats]
            features[f'{cat_name}_mean'] = cat_data.mean(axis=1)
            features[f'{cat_name}_std'] = cat_data.std(axis=1)
            features[f'{cat_name}_min'] = cat_data.min(axis=1)
            features[f'{cat_name}_max'] = cat_data.max(axis=1)
            features[f'{cat_name}_range'] = features[f'{cat_name}_max'] - features[f'{cat_name}_min']
            features[f'{cat_name}_skew'] = cat_data.apply(lambda x: self.skew(x, nan_policy='omit'), axis=1)
            features[f'{cat_name}_kurt'] = cat_data.apply(lambda x: self.kurtosis(x, nan_policy='omit'), axis=1)
            features[f'{cat_name}_cv'] = features[f'{cat_name}_std'] / (features[f'{cat_name}_mean'].abs() + eps)

        # Cross-category features
        features['market_strength'] = features['M_mean'] / (features['V_mean'] + eps)
        features['risk_adjusted_return'] = features['P_mean'] / (features['V_mean'] + eps)
        features['sentiment_to_volatility'] = features['S_mean'] / (features['V_mean'] + eps)
        features['market_to_price'] = features['M_mean'] / (features['P_mean'] + eps)
        features['total_volatility'] = np.sqrt(features['V_mean']**2 + features['M_std']**2 + eps)
        features['market_turbulence'] = features['M_std'] * features['V_std']
        features['price_momentum'] = features['P_max'] - features['P_min']
        features['normalized_market'] = features['M_mean'] / (features['total_volatility'] + eps)
        features['normalized_sentiment'] = features['S_mean'] / (features['total_volatility'] + eps)

        # Volatility features
        features['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3
        features['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)
        features['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
        features['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
        features['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])
        features['market_vol_composite'] = np.sqrt(X_base['M4']**2 + X_base['M2']**2 + X_base['M3']**2 + eps)
        features['price_vol'] = np.sqrt((X_base['P8'] - X_base['P7'])**2 + (X_base['P7'] - X_base['P5'])**2 + eps)

        # Ratio features
        for feat1 in top_5:
            for feat2 in top_5:
                if feat1 != feat2:
                    features[f'ratio_{feat1}_to_{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)

        return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)

    def train_if_needed(self):
        if self.ready:
            return

        print("[INFO] Starting ensemble training...")

        # Load train data
        def _load_train():
            candidates = [
                'train.csv',
                './train.csv',
                '/kaggle/input/hull-tactical-market-prediction/train.csv',
                '/kaggle/working/train.csv',
                'data/train.csv',
            ]
            for path in candidates:
                if os.path.exists(path):
                    print(f"[INFO] Found train.csv at: {path}")
                    return pd.read_csv(path)
            raise FileNotFoundError('train.csv not found')

        train = _load_train()
        print(f"[INFO] Train data shape: {train.shape}")

        # Create all features
        X_all = self.create_all_features(train)
        print(f"[INFO] Features created: {X_all.shape}")

        # Volatility proxy for regime detection
        vol_proxy = X_all['vol_mean_V'].to_numpy()
        self.vol_75 = np.percentile(vol_proxy, 75)
        self.vol_25 = np.percentile(vol_proxy, 25)

        y = train['market_forward_excess_returns'].to_numpy()

        # Train all models
        self.models = []

        for config in self.model_configs:
            print(f"[INFO] Training {config['name']}...")

            # Select features
            X = X_all[config['features']].copy()

            # Scale
            scaler = self.StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train
            model = self.XGBRegressor(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                learning_rate=config['lr'],
                subsample=1.0,
                colsample_bytree=0.6,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)

            self.models.append({
                'model': model,
                'scaler': scaler,
                'features': config['features'],
                'k': config['k'],
                'weight': config['weight']
            })

        print(f"[INFO] Trained {len(self.models)} models")
        print(f"[INFO] Volatility thresholds: High={self.vol_75:.4f}, Low={self.vol_25:.4f}")

        self.ready = True

    def predict(self, test_batch):
        self.train_if_needed()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        # Convert Polars to Pandas
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Create all features
        X_all = self.create_all_features(df)

        # Get volatility proxy
        vol_proxy_val = X_all['vol_mean_V'].values[0]

        # Ensemble prediction with regime-based weighting
        predictions = []
        k_values = []
        weights = []

        for model_dict in self.models:
            model = model_dict['model']
            scaler = model_dict['scaler']
            features = model_dict['features']
            k = model_dict['k']
            weight = model_dict['weight']

            # Extract and scale features
            X = X_all[features].values.reshape(1, -1)
            X_scaled = scaler.transform(X)

            # Predict
            pred = model.predict(X_scaled)[0]

            predictions.append(pred)
            k_values.append(k)
            weights.append(weight)

        # Regime-based weight adjustment
        if vol_proxy_val > self.vol_75:  # High volatility - favor conservative
            regime_weights = np.array([3.0, 2.0, 1.0, 0.5, 0.25])
        elif vol_proxy_val < self.vol_25:  # Low volatility - balanced
            regime_weights = np.array([2.0, 2.0, 1.5, 1.0, 0.5])
        else:  # Normal - performance-based
            regime_weights = np.array(weights)

        regime_weights = regime_weights / regime_weights.sum()

        # Weighted average prediction
        excess_pred = np.average(predictions, weights=regime_weights)

        # Kelly Criterion inspired position sizing
        confidence = abs(excess_pred)
        win_prob = 1.0 / (1.0 + np.exp(-excess_pred * 1000))
        kelly_fraction = 2 * win_prob - 1.0

        # Quantile-based K adjustment
        pred_abs = abs(excess_pred)
        if pred_abs > 0.0015:  # Very confident
            k_multiplier = 2.0
        elif pred_abs < 0.00005:  # Very uncertain
            k_multiplier = 0.5
        else:
            k_multiplier = 1.0

        # Dynamic K based on volatility
        vol_adjustment = 1.0 / (1.0 + 1.0 * vol_proxy_val)

        # Base K (weighted average)
        k_base = np.average(k_values, weights=regime_weights)

        # Final K
        k_final = k_base * vol_adjustment * k_multiplier

        # Position with Kelly adjustment
        position_kelly_adjusted = 1.0 + (excess_pred * k_final) * abs(kelly_fraction)

        # Clip to [0, 2]
        position = np.clip(position_kelly_adjusted, 0.0, 2.0)

        return float(position)


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-019")
    print("[INFO] Target: 10+ Public Score")
    print("[INFO] CV Sharpe: 3.54 ± 0.77")
    print("[INFO] Expected: 20-30 Public Score")
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {os.getenv('KAGGLE_IS_COMPETITION_RERUN')}")

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("[INFO] Running in COMPETITION RERUN mode - starting server")
        srv = EXP019Server()
        srv.serve()
    else:
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        srv = EXP019Server()
        srv.server.start()
        try:
            # Find competition data
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',
                '.',
                'data',
            ]
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    print(f"[INFO] Found test.csv at: {comp_dir}")
                    break

            if comp_dir is None:
                print("[INFO] Running gateway with default paths")
                DefaultGateway().run()
            else:
                print(f"[INFO] Running gateway with data_paths: {comp_dir}")
                DefaultGateway(data_paths=(comp_dir,)).run()

            # Verify submission.parquet
            if os.path.exists('submission.parquet'):
                sub = pd.read_parquet('submission.parquet')
                print(f"\n[SUCCESS] submission.parquet created! Shape: {sub.shape}")
                print(f"[INFO] Prediction stats - Mean: {sub['prediction'].mean():.4f}, Std: {sub['prediction'].std():.4f}")
                print(f"[INFO] Prediction range: [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")
                print("\n[INFO] First 5 predictions:")
                print(sub.head())
                print("\n" + "="*80)
                print("READY FOR KAGGLE SUBMISSION!")
                print("="*80)
                print("Expected Public Score: 20-30 (Target: 10+)")
                print("="*80)
            else:
                print("[ERROR] submission.parquet not created!")
        finally:
            srv.server.stop(0)

        print("[END] Script complete")
