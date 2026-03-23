import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

import warnings
import ast
import joblib
import os
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import optuna
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.feature_selector = None
        self.scaler_robust = RobustScaler()
        self.is_fitted = False
        
    def create_interaction_features(self, df):
        features_dict = {}
        
        time_features = ['time_type_주간', 'time_type_오후', 'time_type_야간', 'time_type_오전']
        food_features = [col for col in df.columns if col.startswith('MCT_TYPE_')]
        
        for time_feat in time_features:
            for food_feat in food_features:
                feature_name = f"{time_feat}_{food_feat}"
                features_dict[feature_name] = df[time_feat] * df[food_feat]
        
        gender_features = ['gender_target_female', 'gender_target_male']
        for gender_feat in gender_features:
            for food_feat in food_features:
                feature_name = f"{gender_feat}_{food_feat}"
                features_dict[feature_name] = df[gender_feat] * df[food_feat]
        
        for food_feat in food_features:
            features_dict[f"Latitude_{food_feat}"] = df['Latitude'] * df[food_feat]
            features_dict[f"Longitude_{food_feat}"] = df['Longitude'] * df[food_feat]
        
        continuous_features = ['LOCAL_UE_CNT_RAT', 'young_ratio', 'Latitude', 'Longitude']
        for i, feat1 in enumerate(continuous_features):
            for feat2 in continuous_features[i+1:]:
                features_dict[f"{feat1}_{feat2}"] = df[feat1] * df[feat2]
        
        return pd.DataFrame(features_dict)
    
    def create_aggregated_features(self, df):
        features_dict = {}
        
        food_features = [col for col in df.columns if col.startswith('MCT_TYPE_')]
        features_dict['total_food_types'] = df[food_features].sum(axis=1)
        
        time_features = ['time_type_주간', 'time_type_오후', 'time_type_야간', 'time_type_오전']
        features_dict['total_time_types'] = df[time_features].sum(axis=1)
        
        features_dict['gender_specialization'] = df['gender_target_female'] + df['gender_target_male']
                
        features_dict['young_local_interaction'] = df['young_ratio'] * df['LOCAL_UE_CNT_RAT']
        
        return pd.DataFrame(features_dict)
    
    def create_binned_features(self, df):
        features_dict = {}
        
        continuous_features = ['LOCAL_UE_CNT_RAT', 'young_ratio', 'Latitude', 'Longitude']
        
        for feat in continuous_features:
            features_dict[f"{feat}_binned"] = pd.cut(df[feat], bins=5, labels=False)
            features_dict[f"{feat}_level"] = pd.cut(df[feat], bins=3, labels=['low', 'mid', 'high'])
        
        return pd.DataFrame(features_dict)
    
    def fit_transform(self, df):
        result_df = df.copy()
        
        interaction_features = self.create_interaction_features(df)
        result_df = pd.concat([result_df, interaction_features], axis=1)
        
        aggregated_features = self.create_aggregated_features(df)
        result_df = pd.concat([result_df, aggregated_features], axis=1)
        
        binned_features = self.create_binned_features(df)
        binned_encoded = pd.get_dummies(binned_features, columns=[col for col in binned_features.columns if '_level' in col])
        result_df = pd.concat([result_df, binned_encoded], axis=1)
        
        self.is_fitted = True
        
        return result_df
    
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("not self.is_fitted")
        return self.fit_transform(df) 

class OptimizedEnsembleModel:
    def __init__(self, random_state=42, use_feature_engineering=True, use_hyperopt=False):
        self.random_state = random_state
        self.use_feature_engineering = use_feature_engineering
        self.use_hyperopt = use_hyperopt
        self.models = {}
        self.weights = {}
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.feature_engineering = AdvancedFeatureEngineering(random_state)
        self.embedding_reducers = {
            'pca': PCA(n_components=50, random_state=random_state),
            'svd': TruncatedSVD(n_components=50, random_state=random_state)
        }
        self.feature_selectors = {}
        self.meta_model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.level2_models = {}
        self.is_fitted = False
        self.training_info = {}
        self.smote = SMOTE(random_state=random_state)
        
    def optimize_hyperparameters(self, model_name, model_class, param_grid, X, y, cv=3):
        grid_search = GridSearchCV(
            estimator=model_class,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def prepare_enhanced_data(self, df):
        if self.use_feature_engineering:
            df_enhanced = self.feature_engineering.fit_transform(df) if not self.feature_engineering.is_fitted else self.feature_engineering.transform(df)
        else:
            df_enhanced = df.copy()
        
        y = df_enhanced['UE_AMT_GRP_encoded'].values
        X_features = df_enhanced.drop(['UE_AMT_GRP_encoded'], axis=1)
        
        embedding_data = None
        if 'keyword_embeded' in X_features.columns:
            embedding_data = []
            for embed_str in X_features['keyword_embeded']:
                try:
                    embed_array = np.array(ast.literal_eval(embed_str))
                    embedding_data.append(embed_array)
                except:
                    embedding_data.append(np.zeros(1024))
            
            embedding_data = np.array(embedding_data)
            X_features = X_features.drop(['keyword_embeded'], axis=1)
        
        if 'clustered_keyword' in X_features.columns:
            X_features = X_features.drop(['clustered_keyword'], axis=1)
        
        binary_features = []
        continuous_features = []
        
        for col in X_features.columns:
            if X_features[col].nunique() == 2 and set(X_features[col].unique()).issubset({0, 1}):
                binary_features.append(col)
            else:
                continuous_features.append(col)
        
        return {
            'y': y,
            'X_all': X_features.values,
            'X_binary': X_features[binary_features].values if binary_features else None,
            'X_continuous': X_features[continuous_features].values if continuous_features else None,
            'X_embedding': embedding_data,
            'feature_names': list(X_features.columns),
            'binary_features': binary_features,
            'continuous_features': continuous_features
        }
    
    def build_optimized_models(self, data_dict):
        X_all = data_dict['X_all']
        y = data_dict['y']
        
        models = {}
        
        if self.use_hyperopt:
            rf_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf_model = self.optimize_hyperparameters('RandomForest', RandomForestClassifier(random_state=self.random_state, n_jobs=-1), rf_params, X_all, y)
        else: # 관련 레퍼 중에 관련한 거 있어신디 /// 찾아보고 -
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        models['rf_optimized'] = (rf_model, X_all)
        
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        models['extra_trees'] = (et_model, X_all)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        models['xgb_optimized'] = (xgb_model, X_all)
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1, ### 너무 느린데 요거 살짝 더 늘려도 괜찮을라나
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        models['lgb_optimized'] = (lgb_model, X_all)
        
        if CATBOOST_AVAILABLE:
            cat_model = cb.CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=self.random_state,
                verbose=False
            )
            models['catboost'] = (cat_model, X_all)
        
        X_scaled_standard = self.scalers['standard'].fit_transform(X_all)
        
        nn1_model = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        models['nn_deep'] = (nn1_model, X_scaled_standard)
        
        nn2_model = MLPClassifier(
            hidden_layer_sizes=(150, 75),
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.01
        )
        models['nn_wide'] = (nn2_model, X_scaled_standard)
        
        if data_dict['X_embedding'] is not None:
            X_embed_pca = self.embedding_reducers['pca'].fit_transform(data_dict['X_embedding'])
            
            X_embed_svd = self.embedding_reducers['svd'].fit_transform(data_dict['X_embedding'])
            
            nn_embed_pca = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
            models['nn_embed_pca'] = (nn_embed_pca, X_embed_pca)
            
            nn_embed_svd = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500, ## /// 줄여야? 
                random_state=self.random_state,
                early_stopping=True
            )
            models['nn_embed_svd'] = (nn_embed_svd, X_embed_svd)
        
        X_scaled_robust = self.scalers['robust'].fit_transform(X_all)
        
        svm_model = SVC( ### 스븜이 성능 왜이래
            probability=True,
            random_state=self.random_state,
            C=1.0,
            gamma='scale'
        )
        models['svm_optimized'] = (svm_model, X_scaled_robust)
        
        knn_model = KNeighborsClassifier(n_neighbors=15, weights='distance')
        models['knn'] = (knn_model, X_scaled_standard)
        
        try:
            qda_model = QuadraticDiscriminantAnalysis()
            models['qda'] = (qda_model, X_scaled_standard)
        except:
            print("except // qda <<<<<<<<< 왜 계속 뭔가뭔가 오류가 나는")
        
        logistic_model = LogisticRegression(
            random_state=self.random_state, 
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        )
        models['logistic'] = (logistic_model, X_scaled_standard)
        
        balanced_rf = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        models['balanced_rf'] = (balanced_rf, X_all)
        
        return models
    
    def calculate_advanced_weights(self, models, y, cv=5):
        
        weights = {}
        performance_metrics = {}
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for model_name, (model, X_data) in models.items():
            try:
                accuracy_scores = cross_val_score(model, X_data, y, cv=skf, scoring='accuracy', n_jobs=-1)
                f1_scores = cross_val_score(model, X_data, y, cv=skf, scoring='f1_macro', n_jobs=-1)
                
                mean_accuracy = np.mean(accuracy_scores)
                mean_f1 = np.mean(f1_scores)
                
                composite_score = 0.7 * mean_accuracy + 0.3 * mean_f1
                
                performance_metrics[model_name] = {
                    'accuracy': mean_accuracy,
                    'f1': mean_f1,
                    'composite': composite_score,
                    'std': np.std(accuracy_scores)
                }
                
            except Exception as e:
                performance_metrics[model_name] = {
                    'accuracy': 0.5,
                    'f1': 0.3,
                    'composite': 0.4,
                    'std': 0.1
                }
        
        composite_scores = np.array([metrics['composite'] for metrics in performance_metrics.values()])
        
        exp_scores = np.exp(composite_scores * 8) 
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        stability_scores = 1 / (1 + np.array([metrics['std'] for metrics in performance_metrics.values()]))
        stability_weights = stability_scores / np.sum(stability_scores)
        
        final_weights = 0.8 * softmax_weights + 0.2 * stability_weights ## 8:2로 해도 상당히 많이 강건하게 나오네
        
        for i, model_name in enumerate(performance_metrics.keys()):
            weights[model_name] = final_weights[i]
            print(f"{model_name}: weight = {weights[model_name]:.4f}")
        
        return weights, performance_metrics
    
    def build_stacked_ensemble(self, models, data_dict):
        y = data_dict['y']
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        level1_predictions = np.zeros((len(y), len(models) * 6)) 
        
        col_idx = 0
        for model_name, (model, X_data) in models.items():
            print(f"  -> {model_name} Level-1 예측 생성 중...")
            
            fold_predictions = np.zeros((len(y), 6))
            
            for train_idx, val_idx in skf.split(X_data, y):
                X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
                y_train_fold = y[train_idx]
                
                model_clone = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_clone.fit(X_train_fold, y_train_fold)
                
                if hasattr(model_clone, 'predict_proba'):
                    val_pred_proba = model_clone.predict_proba(X_val_fold)
                    fold_predictions[val_idx] = val_pred_proba
                else:
                    val_pred = model_clone.predict(X_val_fold)
                    val_pred_proba = np.zeros((len(val_pred), 6))
                    for i, pred in enumerate(val_pred):
                        if 0 <= pred < 6:
                            val_pred_proba[i, pred] = 1.0
                    fold_predictions[val_idx] = val_pred_proba
            
            level1_predictions[:, col_idx:col_idx+6] = fold_predictions
            col_idx += 6
        
        level2_models = {
            'lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='mlogloss')
        }
        
        for name, model in level2_models.items():
            model.fit(level1_predictions, y)
        
        self.level2_models = level2_models
        
        return level1_predictions
    
    def fit(self, df):
        self.training_info = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': df.shape,
            'target_distribution': df['UE_AMT_GRP_encoded'].value_counts().to_dict(),
            'feature_engineering': self.use_feature_engineering,
            'hyperparameter_optimization': self.use_hyperopt
        }
        
        data_dict = self.prepare_enhanced_data(df)
        y = data_dict['y']
        
        print(f"len(data_dict['feature_names']): {len(data_dict['feature_names'])}")
        print(f"len(data_dict['binary_features']) if data_dict['binary_features'] is not None else 0: {len(data_dict['binary_features']) if data_dict['binary_features'] is not None else 0}")
        print(f"len(data_dict['continuous_features']) if data_dict['continuous_features'] is not None else 0: {len(data_dict['continuous_features']) if data_dict['continuous_features'] is not None else 0}")
        print(f"data_dict['X_embedding'] is not None: {'yess' if data_dict['X_embedding'] is not None else 'nope'}")
        
        models = self.build_optimized_models(data_dict)
        print(f"len(models): {len(models)}")
        
        trained_models = {}
        for model_name, (model, X_data) in models.items():
            print(f"{model_name}: {X_data.shape}")
            try:
                model.fit(X_data, y)
                trained_models[model_name] = (model, X_data)
            except Exception as e:
                print(f"{model_name}\n\n{e}")
        
        self.models = trained_models
        
        self.weights, performance_metrics = self.calculate_advanced_weights(self.models, y)
        
        level1_predictions = self.build_stacked_ensemble(self.models, data_dict)
        
        self.meta_model.fit(level1_predictions, y)
        
        self.is_fitted = True
        
        print(f"len(self.models): {len(self.models)}")
        print(f"len(self.level2_models): {len(self.level2_models)}")
        
        sorted_models = sorted(performance_metrics.items(), key=lambda x: x[1]['composite'], reverse=True)
        for model_name, metrics in sorted_models:
            print(f"{model_name}: {metrics['composite']} / weight: {self.weights.get(model_name, 0)}")
        
        return self
    
    def predict_proba(self, df):
        if not self.is_fitted:
            raise ValueError("not self.is_fitted")
        
        data_dict = self.prepare_enhanced_data(df)
        
        all_predictions = []
        weights_list = []
        
        for model_name, (model, _) in self.models.items():
            if 'embed_pca' in model_name:
                X_data = self.embedding_reducers['pca'].transform(data_dict['X_embedding'])
            elif 'embed_svd' in model_name: ### svd <<<<<<< 왤케 빡셈
                X_data = self.embedding_reducers['svd'].transform(data_dict['X_embedding'])
            elif any(x in model_name for x in ['nn_', 'svm_', 'knn', 'qda', 'logistic']):
                if 'robust' in model_name: ## <<< ?????????????????????????
                    X_data = self.scalers['robust'].transform(data_dict['X_all'])
                else:
                    X_data = self.scalers['standard'].transform(data_dict['X_all'])
            else:
                X_data = data_dict['X_all']
            
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_data)
                    all_predictions.append(pred_proba)
                    weights_list.append(self.weights.get(model_name, 1.0 / len(self.models)))
                else:
                    print(";;")
            except Exception as e:
                print(f"{model_name}\n\n{e}")
        
        if all_predictions:
            weights_array = np.array(weights_list)
            weights_array = weights_array / np.sum(weights_array)
            
            weighted_pred = np.zeros_like(all_predictions[0])
            for i, pred in enumerate(all_predictions):
                weighted_pred += weights_array[i] * pred
        else:
            raise ValueError("Valueerr")
        
        level1_predictions = np.zeros((len(df), len(self.models) * 6))
        col_idx = 0
        
        for model_name, (model, _) in self.models.items(): ## 얘까지 살짝 더 나눠볼-
            if 'embed_pca' in model_name:
                X_data = self.embedding_reducers['pca'].transform(data_dict['X_embedding'])
            elif 'embed_svd' in model_name:
                X_data = self.embedding_reducers['svd'].transform(data_dict['X_embedding'])
            elif any(x in model_name for x in ['nn_', 'svm_', 'knn', 'qda', 'logistic']):
                if 'robust' in model_name:
                    X_data = self.scalers['robust'].transform(data_dict['X_all'])
                else:
                    X_data = self.scalers['standard'].transform(data_dict['X_all'])
            else:
                X_data = data_dict['X_all']
            
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_data)
                    level1_predictions[:, col_idx:col_idx+6] = pred_proba
                col_idx += 6
            except Exception as e:
                print(f"{model_name}\n\n{e}")
                col_idx += 6
        
        level2_predictions = []
        for name, model in self.level2_models.items():
            try:
                pred_proba = model.predict_proba(level1_predictions)
                level2_predictions.append(pred_proba)
            except Exception as e:
                print(f"{name}\n\n{e}")
        
        if level2_predictions:
            stacked_pred = np.mean(level2_predictions, axis=0)
        else:
            stacked_pred = weighted_pred
        
        meta_pred = self.meta_model.predict_proba(level1_predictions)
        
        final_pred = 0.4 * weighted_pred + 0.35 * stacked_pred + 0.25 * meta_pred ## 더 조정해봐야
        
        return final_pred
    
    def predict(self, df):
        pred_proba = self.predict_proba(df)
        return np.argmax(pred_proba, axis=1)
    
    def evaluate(self, df_test, y_test):
        y_pred = self.predict(df_test)
        y_pred_proba = self.predict_proba(df_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"accuracy: {accuracy}")
        print(f"f1_macro: {f1_macro}")
        print(f"f1_weighted: {f1_weighted}")
        
        print(classification_report(y_test, y_pred))
        
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    
    def save_model(self, filepath=None):
        if not self.is_fitted:
            raise ValueError("not self.is_fitted")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'enhanced_ensemble_model_{timestamp}.pkl'
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        try:
            joblib.dump(self, filepath)
            print(f"{filepath}")
            return filepath
            
        except Exception as e:
            print(f"{e}")
            raise
    
    @classmethod
    def load_model(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath}")
        
        try:
            model = joblib.load(filepath)
            
            if not isinstance(model, cls):
                raise ValueError("not cls")
            
            print(f"{filepath}")
            return model
            
        except Exception as e:
            print(f"{e}")
            raise

def main():
    from process_df import process_df

    df = process_df()

    print(f"df.shape: {df.shape}")
    print(df['UE_AMT_GRP_encoded'].value_counts().sort_index())

    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['UE_AMT_GRP_encoded']
    )

    print(f"train_df.shape: {train_df.shape}")
    print(f"test_df.shape: {test_df.shape}")

    enhanced_ensemble = OptimizedEnsembleModel(
        random_state=42,
        use_feature_engineering=True,
        use_hyperopt=True
    )

    enhanced_ensemble.fit(train_df)

    y_test = test_df['UE_AMT_GRP_encoded'].values
    results = enhanced_ensemble.evaluate(test_df, y_test)

    saved_path = enhanced_ensemble.save_model()
    
    return enhanced_ensemble, results, saved_path

if __name__ == "__main__":
    enhanced_model, evaluation_results, model_path = main() 
    print(model_path)
    print(evaluation_results)