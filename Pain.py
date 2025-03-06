import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from scipy import stats
from scipy.signal import welch, find_peaks
import pickle
import gc
import os
import shap

def extract_ecg_features(ecg_signal):
    if len(ecg_signal) < 2:
        return pd.Series({
            'ecg_rms': 0, 'ecg_peak_to_peak': 0, 'ecg_power': 0,
            'ecg_kurtosis': 0, 'ecg_skewness': 0, 'rr_mean': 0,
            'rr_std': 0, 'rr_pnn50': 0, 'rr_rmssd': 0
        })

    rr_intervals = np.diff(ecg_signal)

    features = {
        'ecg_rms': np.sqrt(np.mean(np.square(ecg_signal))),
        'ecg_peak_to_peak': np.ptp(ecg_signal),
        'ecg_power': np.mean(np.square(ecg_signal)),
        'ecg_kurtosis': stats.kurtosis(ecg_signal),
        'ecg_skewness': stats.skew(ecg_signal),
        'rr_mean': np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'rr_pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals),
        'rr_rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    }
    return pd.Series(features)

def extract_ecg_frequency_features(ecg_signal, fs=1000):
    if len(ecg_signal) < 2:
        return pd.Series({
            'ecg_vlf_power': 0, 'ecg_lf_power': 0,
            'ecg_hf_power': 0, 'ecg_lf_hf_ratio': 0
        })

    frequencies, psd = welch(ecg_signal, fs=fs)

    vlf_mask = (frequencies >= 0.003) & (frequencies < 0.04)
    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)

    vlf_power = np.trapz(psd[vlf_mask], frequencies[vlf_mask]) if np.any(vlf_mask) else 0
    lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
    hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0

    features = {
        'ecg_vlf_power': vlf_power,
        'ecg_lf_power': lf_power,
        'ecg_hf_power': hf_power,
        'ecg_lf_hf_ratio': lf_power/hf_power if hf_power != 0 else 0
    }
    return pd.Series(features)

def extract_hrv_features(ecg_signal, window_size=1000):
    if len(ecg_signal) < window_size:
        return pd.Series({
            'hrv_sdnn': 0, 'hrv_rmssd': 0, 'hrv_pnn50': 0,
            'hrv_triangular_index': 0, 'hrv_tinn': 0
        })

    peaks, _ = find_peaks(ecg_signal, distance=window_size//2)

    if len(peaks) < 2:
        return pd.Series({
            'hrv_sdnn': 0, 'hrv_rmssd': 0, 'hrv_pnn50': 0,
            'hrv_triangular_index': 0, 'hrv_tinn': 0
        })

    rr_intervals = np.diff(peaks)
    hist_heights = np.histogram(rr_intervals)[0]
    max_height = np.max(hist_heights) if len(hist_heights) > 0 else 1

    features = {
        'hrv_sdnn': np.std(rr_intervals),
        'hrv_rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
        'hrv_pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals),
        'hrv_triangular_index': len(rr_intervals) / max_height,
        'hrv_tinn': np.ptp(rr_intervals)
    }
    return pd.Series(features)

def add_derivatives_and_ecg_features(df, exclude_cols):
    derivatives = {}
    cols_to_derive = [col for col in df.columns if col not in exclude_cols]

    print("Calcolando le derivate per le seguenti colonne:", cols_to_derive)

    for col in cols_to_derive:
        derivatives[f'{col}_d1'] = np.gradient(df[col])

    df_derivatives = pd.DataFrame(derivatives)

    print("Calcolando features ECG avanzate...")
    window_size = 1000

    ecg_features_list = []
    for i in range(0, len(df), window_size):
        window = df['Ecg'].iloc[i:i+window_size]
        features = pd.concat([
            extract_ecg_features(window),
            extract_ecg_frequency_features(window),
            extract_hrv_features(window)
        ])
        ecg_features_list.append(features)

    ecg_features_expanded = []
    for features in ecg_features_list:
        ecg_features_expanded.extend([features] * min(window_size, len(df) - len(ecg_features_expanded)))

    ecg_features_expanded = ecg_features_expanded[:len(df)]
    ecg_features_df = pd.DataFrame(ecg_features_expanded)

    columns_to_keep = cols_to_derive + ['COVAS']
    df_original_subset = df[columns_to_keep]

    df_enhanced = pd.concat([df_original_subset, df_derivatives, ecg_features_df], axis=1)
    df_enhanced = df_enhanced.iloc[1:]

    print(f"Righe originali: {len(df)}")
    print(f"Righe dopo feature engineering: {len(df_enhanced)}")
    print(f"Numero totale di features: {len(df_enhanced.columns)-1}")

    return df_enhanced

def tree_to_c_array(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value.flatten()

    nodes = []
    for i in range(n_nodes):
        if children_left[i] == -1:  # foglia
            node = [0, 0.0, -1, -1, value[i]]
        else:
            node = [feature[i], threshold[i], children_left[i],
                   children_right[i], 0.0]
        nodes.append(node)

    return nodes

class DualExtraTreesPipeline:
    def __init__(self, n_estimators=25, random_state=17):
        self.classifier = ExtraTreesClassifier(
            n_estimators=10,
            random_state=23
        )
        self.regressor = ExtraTreesRegressor(
            n_estimators=50,
            random_state=4
        )
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_binary = (y != 0).astype(int)
        print("Training classifier...")
        self.classifier.fit(X_scaled, y_binary)
        mask_nonzero = y != 0
        X_nonzero = X_scaled[mask_nonzero]
        y_nonzero = y[mask_nonzero]
        print("Training regressor...")
        self.regressor.fit(X_nonzero, y_nonzero)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        binary_pred = self.classifier.predict(X_scaled)
        final_predictions = np.zeros(len(X))
        nonzero_mask = binary_pred == 1
        if np.any(nonzero_mask):
            final_predictions[nonzero_mask] = self.regressor.predict(X_scaled[nonzero_mask])
        return final_predictions
    
    def get_shap_values(self, X, model_type='both', sample_size=None):
        """
        Calcola i valori SHAP per il classificatore, il regressore o entrambi.
        
        Parameters:
        -----------
        X : array-like
            Dati di input
        model_type : str, optional (default='both')
            'classifier', 'regressor', o 'both'
        sample_size : int, optional
            Numero di campioni da utilizzare per il calcolo SHAP
            
        Returns:
        --------
        dict
            Dizionario contenente i valori SHAP e gli explainer
        """
        X_scaled = self.scaler.transform(X)
        
        if sample_size is not None:
            indices = np.random.choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
            X_sample = X_scaled[indices]
        else:
            X_sample = X_scaled
            
        results = {}
        
        if model_type in ['classifier', 'both']:
            print("Calculating SHAP values for classifier...")
            classifier_explainer = shap.TreeExplainer(self.classifier)
            classifier_shap_values = classifier_explainer.shap_values(X_sample)
            results['classifier'] = {
                'explainer': classifier_explainer,
                'shap_values': classifier_shap_values,
                'data': X_sample
            }
            
        if model_type in ['regressor', 'both']:
            print("Calculating SHAP values for regressor...")
            regressor_explainer = shap.TreeExplainer(self.regressor)
            regressor_shap_values = regressor_explainer.shap_values(X_sample)
            results['regressor'] = {
                'explainer': regressor_explainer,
                'shap_values': regressor_shap_values,
                'data': X_sample
            }
            
        return results
    
    def save_shap_summary(self, X, output_dir='shap_plots', model_type='both', 
                     sample_size=None, feature_names=None, dpi=300):
    	"""
    	Salva i summary plot SHAP per il classificatore, il regressore o entrambi.
    
    	Parameters:
    	-----------
    	X : array-like
        Dati di input
    	output_dir : str, optional (default='shap_plots')
        Directory dove salvare i plot
    model_type : str, optional (default='both')
        'classifier', 'regressor', o 'both'
    sample_size : int, optional
        Numero di campioni da utilizzare
    feature_names : list, optional
        Lista dei nomi delle feature
    dpi : int, optional (default=300)
        Risoluzione delle immagini salvate
    """
    	os.makedirs(output_dir, exist_ok=True)
    	shap_values = self.get_shap_values(X, model_type, sample_size)
    
    	# Converti feature_names in array numpy se non è None
    	feature_names_array = np.array(feature_names) if feature_names is not None else None
    
    	if model_type in ['classifier', 'both']:
        	plt.figure(figsize=(10, 8))
        	shap.summary_plot(
        	    shap_values['classifier']['shap_values'],
        	    shap_values['classifier']['data'],
        	    feature_names=feature_names_array,  # Usa l'array numpy
        	    show=False
        	)
        	plt.tight_layout()
        	plt.savefig(os.path.join(output_dir, 'classifier_summary.png'), 
        	           dpi=dpi, bbox_inches='tight')
        	plt.close()
        
    	if model_type in ['regressor', 'both']:
        	plt.figure(figsize=(10, 8))
        	shap.summary_plot(
        	    shap_values['regressor']['shap_values'],
        	    shap_values['regressor']['data'],
        	    feature_names=feature_names_array,  # Usa l'array numpy
        	    show=False
        	)
        	plt.tight_layout()
        	plt.savefig(os.path.join(output_dir, 'regressor_summary.png'), 
               	    dpi=dpi, bbox_inches='tight')
        	plt.close()
    	
    def save_shap_dependence(self, X, feature_idx, output_dir='shap_plots', 
                            model_type='both', sample_size=None, feature_names=None, dpi=300):
        """
        Salva i dependence plot SHAP per una feature specifica.
        
        Parameters:
        -----------
        X : array-like
            Dati di input
        feature_idx : int
            Indice della feature da analizzare
        output_dir : str, optional (default='shap_plots')
            Directory dove salvare i plot
        model_type : str, optional (default='both')
            'classifier', 'regressor', o 'both'
        sample_size : int, optional
            Numero di campioni da utilizzare
        feature_names : list, optional
            Lista dei nomi delle feature
        dpi : int, optional (default=300)
            Risoluzione delle immagini salvate
        """
        os.makedirs(output_dir, exist_ok=True)
        shap_values = self.get_shap_values(X, model_type, sample_size)
        feature_name = feature_names[feature_idx] if feature_names is not None else str(feature_idx)
    
        try:
            if model_type in ['classifier', 'both']:
                plt.figure(figsize=(10, 8))
                # Per il classificatore, prendi il primo output se è multi-class
                classifier_shap = shap_values['classifier']['shap_values']
                if isinstance(classifier_shap, list):
                    classifier_shap = classifier_shap[1]  # Prendi i valori per la classe positiva
                    
                shap.dependence_plot(
                    feature_idx,
                    classifier_shap,
                    shap_values['classifier']['data'],
                    feature_names=feature_names,
                    show=False,
                    interaction_index=None  # Disabilita il calcolo delle interazioni
                )
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'classifier_dependence_{feature_name}.png'), 
                           dpi=dpi, bbox_inches='tight')
                plt.close()
                
            if model_type in ['regressor', 'both']:
                plt.figure(figsize=(10, 8))
                shap.dependence_plot(
                    feature_idx,
                    shap_values['regressor']['shap_values'],
                    shap_values['regressor']['data'],
                    feature_names=feature_names,
                    show=False,
                    interaction_index=None  # Disabilita il calcolo delle interazioni
                )
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'regressor_dependence_{feature_name}.png'), 
                           dpi=dpi, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error creating dependence plot for feature {feature_name}: {str(e)}")
            # Fallback a scatter plot semplice se il dependence plot fallisce
            try:
                if model_type in ['classifier', 'both']:
                    plt.figure(figsize=(10, 8))
                    classifier_shap = shap_values['classifier']['shap_values']
                    if isinstance(classifier_shap, list):
                        classifier_shap = classifier_shap[1]
                    plt.scatter(shap_values['classifier']['data'][:, feature_idx], 
                              classifier_shap[:, feature_idx])
                    plt.xlabel(f'Feature value: {feature_name}')
                    plt.ylabel('SHAP value')
                    plt.title(f'Classifier SHAP values for {feature_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'classifier_dependence_{feature_name}.png'), 
                               dpi=dpi, bbox_inches='tight')
                    plt.close()
                    
                if model_type in ['regressor', 'both']:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(shap_values['regressor']['data'][:, feature_idx], 
                              shap_values['regressor']['shap_values'][:, feature_idx])
                    plt.xlabel(f'Feature value: {feature_name}')
                    plt.ylabel('SHAP value')
                    plt.title(f'Regressor SHAP values for {feature_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'regressor_dependence_{feature_name}.png'), 
                               dpi=dpi, bbox_inches='tight')
                    plt.close()
            except Exception as e2:
                print(f"Error creating fallback plot for feature {feature_name}: {str(e2)}")
            	    
            	    
    def plot_bland_altman(self, y_true, y_pred, title='Bland-Altman Plot', filename='ba_plot_endeca.png'):
        mean = np.mean([y_true, y_pred], axis=0)
        diff = y_pred - y_true
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        upper_loa = md + 1.96 * sd
        lower_loa = md - 1.96 * sd

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mean, diff, alpha=0.5, s=1)

        ax.axhline(md, color='k', linestyle='-', label='Mean difference')
        ax.axhline(upper_loa, color='r', linestyle='--', label='+1.96 SD')
        ax.axhline(lower_loa, color='r', linestyle='--', label='-1.96 SD')

        plt.ylim(md - 1.96 * sd-1, md + 1.96 * sd+1)
        ax.set_xlabel('Media dei valori (Predetti + Reali)/2')
        ax.set_ylabel('Differenza (Predetti - Reali)')
        ax.set_title(title)
        ax.legend()

        stats_text = f'Media diff: {md:.2f}\nSD diff: {sd:.2f}\n'
        stats_text += f'Upper LoA: {upper_loa:.2f}\nLower LoA: {lower_loa:.2f}'
        correlation = stats.pearsonr(mean, diff)[0]
        stats_text += f'\nCorrelazione: {correlation:.3f}'
        massimo = np.max(diff)
        minimo = np.min(diff)
        stats_text += f'\nMassima diff pos: {massimo:.3f}'
        stats_text += f'\nMassima diff neg: {minimo:.3f}'

        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')

        within_loa = np.sum((diff >= lower_loa) & (diff <= upper_loa))
        percent_within = (within_loa / len(diff)) * 100
        plt.annotate(f'{percent_within:.1f}% dei punti\nentro i LoA',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     horizontalalignment='right',
                     verticalalignment='bottom')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

        return md, sd, upper_loa, lower_loa

    def plot_residuals(self, y_true, y_pred, title='Residual Analysis', filename='residual_plot_endeca.png'):
        residuals = y_pred - y_true
        '''
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.scatter(y_pred, residuals, alpha=0.5, s=1)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')

        plt.subplot(122)
        plt.hist(residuals, bins=30)
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()
        '''
        plt.figure(figsize=(12, 4))
        plt.subplot(211)
        plt.plot(y_pred)
        plt.plot(y_true)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')

        plt.subplot(212)
        plt.plot(residuals)
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        gc.collect()


    def plot_feature_importance(self, X, model, title, filename):
        plt.figure(figsize=(15, 8))
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        plt.title(f"Feature Importances - {title}")
        plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix_endeca.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

    def plot_correlation_matrix(self, X, filename='correlation_heatmap_endeca.png'):
        plt.figure(figsize=(15, 12))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

    def generate_all_plots(self, X_test, y_test):
        print("\nGenerating plots...")
        y_pred = self.predict(X_test)
        y_binary_test = (y_test != 0).astype(int)
        y_binary_pred = (y_pred != 0).astype(int)

        self.plot_confusion_matrix(y_binary_test, y_binary_pred)
        self.plot_feature_importance(X_test, self.classifier,
                                   "Classifier (Zero vs Non-Zero)",
                                   'feature_importance_classifier_endeca.png')
        self.plot_feature_importance(X_test, self.regressor,
                                   "Regressor (Non-Zero Values)",
                                   'feature_importance_regressor_endeca.png')
        self.plot_correlation_matrix(X_test)

        mask_nonzero = y_test != 0
        y_test_nonzero = y_test[mask_nonzero]
        y_pred_nonzero = y_pred[mask_nonzero]

        self.plot_bland_altman(y_test, y_pred,
                             'Bland-Altman Plot: All Values',
                             'ba_summary_all_endeca.png')
        self.plot_bland_altman(y_test_nonzero, y_pred_nonzero,
                             'Bland-Altman Plot: Non-Zero Values',
                             'ba_summary_nonzero_endeca.png')

        self.plot_residuals(y_test, y_pred,
                          'Residual Analysis (All Values)',
                          'residual_analysis_all_endeca.png')
        self.plot_residuals(y_test_nonzero, y_pred_nonzero,
                          'Residual Analysis (Non-Zero Values)',
                          'residual_analysis_nonzero_endeca.png')

    def evaluate(self, X, y, generate_plots=True):
        X_scaled = self.scaler.transform(X)
        y_binary = (y != 0).astype(int)
        y_pred = self.predict(X)
        y_binary_pred = self.classifier.predict(X_scaled)

        print("\nClassification Metrics (Zero vs Non-Zero):")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_binary, y_binary_pred))
        print("\nClassification Report:")
        print(classification_report(y_binary, y_binary_pred))

        mask_true_nonzero = y != 0
        mask_pred_nonzero = y_pred != 0

        if np.any(mask_true_nonzero & mask_pred_nonzero):
            print("\nRegression Metrics (Non-Zero Cases Only):")
            y_true_nonzero = y[mask_true_nonzero & mask_pred_nonzero]
            y_pred_nonzero = y_pred[mask_true_nonzero & mask_pred_nonzero]

            rmse = np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero))
            r2 = r2_score(y_true_nonzero, y_pred_nonzero)

            print(f"RMSE: {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")

        if generate_plots:
            self.generate_all_plots(X, y)

    def export_model_to_c(self, output_file, feature_names=None):
        """Esporta il modello in un header file C"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.scaler.mean_.shape[0])]

        with open(output_file, 'w') as f:
            # Header
            f.write("#ifndef MODEL_PARAMS_H\n")
            f.write("#define MODEL_PARAMS_H\n\n")
            f.write("#include <stddef.h>\n\n")

            # Feature names enum
            f.write("// Feature indices\n")
            f.write("enum FeatureIndex {\n")
            for i, name in enumerate(feature_names):
                f.write(f"    {name.upper()} = {i},\n")
            f.write("};\n\n")

            # Parametri di scaling
            f.write("// Scaling parameters\n")
            f.write(f"#define N_FEATURES {len(self.scaler.mean_)}\n")

            f.write("static const float FEATURE_MEANS[] = {")
            f.write(", ".join([f"{x:.6f}f" for x in self.scaler.mean_]))
            f.write("};\n\n")

            f.write("static const float FEATURE_STDS[] = {")
            f.write(", ".join([f"{x:.6f}f" for x in self.scaler.scale_]))
            f.write("};\n\n")

            # Struttura dei classificatori
            f.write("// Classifier trees\n")
            f.write(f"#define N_CLASSIFIER_TREES {len(self.classifier.estimators_)}\n")
            f.write(f"#define MAX_NODES_PER_TREE {max(tree.tree_.node_count for tree in self.classifier.estimators_)}\n\n")

            for i, tree in enumerate(self.classifier.estimators_):
                nodes = tree_to_c_array(tree)
                f.write(f"static const float CLASSIFIER_TREE_{i}[][5] = {{\n")
                for node in nodes:
                    f.write("    {")
                    f.write(", ".join([f"{x:.6f}f" for x in node]))
                    f.write("},\n")
                f.write("};\n\n")

            # Array di puntatori ai classificatori
            f.write("static const float (*CLASSIFIER_TREES[])[5] = {\n")
            for i in range(len(self.classifier.estimators_)):
                f.write(f"    CLASSIFIER_TREE_{i},\n")
            f.write("};\n\n")

            # Struttura dei regressori
            f.write("// Regressor trees\n")
            f.write(f"#define N_REGRESSOR_TREES {len(self.regressor.estimators_)}\n")

            for i, tree in enumerate(self.regressor.estimators_):
                nodes = tree_to_c_array(tree)
                f.write(f"static const float REGRESSOR_TREE_{i}[][5] = {{\n")
                for node in nodes:
                    f.write("    {")
                    f.write(", ".join([f"{x:.6f}f" for x in node]))
                    f.write("},\n")
                f.write("};\n\n")

            # Array di puntatori ai regressori
            f.write("static const float (*REGRESSOR_TREES[])[5] = {\n")
            for i in range(len(self.regressor.estimators_)):
                f.write(f"    REGRESSOR_TREE_{i},\n")
            f.write("};\n\n")

            f.write("#endif // MODEL_PARAMS_H\n")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Caricamento dati
    print("Caricamento dati dal database...")
    conn = sqlite3.connect('database_completo.db')
    query = """
    SELECT * FROM dati_completi
    LIMIT (SELECT CAST(COUNT(*) * 1 AS INTEGER) FROM dati_completi)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Preparazione dati
    df['COVAS'] = pd.to_numeric(df['COVAS'], errors='coerce')
    df = df.dropna(subset=['COVAS'])
    df = df.select_dtypes(include=[np.number])

    # Colonne da escludere
    exclude_cols = ['Seconds', 'Tmp', 'File_ID', 'Eda_RB', 'Timestamp', 'Heater [C]',
                    'Heater_cleaned', 'Emg', 'Bvp', 'COVAS', 'Eda_E4']

    # Feature engineering
    print("\nApplicazione feature engineering...")
    df_enhanced = add_derivatives_and_ecg_features(df, exclude_cols)

    # Verifica e pulizia NaN
    print("\nVerifica valori mancanti:")
    nan_cols = df_enhanced.isna().sum()
    print("Colonne con NaN:", nan_cols[nan_cols > 0])
    df_enhanced = df_enhanced.dropna()

    # Features da rimuovere
    features_to_drop = [
        'COVAS',
        'Ecg',
        'Resp_d1',
        'Ecg_d1',
        'ecg_lf_hf_ratio',
        'rr_pnn50',
        'ecg_lf_power',
        'ecg_vlf_power',
        'ecg_hf_power'
    ]

    # Preparazione X e y
    X = df_enhanced.drop(features_to_drop, axis=1)
    y = df_enhanced['COVAS']

    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Bilanciamento del training set
    print("\nBilanciamento del dataset...")
    n_nonzero = len(y_train[y_train != 0])
    zero_indices = y_train[y_train == 0].index
    np.random.seed(42)
    sampled_zero_indices = np.random.choice(zero_indices, size=n_nonzero, replace=False)
    balanced_indices = np.concatenate([sampled_zero_indices, y_train[y_train != 0].index])

    n_nonzero_t = len(y_test[y_test != 0])
    zero_indices_t = y_test[y_test == 0].index
    np.random.seed(42)
    sampled_zero_indices_t = np.random.choice(zero_indices_t, size=n_nonzero_t, replace=False)
    balanced_indices_t = np.concatenate([sampled_zero_indices_t, y_test[y_test != 0].index])

    X_train_balanced = X_train.loc[balanced_indices]
    y_train_balanced = y_train.loc[balanced_indices]

    X_test_balanced = X_test.loc[balanced_indices_t]
    y_test_balanced = y_test.loc[balanced_indices_t]

    print(f"Dimensioni training set bilanciato: {X_train_balanced.shape}")
    print(f"Proporzione zeri nel training bilanciato: {(y_train_balanced == 0).mean():.2%}")

    print(f"Dimensioni training set bilanciato: {X_test_balanced.shape}")
    print(f"Proporzione zeri nel test bilanciato: {(y_test_balanced == 0).mean():.2%}")

    # Training e valutazione
    pipeline = DualExtraTreesPipeline()
    pipeline.fit(X_train_balanced, y_train_balanced)

    print("\nValutazione sul test set:")
    pipeline.evaluate(X_test_balanced, y_test_balanced, generate_plots=True)

    # Esportazione del modello
    print("\nEsportazione del modello in formato C...")
    feature_names = [col.lower().replace(' ', '_') for col in X.columns]
    pipeline.export_model_to_c('model_params_b.h', feature_names)

    # Salvataggio del modello Python
    pipeline.save('dual_extratrees_model_endeca.pkl')
    
    # Salva i summary plot per entrambi i modelli
    pipeline.save_shap_summary(X_train_balanced, output_dir='./', model_type='both', feature_names=feature_names)

    # Salva solo il summary plot del classificatore
    pipeline.save_shap_summary(X_train_balanced, output_dir='./', model_type='classifier', feature_names=feature_names)

    # Salva il dependence plot per una feature specifica (es. feature 0)
    for i in range(X_train_balanced.shape[1]):
        pipeline.save_shap_dependence(X_train_balanced, feature_idx=i, output_dir='./', model_type='both', feature_names=feature_names)
    
    # Per dataset grandi, usa un campione più piccolo
    pipeline.save_shap_summary(X_train_balanced, output_dir='shap_plots', sample_size=1000, feature_names=feature_names)

    print("\nCompleted")
