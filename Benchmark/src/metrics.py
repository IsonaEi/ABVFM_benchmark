
import numpy as np
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist

class MetricsEngine:
    def __init__(self):
        pass

    def compute_ssi(self, features, labels, window=15):
        """
        Computes State Stability Index (SSI) around transitions.
        SSI = Inter-state Distance / Intra-state Variance
        """
        # Identify transitions
        y = labels
        transitions = np.where(y[1:] != y[:-1])[0] + 1
        
        ssi_scores = []
        for t in transitions:
            # Define Pre and Post windows
            pre_idx =  max(0, t - window)
            post_idx = min(len(features), t + window)
            
            # Check if we have enough data
            if t - pre_idx < 2 or post_idx - t < 2: continue
            
            pre_feats = features[pre_idx:t]
            post_feats = features[t:post_idx]
            
            # Centroids
            pre_mean = np.mean(pre_feats, axis=0)
            post_mean = np.mean(post_feats, axis=0)
            
            # Inter-state Distance (Euclidean between centroids)
            inter_dist = np.linalg.norm(pre_mean - post_mean)
            
            # Intra-state Variance (Mean distance to own centroid)
            pre_var = np.mean(np.linalg.norm(pre_feats - pre_mean, axis=1))
            post_var = np.mean(np.linalg.norm(post_feats - post_mean, axis=1))
            intra_var = (pre_var + post_var) / 2 + 1e-6 # Avoid div by zero
            
            ssi = inter_dist / intra_var
            ssi_scores.append(ssi)
            
        return ssi_scores if ssi_scores else []

    def compute_silhouette(self, features, labels, sample_size=5000):
        """Computes Silhouette Score (subsampled for speed)."""
        if len(np.unique(labels)) < 2: return -1.0
        
        # Subsample if too large
        if len(features) > sample_size:
            idx = np.random.choice(len(features), sample_size, replace=False)
            X = features[idx]
            y = labels[idx]
        else:
            X, y = features, labels
            
        return silhouette_score(X, y)

    def compute_cqd(self, dino_embeds, skeletal_embeds, labels):
        """
        Cluster Quality Degradation (CQD).
        Compares Silhouette Coefficient in DINO vs Skeletal space using SAME labels.
        """
        s_castle = self.compute_silhouette(dino_embeds, labels)
        s_keypoint = self.compute_silhouette(skeletal_embeds, labels)
        
        degradation = s_castle - s_keypoint
        return {
            's_castle': s_castle,
            's_keypoint': s_keypoint,
            'degradation': degradation
        }
        
    def compute_nmi(self, labels_true, labels_pred):
        """Computes Normalized Mutual Information."""
        # Ensure lengths match
        min_len = min(len(labels_true), len(labels_pred))
        return normalized_mutual_info_score(labels_true[:min_len], labels_pred[:min_len])

    def compute_mann_whitney(self, traces_dict, method_key='CASTLE'):
        """
        Comparises 'Peak Amplitude' of event-triggered traces using Mann-Whitney U test.
        traces_dict: {method_name: {metric_name: [peak_values_array]}}
        Returns: DataFrame with p-values comparing 'method_key' vs others.
        """
        from scipy.stats import mannwhitneyu
        import pandas as pd
        
        results = []
        
        # Check if baseline method exists
        if method_key not in traces_dict:
            return None
            
        baseline_metrics = traces_dict[method_key]
        
        for method, metrics in traces_dict.items():
            if method == method_key: continue
            
            for metric_name, values in metrics.items():
                if metric_name not in baseline_metrics: continue
                
                base_vals = baseline_metrics[metric_name]
                curr_vals = values
                
                # drop nans caused by insufficient data
                base_vals = base_vals[~np.isnan(base_vals)]
                curr_vals = curr_vals[~np.isnan(curr_vals)]
                
                if len(base_vals) < 5 or len(curr_vals) < 5:
                    p_val = np.nan
                    stat = np.nan
                else:
                    # Alternative: greater? less? two-sided?
                    # "CASTLE peak > Others" => alternative='greater'
                    stat, p_val = mannwhitneyu(base_vals, curr_vals, alternative='two-sided')
                    
                results.append({
                    'Comparison': f"{method_key} vs {method}",
                    'Metric': metric_name,
                    'U-Stat': stat,
                    'p-value': p_val,
                    'Mean_Base': np.mean(base_vals) if len(base_vals)>0 else np.nan,
                    'Mean_Comp': np.mean(curr_vals) if len(curr_vals)>0 else np.nan
                })
                
        return pd.DataFrame(results)

    def get_event_triggered_traces(self, score, labels, window=45, return_peaks=False, filter_by_percentile=None):
        """
        Extracts traces of 'score' centered around label transitions.
        If return_peaks=True, returns the Peak Amplitude (max abs value in window) for each event.
        filter_by_percentile: If set (e.g. 10), only keeps events where peak amplitude is in top X%.
        """
        from scipy.stats import sem
        y = labels
        # Find transitions (start of new behavior)
        transitions = np.where(y[1:] != y[:-1])[0] + 1
        
        traces = []
        peaks = []
        valid_transitions = []
        
        # First Pass: Collect all raw traces and peaks
        raw_traces = []
        raw_peaks = []
        
        T_score = len(score)
        
        for t_idx in transitions:
            start, end = t_idx - window, t_idx + window
            if start >= 0 and end < T_score:
                segment = score[start:end]
                peak_val = np.max(np.abs(segment))
                
                raw_traces.append(segment)
                raw_peaks.append(peak_val)
                valid_transitions.append(t_idx)

        if not raw_traces:
             if return_peaks: return None, None, None, None
             return None, None, None

        raw_traces = np.array(raw_traces)
        raw_peaks = np.array(raw_peaks)
        
        # Filtering Logic
        if filter_by_percentile is not None and filter_by_percentile > 0:
            threshold = np.percentile(raw_peaks, 100 - filter_by_percentile)
            mask = raw_peaks >= threshold
            
            traces = raw_traces[mask]
            peaks = raw_peaks[mask]
            # valid_transitions filtered if needed, but not returned here
        else:
            traces = raw_traces
            peaks = raw_peaks
            
        if len(traces) == 0:
             if return_peaks: return None, None, None, None
             return None, None, None
        
        stack = traces # (N_events, Window)
        mean_trace = np.nanmedian(stack, axis=0) # Use Median for robustness
        sem_trace = sem(stack, axis=0, nan_policy='omit')
        
        if return_peaks:
            return mean_trace, sem_trace, stack, peaks
            
        return mean_trace, sem_trace, stack

    def compute_zscore(self, data):
        """Computes Z-Score of input array (ignoring NaNs)."""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0: return np.zeros_like(data)
        return (data - mean) / std

    def compute_label_stats(self, labels, fps):
        """
        Computes basic statistics for labels:
        - Number of Classes
        - Number of Transitions
        - Average Duration (ms)
        """
        n = len(labels)
        if n == 0: return {'n_classes': 0, 'n_transitions': 0, 'avg_duration_ms': 0}
        
        y = np.array(labels)
        # Find transitions
        mismatch = np.where(y[1:] != y[:-1])[0]
        n_transitions = len(mismatch)
        
        # Calculate durations
        run_ends = np.append(mismatch, n-1)
        run_starts = np.insert(run_ends[:-1] + 1, 0, 0)
        run_lengths = run_ends - run_starts + 1
        durations_sec = run_lengths / fps
        avg_duration_ms = np.mean(durations_sec) * 1000
        
        n_classes = len(np.unique(labels))
        
        return {
            'n_classes': n_classes,
            'n_transitions': n_transitions,
            'median_duration_ms': np.median(durations_sec) * 1000
        }

    def compute_reconstruction_score(self, source_features, target_features):
        """
        Superset Proof: Can we predict Target from Source?
        Train a Linear Regressor (Ridge) and return R^2 score.
        source: (N, D_src)
        target: (N, D_tgt)
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        
        # Determine sample size to avoid OOM or slow training
        max_samples = 10000
        if len(source_features) > max_samples:
            idx = np.random.choice(len(source_features), max_samples, replace=False)
            X = source_features[idx]
            y = target_features[idx]
        else:
            X = source_features
            y = target_features
            
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Ridge Regression
        # Alpha=1.0 is default, reasonable for this verification
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        
        # Evaluate R^2
        score = reg.score(X_test, y_test)
        return score

    def compute_pca_dimensionality(self, features, variance_threshold=0.90):
        """
        Intrinsic Dimensionality: How many PCs needed for 90% variance?
        """
        from sklearn.decomposition import PCA
        
        # PCA needs enough samples > n_components
        max_samples = 5000
        if len(features) > max_samples:
            idx = np.random.choice(len(features), max_samples, replace=False)
            X = features[idx]
        else:
            X = features
            
        # Full PCA or max 100 components to save time if dims are huge
        n_components = min(X.shape[1], 100)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        # Cumulative Variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Find threshold crossing
        n_dims = np.searchsorted(cum_var, variance_threshold) + 1
        return n_dims, cum_var

