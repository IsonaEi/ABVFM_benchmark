
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
import os

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = {
            'KPMS': '#1f77b4',       # Blue
            'CASTLE': '#d62728',     # Red
            'BSOID': '#2ca02c',      # Green
            'VAME': '#ff7f0e'        # Orange
        }

    def _set_style(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 16,
            'axes.labelsize': 20,
            'axes.titlesize': 22,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'lines.linewidth': 2.5
        })

    def plot_combined_traces(self, traces_dict, fps, title, ylabel, filename, colors=None, figsize=(10, 6)):
        """
        Plots combined event-triggered traces (Mean +/- SEM).
        traces_dict: { 'MethodName': {'mean': ..., 'sem': ...} }
        """
        self._set_style()
        fig, ax = plt.subplots(figsize=figsize)
        
        any_key = list(traces_dict.keys())[0]
        n_points = len(traces_dict[any_key]['mean'])
        t = (np.arange(n_points) - n_points // 2) / fps
        
        for name, data in traces_dict.items():
            mean = data['mean']
            sem_val = data['sem']
            
            clean_name = name.upper()
            color = self.colors.get(clean_name, 'gray')
            if colors and name in colors: color = colors[name]
            
            # User requested semi-transparent lines
            ax.plot(t, mean, label=clean_name, color=color, linewidth=2.5, alpha=0.8)
            ax.fill_between(t, mean - sem_val, mean + sem_val, color=color, alpha=0.15)
            
        ax.set_title(title, pad=20)
        ax.set_xlabel('Time from Transition (s)')
        ax.set_ylabel(ylabel)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    # ... (other methods) ...

    def plot_killer_case_comparison(self, data_dict, title, ylabel, filename):
        """
        Comparison of Killer Case Metrics (Residuals) across methods.
        Delegates to plot_violin_comparison to ensure consistent aesthetics (Violin + Median Labels) with SSI plot.
        """
        self.plot_violin_comparison(data_dict, title, ylabel, filename)

    def plot_ethogram_and_durations(self, methods_data, fps_ground_truth, filename_suffix="fig3_style"):
        """
        Mimics Fig 3a/b: Ethogram + State Duration Distribution.
        """
        self._set_style()
        n_methods = len(methods_data)
        fig = plt.figure(figsize=(15, 4 * n_methods))
        gs = fig.add_gridspec(n_methods, 4) 
        
        cmaps = {
            'KPMS': plt.cm.Blues,
            'CASTLE': plt.cm.Reds,
            'BSOID': plt.cm.Greens,
            'VAME': plt.cm.Oranges
        }
        
        all_durations = []
        for m in methods_data:
            y = m['labels']
            run_ends = np.where(y[1:] != y[:-1])[0]
            run_lengths = np.diff(np.concatenate(([0], run_ends, [len(y)-1]))) + 1
            durations = run_lengths / m['fps']
            all_durations.extend(durations)
        max_duration = np.percentile(all_durations, 99) if all_durations else 5.0
        
        for i, m in enumerate(methods_data):
            name = m['name']
            clean_name = name.upper()
            labels = m['labels']
            fps = m['fps']
            
            ax_eth = fig.add_subplot(gs[i, :3])
            max_frames = min(len(labels), int(60 * fps))
            display_labels = labels[:max_frames]
            unique_labels = np.unique(display_labels)
            label_map = {l: idx for idx, l in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in display_labels])
            cmap = cmaps.get(clean_name, plt.cm.Purples)
            barcode = mapped_labels[np.newaxis, :]
            
            ax_eth.imshow(barcode, aspect='auto', cmap=cmap, interpolation='nearest')
            ax_eth.set_title(f"{clean_name} Ethogram", loc='left')
            ax_eth.set_yticks([])
            ticks = np.linspace(0, max_frames, 7)
            ax_eth.set_xticks(ticks)
            ax_eth.set_xticklabels([f"{int(t/fps)}s" for t in ticks])
            if i == n_methods - 1: ax_eth.set_xlabel("Time (s)")
            
            ax_hist = fig.add_subplot(gs[i, 3])
            run_ends = np.where(labels[1:] != labels[:-1])[0]
            run_lengths = np.diff(np.concatenate(([0], run_ends, [len(labels)-1]))) + 1
            durations_sec = run_lengths / fps
            
            median_val = np.median(durations_sec)
            
            color = self.colors.get(clean_name, 'gray')
            # Updated: Linear Scale 0-2s, Binwidth 0.05s for finer resolution
            # Calculate percentage for Y-axis
            weights = np.ones_like(durations_sec) / len(durations_sec) * 100
            sns.histplot(x=durations_sec, ax=ax_hist, color=color, weights=weights, element='step', fill=True, log_scale=False, binwidth=0.05)
            
            ax_hist.set_xlim(0, 2.0)
            ax_hist.set_title("State Duration")
            ax_hist.set_xlabel("Duration (s)")
            if i == 0: ax_hist.set_ylabel("Probability (%)")
            else: ax_hist.set_ylabel("")
            
            ax_hist.axvline(median_val, color='k', linestyle='--', linewidth=2, label=f'Median: {median_val*1000:.0f}ms')
            ax_hist.text(median_val + 0.1, ax_hist.get_ylim()[1]*0.8, f'{median_val*1000:.0f}ms', color='k', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"benchmark_{filename_suffix}.png"), dpi=300)
        plt.close()

    def plot_violin_comparison(self, data_dict, title, ylabel, filename, colors=None, baseline_mean=None):
        self._set_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(data_dict.keys())
        valid_labels = [l for l in labels if len(data_dict[l]) > 0 and not np.all(np.isnan(data_dict[l]))]
        if not valid_labels:
            plt.close()
            return
        values = [data_dict[l] for l in valid_labels]
        parts = ax.violinplot(values, showmeans=True, showmedians=False, showextrema=False, points=100)
        
        for i, pc in enumerate(parts['bodies']):
            method = valid_labels[i]
            color = self.colors.get(method.upper(), 'gray')
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        if 'cmeans' in parts:
             parts['cmeans'].set_edgecolor('black')
             parts['cmeans'].set_linewidth(1.5)
             
        # Add labels for means
        means = [np.mean(v) for v in values]
        for i, m in enumerate(means):
             # Position slightly above the mean line
             ax.text(i+1, m, f'{m:.3f}', ha='center', va='bottom', fontweight='bold', color='black', fontsize=12)

        ax.set_xticks(np.arange(1, len(valid_labels) + 1))
        ax.set_xticklabels([l.upper() for l in valid_labels])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        all_vals = np.concatenate(values)
        if len(all_vals) > 0:
             # User request: Focus on middle 99.98% (Remove top/bottom 0.01%)
             p_min, p_max = np.percentile(all_vals, [0.01, 99.9])
             ax.set_ylim(bottom=max(0, p_min), top=p_max)
             
        if baseline_mean is not None:
             ax.axhline(baseline_mean, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=0, label=f'Baseline Mean: {baseline_mean:.3f}')
             ax.legend(loc='upper right', frameon=True) # Ensure legend is visible
                 
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_sorted_confusion_matrix(self, cm, row_classes, col_classes, row_name, col_name, filename):
        """
        Plots a Row-Normalized Confusion Matrix, sorted by hierarchical clustering 
        to group co-occurring behaviors.
        """
        self._set_style()
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        
        # 1. Row Normalization (P(Col | Row))
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=row_sums!=0)
        
        # 2. Sorting (Hierarchical Clustering)
        # Sort Rows
        if cm.shape[0] > 1:
            try:
                # Use euclidean distance between probability distributions
                row_linkage = linkage(pdist(cm_norm, metric='euclidean'), method='ward')
                row_order = leaves_list(row_linkage)
            except Exception as e:
                print(f"Warning: Row sorting failed ({e}), using default order.")
                row_order = np.arange(cm.shape[0])
        else:
             row_order = np.array([0])
             
        # Sort Cols (Cluster columns based on how they respond to rows)
        if cm.shape[1] > 1:
            try:
                col_linkage = linkage(pdist(cm_norm.T, metric='euclidean'), method='ward')
                col_order = leaves_list(col_linkage)
            except Exception as e:
                print(f"Warning: Col sorting failed ({e}), using default order.")
                col_order = np.arange(cm.shape[1])
        else:
            col_order = np.array([0])
            
        cm_sorted = cm_norm[row_order][:, col_order]
        row_labels = [str(row_classes[i]) for i in row_order]
        col_labels = [str(col_classes[i]) for i in col_order]
        
        # 3. Plotting
        # Dynamic size: at least 8x8, add space for labels
        w = max(10, len(col_classes) * 0.4)
        h = max(8, len(row_classes) * 0.4)
        fig, ax = plt.subplots(figsize=(w, h)) 
        
        sns.heatmap(cm_sorted, annot=False, cmap='Blues', ax=ax, square=True,
                    xticklabels=col_labels, yticklabels=row_labels,
                    cbar_kws={'label': f'P({col_name} | {row_name})'})
                    
        ax.set_title(f"Confusion Matrix: {row_name} vs {col_name} (Sorted)")
        ax.set_xlabel(f"{col_name} Labels")
        ax.set_ylabel(f"{row_name} Labels")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_label_duration_histogram(self, labels, fps, method_name, filename):
        """
        Plots a histogram of total duration (in seconds) for each class label,
        sorted by duration in descending order.
        """
        self._set_style()
        
        # Count frames per label
        unique_labels, counts = np.unique(labels, return_counts=True)
        durations = counts / fps
        
        # Sort by duration descending
        sort_idxs = np.argsort(durations)[::-1]
        sorted_labels = unique_labels[sort_idxs]
        sorted_durations = durations[sort_idxs]
        
        # Convert labels to string for plotting
        sorted_labels_str = [str(l) for l in sorted_labels]
        
        # Plot
        # Dynamic width based on number of classes
        w = max(10, len(sorted_labels) * 0.3)
        fig, ax = plt.subplots(figsize=(w, 6))
        
        color = self.colors.get(method_name.upper(), '#1f77b4')
        
        # Calculate percentage
        total_duration = np.sum(durations)
        percents = (sorted_durations / total_duration) * 100
        
        sns.barplot(x=sorted_labels_str, y=percents, color=color, ax=ax)
        
        ax.set_title(f"{method_name} - Class Abundance")
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Percentage (%)")
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        if len(sorted_labels) > 20:
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_pca_variance(self, pca_data, filename):
        self._set_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'DINO': '#d62728', 'Keypoints': '#1f77b4'}
        for name, data in pca_data.items():
            cum_var = data['cum_var']
            n_90 = data['n_90']
            color = colors.get(name, 'black')
            x = np.arange(1, len(cum_var) + 1)
            ax.plot(x, cum_var, label=f"{name} (90% at PC-{n_90})", color=color, linewidth=2.5)
            ax.scatter([n_90], [0.90], color=color, s=50, zorder=5)
            ax.axvline(n_90, color=color, linestyle=':', alpha=0.5)
        ax.axhline(0.90, color='gray', linestyle='--', label='90% Variance')
        ax.set_title("Intrinsic Dimensionality (PCA Variance)")
        ax.set_xlabel("Principal Component Index")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_reconstruction_scores(self, scores_dict, filename):
        self._set_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define categories and their corresponding labels/colors
        # Mapping from internal keys to display labels and colors
        mapping = {
            'DINO->KP': {'label': 'DINO -> KP', 'color': '#2ca02c'},   # Solid Green
            'Rand->KP': {'label': 'Random -> KP', 'color': '#94d2bd'}, # Light Green
            'KP->DINO': {'label': 'KP -> DINO', 'color': '#d62728'},   # Solid Red
            'Rand->DINO': {'label': 'Random -> DINO', 'color': '#ffadad'} # Light Red
        }
        
        plot_names = []
        plot_values = []
        plot_colors = []
        
        # Use explicit order if keys exist
        for key in ['DINO->KP', 'Rand->KP', 'KP->DINO', 'Rand->DINO']:
            if key in scores_dict:
                info = mapping.get(key, {'label': key, 'color': 'gray'})
                plot_names.append(info['label'])
                plot_values.append(max(0, scores_dict[key])) # R2 can be negative, clip for plot
                plot_colors.append(info['color'])
        
        # If any other keys exist, add them too
        for key in scores_dict:
            if key not in ['DINO->KP', 'Rand->KP', 'KP->DINO', 'Rand->DINO']:
                plot_names.append(key)
                plot_values.append(max(0, scores_dict[key]))
                plot_colors.append('gray')

        bars = ax.bar(plot_names, plot_values, color=plot_colors, alpha=0.8, edgecolor='black', width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("R² Score (Reconstruction Accuracy)")
        ax.set_title("Feature Completeness (Superset Proof & Chance Level)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_killer_case_comparison(self, data_dict, title, ylabel, filename):
        """
        Comparison of Killer Case Metrics (Residuals) across methods.
        Delegates to plot_violin_comparison to ensure consistent aesthetics (Violin + Median Labels) with SSI plot.
        """
        self.plot_violin_comparison(data_dict, title, ylabel, filename)
