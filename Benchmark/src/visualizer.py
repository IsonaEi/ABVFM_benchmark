
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
import os

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = {
            'KPMS': '#E63946',       # Red
            'CASTLE': '#457B9D',     # Blue
            'BSOID': '#2A9D8F',      # Green
            'RANDOM': '#424242'      # Gray/Black
        }

    def _get_clean_key(self, name):
        """Helper to normalize method names (remove hyphens, underscores)."""
        return name.upper().replace('-', '').replace('_', '')

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
            
            clean_key = self._get_clean_key(name)
            color = self.colors.get(clean_key, 'gray')
            if colors and name in colors: color = colors[name]
            
            ax.plot(t, mean, label=name.upper(), color=color, linewidth=2.5, alpha=0.8)
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
        Updated: Saturation depends on class frequency (Top classes = Fully saturated).
        """
        self._set_style()
        n_methods = len(methods_data)
        fig = plt.figure(figsize=(15, 4 * n_methods))
        gs = fig.add_gridspec(n_methods, 4) 
        
        # Base Colormaps for each method family
        base_cmaps = {
            'KPMS': plt.cm.Reds,
            'CASTLE': plt.cm.Blues,
            'BSOID': plt.cm.Greens,
            'RANDOM': plt.cm.Greys
        }
        
        all_durations = []
        for m in methods_data:
            y = m['labels']
            run_ends = np.where(y[1:] != y[:-1])[0]
            run_lengths = np.diff(np.concatenate(([0], run_ends, [len(y)-1]))) + 1
            durations = run_lengths / m['fps']
            all_durations.extend(durations)
        
        for i, m in enumerate(methods_data):
            name = m['name']
            clean_key = self._get_clean_key(name)
            labels = m['labels']
            fps = m['fps']
            
            # 1. Calc Class Frequencies for Saturation
            unique_labels, counts = np.unique(labels, return_counts=True)
            freq_map = {l: c for l, c in zip(unique_labels, counts)}
            
            # Sort labels by frequency
            sorted_by_freq = sorted(unique_labels, key=lambda l: freq_map[l], reverse=True)
            
            # Create a Custom Subplot-specific Colormap
            n_classes = len(unique_labels)
            base_cmap = base_cmaps.get(clean_key, plt.cm.Purples)
            
            # Map labels to intensities based on rank
            rank_map = {l: idx for idx, l in enumerate(sorted_by_freq)}
            
            def get_color_with_saturation(label):
                rank = rank_map.get(label, 0)
                # Map rank to intensity [0.3, 0.9]
                intensity = 0.9 - (rank / max(1, n_classes-1)) * 0.6
                return base_cmap(intensity)

            ax_eth = fig.add_subplot(gs[i, :3])
            max_frames = min(len(labels), int(60 * fps))
            display_labels = labels[:max_frames]
            
            # Convert to color array directly for imshow
            barcode_colors = np.array([get_color_with_saturation(l) for l in display_labels])
            barcode_colors = barcode_colors[np.newaxis, :, :] # (1, T, 4)
            
            ax_eth.imshow(barcode_colors, aspect='auto', interpolation='nearest')
            ax_eth.set_title(f"{name} Ethogram", loc='left', color=self.colors.get(clean_key, 'black'))
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
            
            color = self.colors.get(clean_key, 'gray')
            weights = np.ones_like(durations_sec) / len(durations_sec) * 100
            sns.histplot(x=durations_sec, ax=ax_hist, color=color, weights=weights, element='step', fill=True, log_scale=False, binwidth=0.05)
            
            ax_hist.set_xlim(0, 2.0)
            ax_hist.set_title("State Duration")
            ax_hist.set_xlabel("Duration (s)")
            if i == 0: ax_hist.set_ylabel("Probability (%)")
            else: ax_hist.set_ylabel("")
            
            ax_hist.axvline(median_val, color='k', linestyle='--', linewidth=2)
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
        
        # User Request: Adjust values to be "Difference from Random Mean"
        # Everything (including Random) is shifted by baseline_mean
        plot_values = []
        for l in valid_labels:
             vals = data_dict[l]
             if baseline_mean is not None:
                  plot_values.append(np.array(vals) - baseline_mean)
             else:
                  plot_values.append(vals)

        parts = ax.violinplot(plot_values, showmeans=True, showmedians=False, showextrema=False, points=100)
        
        for i, pc in enumerate(parts['bodies']):
            method = valid_labels[i]
            # Normalize key for coloring
            m_clean = self._get_clean_key(method)
            color = self.colors.get(m_clean, 'gray')
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        if 'cmeans' in parts:
             parts['cmeans'].set_edgecolor('black')
             parts['cmeans'].set_linewidth(1.5)
             
        # Add labels for means
        means = [np.mean(v) for v in plot_values]
        for i, m in enumerate(means):
             ax.text(i+1, m, f'{m:.3f}', ha='center', va='bottom', fontweight='bold', color='black', fontsize=12)

        ax.set_xticks(np.arange(1, len(valid_labels) + 1))
        ax.set_xticklabels([l.upper() for l in valid_labels])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        # Random Baseline Indicator (If we subtracted it, it should be at 0)
        if baseline_mean is not None:
             ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=0, label='Chance Level')
             ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_aligned_confusion_matrix(self, cm, row_classes, col_classes, row_name, col_name, filename, ari=None, nmi=None, normalize=True):
        """
        Plots a Confusion Matrix optimally aligned using the Hungarian Algorithm.
        If normalize=True, performs Row-Normalization (P(Col|Row)).
        Uses Linear Scale [0, 1] for conditional probability.
        """
        self._set_style()
        from scipy.optimize import linear_sum_assignment
        
        # 1. Normalization
        if normalize:
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            # Avoid division by zero
            cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=row_sums!=0)
            cbar_label = f'Probability P({col_name} | {row_name})'
        else:
            cm_norm = cm.astype('float')
            cbar_label = 'Counts'

        # 2. Hungarian Alignment (maximize diagonal)
        # We use the normalized matrix for alignment to find "best probability match"
        cost_matrix = cm_norm.max() - cm_norm
        
        # Pad matrix if non-square to ensure all smaller-dim clusters get assigned
        r, c = cm_norm.shape
        if r != c:
            dim = max(r, c)
            pad_cost = np.zeros((dim, dim))
            # Fill original cost
            pad_cost[:r, :c] = cost_matrix
            # Fill padding with very high cost to discourage matching (or 0 if neutral)
            # Actually linear_sum_assignment handles rectangular, but let's just use it directly
            # SciPy linear_sum_assignment solves min cost for rectangular inputs
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
        # Reorder Columns to match Rows
        # row_ind is usually 0..N, col_ind is the permutation
        # We want the columns `col_ind` to appear in the order of `row_ind`
        
        # If matrix is rectangular, we need to be careful.
        # Let's say Row 0 matches Col 5. We want Col 5 to be the 0-th column in the plot.
        
        permuted_col_indices = []
        used_cols = set()
        
        # Create a map from row_index -> matched_col_index
        row_to_col = {r: c for r, c in zip(row_ind, col_ind)}
        
        # Construct new column order
        sorted_rows = np.arange(r) # Keep rows as is (or could sort by frequency)
        
        for r_idx in sorted_rows:
            if r_idx in row_to_col:
                c_idx = row_to_col[r_idx]
                permuted_col_indices.append(c_idx)
                used_cols.add(c_idx)
            else:
                 # No match found (shouldn't happen for rectangular if R <= C)
                 pass
                 
        # Append unmatched columns
        for c_idx in range(c):
            if c_idx not in used_cols:
                permuted_col_indices.append(c_idx)
                
        # Apply permutation
        cm_sorted = cm_norm[:, permuted_col_indices]
        sorted_col_classes = [col_classes[i] for i in permuted_col_indices]
        
        # 3. Plotting
        w = max(12, len(col_classes) * 0.3)
        h = max(10, len(row_classes) * 0.3)
        fig, ax = plt.subplots(figsize=(w, h)) 
        
        # Linear Scale for Conditional Probability
        vmin = 0.0
        vmax = 1.0 
        
        # Annotation formatting: Show value only if > threshold to avoid clutter
        annot_data = np.where(cm_sorted > 0.01, cm_sorted, 0) # Show if > 1%
        
        sns.heatmap(cm_sorted, annot=annot_data, fmt='.2f', annot_kws={"size": 5},
                    cmap='Blues', ax=ax, square=False,
                    xticklabels=[str(c) for c in sorted_col_classes], 
                    yticklabels=[str(r) for r in row_classes],
                    vmin=vmin, vmax=vmax,
                    cbar_kws={'label': f'Joint Probability P({row_name}, {col_name})'})
                    
        title = f"Aligned Confusion Matrix: {row_name} vs {col_name}\n(Hungarian Alignment)"
        if nmi is not None:
             title += f" [NMI: {nmi:.3f}]"
             
        ax.set_title(title, pad=20)
        ax.set_xlabel(f"{col_name} Labels (Reordered)")
        ax.set_ylabel(f"{row_name} Labels")
        plt.xticks(rotation=90, fontsize=max(6, 12 - len(col_classes)//20))
        plt.yticks(rotation=0, fontsize=max(6, 12 - len(row_classes)//20))
        
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
        
        clean_key = self._get_clean_key(method_name)
        color = self.colors.get(clean_key, '#1f77b4')
        
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
        """
        Modified: Random baseline is now shown as a chance-level indicator on top of main bars.
        Ensured Y-axis scale covers the chance levels.
        """
        self._set_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['DINO -> KP', 'KP -> DINO']
        main_values = [scores_dict.get('DINO->KP', 0), scores_dict.get('KP->DINO', 0)]
        rand_values = [scores_dict.get('Rand->KP', 0), scores_dict.get('Rand->DINO', 0)]
        
        # Ensure we don't have negative R2 for plotting
        main_values = [max(0, v) for v in main_values]
        rand_values = [max(0.001, v) for v in rand_values] # Small positive for visibility
        
        colors = [self.colors['CASTLE'], self.colors['KPMS']]
        
        bars = ax.bar(categories, main_values, color=colors, alpha=0.8, edgecolor='black', width=0.5)
        
        # Max value for scaling
        max_seen = max(max(main_values), max(rand_values))
        
        # Draw Chance Level markers
        for i, (bar, rand_val) in enumerate(zip(bars, rand_values)):
            x = bar.get_x()
            w = bar.get_width()
            # Draw a dashed line for shuffle chance
            ax.hlines(rand_val, x, x + w, color='black', linestyles='--', linewidth=2, label='Shuffle Chance' if i==0 else "")
            
            # Label actual value
            ax.text(x + w/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')
            # Label chance value (avoid overlap if very close)
            offset = 0.05 if abs(bar.get_height() - rand_val) > 0.1 else -0.05
            ax.text(x + w/2, rand_val + offset, f'Chance: {rand_val:.3f}', ha='center', va='bottom' if offset > 0 else 'top', fontsize=12, color='#424242')

        ax.set_ylim(0, max(1.1, max_seen * 1.2))
        ax.set_ylabel("R² Score (Reconstruction Accuracy)")
        ax.set_title("Feature Completeness vs. Shuffle Chance")
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_residual_scatter(self, z_of, z_sv, residuals, filename):
        """
        New: Scatter plot for Killer Case (Optical Flow vs Skeleton Velocity).
        """
        self._set_style()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Subsample for plotting performance if needed
        if len(z_of) > 10000:
            idx = np.random.choice(len(z_of), 10000, replace=False)
            x, y = z_sv[idx], z_of[idx]
        else:
            x, y = z_sv, z_of
            
        hb = ax.hexbin(x, y, gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Point Density')
        
        # Regression Line
        # Re-calc slope briefly
        from scipy.stats import linregress
        res = linregress(z_sv, z_of)
        x_range = np.array([np.min(z_sv), np.max(z_sv)])
        ax.plot(x_range, res.slope * x_range + res.intercept, color='red', linestyle='--', linewidth=2, label=f'Linear Fit (Slope: {res.slope:.2f})')
        
        ax.set_xlabel("Skeleton Velocity (Z-Score)")
        ax.set_ylabel("Optical Flow Magnitude (Z-Score)")
        ax.set_title("Motion Gap: Flow vs Skeleton Discrepancy")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_killer_case_comparison(self, data_dict, title, ylabel, filename):
        """
        Comparison of Killer Case Metrics (Residuals) across methods.
        Delegates to plot_violin_comparison to ensure consistent aesthetics (Violin + Median Labels) with SSI plot.
        """
        self.plot_violin_comparison(data_dict, title, ylabel, filename)
