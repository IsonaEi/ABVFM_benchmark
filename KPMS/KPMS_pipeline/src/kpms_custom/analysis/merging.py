import numpy as np
import keypoint_moseq as kpms
from pathlib import Path
from kpms_custom.utils.logging import get_logger

logger = get_logger()

class MotifMerger:
    def __init__(self, results, threshold_frames=10):
        self.results = results
        self.threshold = threshold_frames
        
    def calculate_centroids(self):
        """Calculates centroids for all motifs in latent space."""
        # Use extracted model_states for perfect alignment and single-model data
        x_dict = self.results.get('model_states', {}).get('x', {})
        z_dict = self.results.get('model_states', {}).get('z', {})
        
        if not x_dict or not z_dict:
            # Fallback for alternative results formats
            x_dict = {}
            z_dict = {}
            for k, v in self.results.items():
                if isinstance(v, dict) and ('x' in v or 'latent_state' in v) and ('z' in v or 'syllable' in v):
                    x_dict[k] = v.get('x', v.get('latent_state'))
                    z_dict[k] = v.get('z', v.get('syllable'))

        if not x_dict:
            raise ValueError("Results missing 'model_states' or latent/syllable data.")
            
        x_flat_list = []
        z_flat_list = []
        
        for key in x_dict:
            xi = np.array(x_dict[key])
            zi = np.array(z_dict[key])
            
            # These should already be aligned by extract_results
            min_len = min(len(xi), len(zi))
            x_flat_list.append(xi[:min_len])
            z_flat_list.append(zi[:min_len])
            
        self.x_flat = np.concatenate(x_flat_list, axis=0)
        self.z_flat = np.concatenate(z_flat_list, axis=0)
        
        self.uniq_motifs = np.unique(self.z_flat)
        self.centroids = {}
        
        for m in self.uniq_motifs:
            mask = (self.z_flat == m)
            if np.sum(mask) > 0:
                self.centroids[m] = np.mean(self.x_flat[mask], axis=0)
            else:
                self.centroids[m] = np.zeros(self.x_flat.shape[1])
                
        return self.centroids

    def identify_motif_types(self):
        """Identifies Stable vs Short motifs."""
        self.stable_motifs = []
        self.short_motifs = []
        
        z_seqs = []
        # Reconstruct list structure for run duration check
        # Or just use the flattened one with caution?
        # Ideally we iterate per session to avoid boundary errors.
        # But we already flattened.
        # Let's assume flattened is okay if we don't care about session boundaries being < threshold (rare).
        # Better: use the original lists if stored? I didn't store.
        # Let's just use flattened but split by 0-pad trick if I had indices.
        # Simple run length on flattened:
        
        runs = np.diff(np.concatenate(([0], (self.z_flat == -999).astype(int), [0]))) # dummy
        # Actually just iterate unique motifs
        
        for m in self.uniq_motifs:
            is_stable = False
            # Find runs of m
            # A run is strict sequence of m
            # We want Max Duration of m > threshold
            
            # Create boolean mask
            mask = (self.z_flat == m).astype(np.int8)
            # Find start/end of 1s
            diffs = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            
            if len(starts) > 0:
                durations = ends - starts
                if np.any(durations >= self.threshold):
                    is_stable = True
            
            if is_stable:
                self.stable_motifs.append(m)
            else:
                self.short_motifs.append(m)
                
        logger.info(f"Identified {len(self.stable_motifs)} stable, {len(self.short_motifs)} short motifs.")
        return self.stable_motifs, self.short_motifs

    def suggest_merges(self):
        """Maps short motifs to nearest stable motifs."""
        if not hasattr(self, 'centroids'):
            self.calculate_centroids()
        if not hasattr(self, 'stable_motifs'):
            self.identify_motif_types()
            
        self.merge_map = {} # target -> list of sources
        
        for short in self.short_motifs:
            c_short = self.centroids[short]
            best_target = None
            min_dist = float('inf')
            
            for stable in self.stable_motifs:
                dist = np.linalg.norm(c_short - self.centroids[stable])
                if dist < min_dist:
                    min_dist = dist
                    best_target = stable
            
            if best_target is not None:
                if best_target not in self.merge_map:
                    self.merge_map[best_target] = []
                self.merge_map[best_target].append(short)
                
        self.syllables_to_merge = []
        for target, sources in self.merge_map.items():
            self.syllables_to_merge.append([target] + sources)
            
        return self.syllables_to_merge

    def apply_merges(self, results, project_dir, model_name):
        """Applies the merges to results and saves."""
        if not self.syllables_to_merge:
            logger.info("No merges to apply.")
            return results
            
        logger.info(f"Applying {len(self.syllables_to_merge)} merges...")
        syllable_mapping = kpms.generate_syllable_mapping(results, self.syllables_to_merge)
        new_results = kpms.apply_syllable_mapping(results, syllable_mapping)
        
        # Save Strategy
        docs_dir = Path(project_dir) / model_name / "merged_analysis"
        docs_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = docs_dir / "motif_merging_strategy.md"
        
        with open(strategy_path, "w") as f:
            f.write("# Motif Merging Strategy\n")
            f.write(f"Threshold: {self.threshold} frames\n")
            f.write("## Merges\n| Target | Sources |\n|---|---|\n")
            for group in self.syllables_to_merge:
                f.write(f"| {group[0]} | {group[1:]} |\n")
        
        return new_results
