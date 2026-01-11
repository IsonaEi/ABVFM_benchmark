
import sys

with open('/home/isonaei/ABVFM_benchmark/B-soid/automated_pipeline/pipeline.py', 'r') as f:
    content = f.read()

marker = "    summary_results = []"
start_idx = content.find(marker)

if start_idx == -1:
    print("Marker not found!")
    sys.exit(1)

# Keep content up to marker (inclusive) -> actually up to end of that line to preserve variable init
end_of_line = content.find('\n', start_idx)
new_content = content[:end_of_line + 1]

# Append new loop code
new_code = r'''    
    # --- OUTER LOOP: Window Size (Requires B-SOiD Feats Re-calc) ---
    for win_frames in window_sizes_frames:
        # win_frames: desired window size in frames
        # B-SOiD hardcoded logic: window_frames = round(fps/10)
        # Therefore: fps_param = win_frames * 10
        fake_fps = win_frames * 10.0
        
        # Calculate Effective FPS for duration calculations
        effective_output_fps = (10.0 * base_fps) / fake_fps
        win_ms = (win_frames / base_fps) * 1000.0
        
        logging.info(f"=== Window Size: {win_frames} frames ({win_ms:.2f} ms) ===")
        logging.info(f"    Fake FPS Param: {fake_fps}")
        logging.info(f"    Effective Output FPS: {effective_output_fps:.2f} Hz")
        
        # 1. Extract Features (B-SOiD Feats)
        # This is the heavy step for Window Size changes
        logging.info("Extracting features with new window size...")
        try:
            f_10fps, f_10fps_sc = bsoid_feats(training_data, fps=fake_fps)
        except Exception as e:
            logging.error(f"Feature extraction failed for win_frames={win_frames}: {e}")
            continue

        # --- INNER LOOP: UMAP Parameters (Requires Embedding Re-calc) ---
        import itertools
        umap_grid = list(itertools.product(n_neighbors_list, min_dist_list))
        
        for n_neighbors, min_dist in umap_grid:
            combo_name = f"Win{win_frames}f_N{n_neighbors}_D{min_dist}"
            logging.info(f"--- Testing Combo: {combo_name} (n_neighbors={n_neighbors}, min_dist={min_dist}) ---")
            
            # 2. UMAP
            logging.info("Running UMAP...")
            # Create temp params dict for this run
            current_umap_params = config['umap_params'].copy()
            current_umap_params['n_neighbors'] = n_neighbors
            current_umap_params['min_dist'] = min_dist
            
            try:
                trained_umap, umap_embeddings = bsoid_umap_embed(f_10fps_sc, current_umap_params)
            except Exception as e:
                logging.error(f"UMAP failed for {combo_name}: {e}")
                continue
            
            # 3. Optimization & Clustering (HDBSCAN)
            logging.info("Running HDBSCAN Optimization...")
            
            # Update config for optimization duration calcs with effective FPS
            config_copy = config.copy()
            config_copy['fps'] = effective_output_fps 
            
            best_assignments, best_min_size = run_optimization(umap_embeddings, config_copy)
            
            # Calculate Clustering Quality Metrics
            import hdbscan
            try:
                final_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=best_min_size,
                    min_samples=best_min_size,
                    prediction_data=True
                ).fit(umap_embeddings)
                dbcv_score = final_clusterer.relative_validity_
            except Exception:
                dbcv_score = 0.0
                
            # --- MLP & Finalizing ---
            num_clusters_hdb = len(np.unique(best_assignments[best_assignments >= 0]))
            
            if num_clusters_hdb >= 2:
                final_labels, clf = train_classifier(f_10fps_sc, best_assignments, config)
                session_labels_source = final_labels
            else:
                session_labels_source = best_assignments
            
            # --- Saving Results ---
            run_output_dir = os.path.join(main_output_dir, combo_name)
            if not os.path.exists(run_output_dir):
                os.makedirs(run_output_dir)
                
            # Split back to files
            current_idx = 0
            results_dict = {}
            # Check safely if filenames is a nested list or flat
            if isinstance(filenames, list) and len(filenames) > 0 and isinstance(filenames[0], list):
                 flat_filenames = filenames[0]
            else:
                 flat_filenames = filenames

            for i, filename in enumerate(flat_filenames):
                # Recalculate feature length using FAKE FPS logic
                n_predictions = calc_feat_length(training_data[i].shape, fake_fps)
                
                # Check for bounds
                end_idx = current_idx + n_predictions
                if end_idx > len(session_labels_source): 
                     # This happens if calc is slightly off due to rounding
                     end_idx = len(session_labels_source)
                
                # Safe slice
                if current_idx < len(session_labels_source):
                     session_labels = session_labels_source[current_idx:end_idx]
                else:
                     session_labels = np.array([])

                results_dict[filename] = session_labels
                
                # Save Label CSV
                basename = os.path.basename(filename).replace('.csv', '')
                out_csv = os.path.join(run_output_dir, f"{basename}_labels.csv")
                
                time_axis = np.arange(len(session_labels)) * (1.0 / effective_output_fps)
                pd.DataFrame({
                    "Time": time_axis,
                    "B-SOiD_Label": session_labels
                }).to_csv(out_csv, index=False)
                
                current_idx = end_idx

            # Plot Ethogram
            plot_ethograms(results_dict, run_output_dir, f"ethogram_{combo_name}", fps=effective_output_fps)
            
            # --- Collect Statistics for Summary ---
            valid_labels = session_labels_source[session_labels_source >= 0]
            if len(valid_labels) > 0:
                num_clusters = len(np.unique(valid_labels))
                # Calculate mean duration from source (across all files)
                bouts = []
                curr = session_labels_source[0]
                count = 0
                bout_lens = []
                for l in session_labels_source:
                    if l == curr: count += 1
                    else:
                        if curr != -1: bout_lens.append(count)
                        curr = l
                        count = 1
                if curr != -1: bout_lens.append(count)
                
                if bout_lens:
                    bout_times_ms = (np.array(bout_lens) / effective_output_fps) * 1000
                    mean_dur = np.mean(bout_times_ms)
                    # Duration Violation rate (<100ms)
                    violation_rate = np.mean(bout_times_ms < 100)
                else:
                    mean_dur = 0
                    violation_rate = 0
            else:
                num_clusters = 0
                mean_dur = 0
                violation_rate = 0
                
            noise_ratio = np.sum(session_labels_source == -1) / len(session_labels_source)
            
            summary_results.append({
                "Run_ID": combo_name,
                "Win_Frames": win_frames,
                "Win_MS": win_ms,
                "Fake_FPS": fake_fps,
                "N_Neighbors": n_neighbors,
                "Min_Dist": min_dist,
                "Num_Clusters": num_clusters,
                "Mean_Duration_ms": mean_dur,
                "DBCV_Score": dbcv_score,
                "Noise_Ratio": noise_ratio,
                "Duration_Violation_Rate": violation_rate
            })
            
            logging.info(f"Run {combo_name} done. Clusters: {num_clusters}, DBCV: {dbcv_score:.3f}, Dur: {mean_dur:.1f}ms")

    # Generate Summary CSV
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv = os.path.join(main_output_dir, f"parameter_summary_{ts_run}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logging.info(f"Saved parameter summary CSV to {summary_csv}")
        
    logging.info(f"Pipeline completed. All results saved to {main_output_dir}")

if __name__ == "__main__":
    main()
'''

new_content += new_code

with open('/home/isonaei/ABVFM_benchmark/B-soid/automated_pipeline/pipeline.py', 'w') as f:
    f.write(new_content)
