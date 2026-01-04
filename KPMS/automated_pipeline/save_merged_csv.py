
import os
import sys
import h5py
import numpy as np
import pandas as pd
import keypoint_moseq as kpms
from pathlib import Path

def main():
    # --- Configuration ---
    project_dir = "/home/isonaei/ABVFM/KPMS/results/20260104_ctrl_30fps"
    models_dir = os.path.join(project_dir, "models")
    model_name = "20260104-053710-3"
    results_path = Path(models_dir) / model_name / "results_merged.h5"
    
    # Output dir for CSVs
    output_dir = Path(models_dir) / model_name / "merged_analysis" / "csv_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading merged results from {results_path}...")
    
    # Load H5 manually to get dictionary
    try:
        results = {}
        with h5py.File(results_path, 'r') as f:
            for session_key in f.keys():
                group = f[session_key]
                session_data = {}
                for k in group.keys():
                    val = group[k][()]
                    # Decode bytes to str if needed
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    session_data[k] = val
                results[session_key] = session_data
    except Exception as e:
        print(f"Error loading H5: {e}")
        return

    print(f"Saving CSVs to {output_dir}...")
    
    # Manual CSV saving to ensure control
    for session_name, data in results.items():
        try:
            # We primarily want 'syllable' (or 'z')
            # Data might have other keys like 'latent_state' etc.
            
            # Convert to DataFrame
            # Find the length from the 'syllable' or 'z' array
            length = 0
            if 'syllable' in data:
                length = len(data['syllable'])
            elif 'z' in data:
                length = len(data['z'])
            
            if length == 0:
                print(f"Skipping empty session {session_name}")
                continue
                
            df_data = {}
            for k, v in data.items():
                if isinstance(v, (np.ndarray, list)):
                    v = np.array(v)
                    if v.shape[0] == length and v.ndim == 1:
                        df_data[k] = v
                    elif v.shape[0] == length and v.ndim == 2:
                        # flatten 2d arrays (like latent state) into columns?
                        # Or just ignore common practice for simple CSV results.
                        # Usually user just wants the label.
                         for dim in range(v.shape[1]):
                            df_data[f"{k}_{dim}"] = v[:, dim]
            
            df = pd.DataFrame(df_data)
            
            # Save
            csv_path = output_dir / f"{session_name}.csv"
            df.to_csv(csv_path, index=False)
            # print(f"Saved {csv_path}") 
            
        except Exception as e:
            print(f"Error saving CSV for {session_name}: {e}")
            
    print(f"Done. Saved {len(results)} CSV files.")

if __name__ == "__main__":
    main()
