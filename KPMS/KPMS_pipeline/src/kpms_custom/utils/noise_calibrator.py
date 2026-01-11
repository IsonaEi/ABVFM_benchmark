import gradio as gr
import yaml
import numpy as np
import pandas as pd
import cv2
import os
import glob
import sys
from pathlib import Path

class NoiseCalibrator:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.config_path = self.project_dir / "config.yml" 
        # Fallback to default_config.yaml if config.yml doesn't exist (common in our structure)
        if not self.config_path.exists():
             self.config_path = self.project_dir.parent / "config" / "default_config.yaml"
             if not self.config_path.exists():
                 # Try the path passed in directly if it's a file
                 if os.path.isfile(project_dir):
                     self.config_path = Path(project_dir)
                     self.project_dir = self.config_path.parent
        
        print(f"Loading config from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.coordinates = {}
        self.confidences = {}
        self.bodyparts = []
        self.video_paths = {}
        
        # Sampling state
        self.sample_keys = [] # List of (video_name, frame_idx, bodypart)
        self.current_idx = 0
        self.annotations = {} # {(video, frame, bodypart): (x, y)}

    def load_data(self, data_dir=None, video_dir=None):
        if data_dir is None:
            data_dir = self.config.get('data_dir', '')
            # Fix if data_dir is empty or purely variable
            if not data_dir or data_dir == "''": 
                # Fallback to video_dir if data_dir missing
                data_dir = self.config.get('video_dir', '')
        
        if video_dir is None:
            video_dir = self.config.get('video_dir', '')
            
        print(f"Loading data from: {data_dir}")
        print(f"Video dir: {video_dir}")

        # Find .h5 files
        h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

        for h5_file in h5_files:
            try:
                # Read DLC h5 file
                df = pd.read_hdf(h5_file)
                # DLC structure: scorer -> bodyparts -> coords (x,y,likelihood)
                scorer = df.columns.get_level_values(0)[0]
                bps = df.columns.get_level_values(1).unique()
                if not self.bodyparts:
                    self.bodyparts = list(bps)
                
                # Extract data
                video_name = os.path.splitext(os.path.basename(h5_file))[0]
                # Try to clean up video name to match video file if possible
                # Often DLC appends scorer name etc.
                # Heuristic: look for matching video in video_dir
                
                coords = []
                confs = []
                
                for bp in self.bodyparts:
                    x = df[scorer][bp]['x'].values
                    y = df[scorer][bp]['y'].values
                    c = df[scorer][bp]['likelihood'].values
                    
                    coords.append(np.stack([x, y], axis=1))
                    confs.append(c)
                
                # Shape: (Frames, Bodyparts, 2)
                self.coordinates[video_name] = np.stack(coords, axis=1)
                # Shape: (Frames, Bodyparts)
                self.confidences[video_name] = np.stack(confs, axis=1)
                
                # Find matching video
                # Simple matching: look for video file that starts with the same prefix or is contained
                # This is a simplified version of kpms logic
                found_video = None
                possible_exts = ['.mp4', '.avi', '.mov']
                
                # specific fix for the user's current file if applicable
                # But generic logic:
                if os.path.exists(os.path.join(video_dir, video_name + '.mp4')):
                     found_video = os.path.join(video_dir, video_name + '.mp4')
                else:
                    # Search
                    for f in os.listdir(video_dir):
                        if any(f.endswith(ext) for ext in possible_exts):
                            # Check if h5 name contains video name or vice versa
                            # DLC often generates: VideoNameDLC_resnet...h5
                            # So video name is a prefix of h5 name
                            if video_name.startswith(os.path.splitext(f)[0]):
                                found_video = os.path.join(video_dir, f)
                                break
                            
                if found_video:
                    self.video_paths[video_name] = found_video
                else:
                    print(f"Warning: No matching video found for {video_name}")
                    
            except Exception as e:
                print(f"Error loading {h5_file}: {e}")

        print(f"Loaded {len(self.coordinates)} files.")
        print(f"Bodyparts: {self.bodyparts}")

    def sample_frames(self, num_samples=50, pseudocount=1e-3):
        # Flatten confidences to find distribution
        all_confs = []
        for v in self.confidences.values():
            all_confs.append(v.flatten())
        
        all_confs = np.concatenate(all_confs) + pseudocount
        
        # Log-space binning to sample low confidence frames
        min_conf, max_conf = np.min(all_confs), np.max(all_confs)
        bins = np.logspace(np.log10(min_conf), np.log10(max_conf), 11)
        
        candidates = []
        # Create candidate list: (video, frame, bodypart_idx)
        for vid_name, conf_arr in self.confidences.items():
            if vid_name not in self.video_paths: continue
            
            n_frames, n_bps = conf_arr.shape
            for f in range(0, n_frames, 10): # Stride to save time
                for b in range(n_bps):
                    c = conf_arr[f, b] + pseudocount
                    candidates.append({
                        'video': vid_name, 
                        'frame': f, 
                        'bp_idx': b, 
                        'conf': c
                    })
        
        df_cand = pd.DataFrame(candidates)
        df_cand['bin'] = pd.cut(df_cand['conf'], bins, labels=False)
        
        # Stratified sample
        samples = []
        samples_per_bin = num_samples // 10 + 1
        
        for b in range(10):
            bin_data = df_cand[df_cand['bin'] == b]
            if not bin_data.empty:
                n = min(len(bin_data), samples_per_bin)
                samples.append(bin_data.sample(n))
        
        if samples:
            final_sample = pd.concat(samples).sample(frac=1).reset_index(drop=True)
            # Use specific number
            final_sample = final_sample.head(num_samples)
            
            for _, row in final_sample.iterrows():
                bp_name = self.bodyparts[row['bp_idx']]
                self.sample_keys.append((row['video'], row['frame'], bp_name))
        
        print(f"Sampled {len(self.sample_keys)} frames.")
        self.current_idx = 0

    def get_current_frame(self):
        if not self.sample_keys:
            return None, "No Data"
        
        if self.current_idx >= len(self.sample_keys):
             return None, "Done"

        vid_name, frame_idx, bp_name = self.sample_keys[self.current_idx]
        vid_path = self.video_paths.get(vid_name)
        
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, f"Error reading frame {frame_idx} from {vid_name}"
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw current estimated position (Yellow Circle)
        bp_idx = self.bodyparts.index(bp_name)
        est_x, est_y = self.coordinates[vid_name][frame_idx, bp_idx]
        
        # Draw yellow circle
        cv2.circle(frame, (int(est_x), int(est_y)), 10, (255, 255, 0), 2)
        
        info = f"Frame {self.current_idx+1}/{len(self.sample_keys)}\nAcc: {self.config.get('added_noise_level', '?')}\nTarget: {bp_name}"
        return frame, info

    def save_annotation(self, click_coords):
        if self.current_idx >= len(self.sample_keys):
            return "Already done."
            
        vid_name, frame_idx, bp_name = self.sample_keys[self.current_idx]
        # click_coords should be (x, y)
        self.annotations[(vid_name, frame_idx, bp_name)] = click_coords
        return f"Saved {bp_name} at {click_coords}"

    def next_frame(self):
        if self.current_idx < len(self.sample_keys):
            self.current_idx += 1
        return self.get_current_frame()

    def prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
        return self.get_current_frame()

    def compute_and_save(self):
        if len(self.annotations) < 5:
            return "Need more annotations (at least 5)"
        
        errors = []
        log_confs = []
        
        for (vid, frame, bp), (user_x, user_y) in self.annotations.items():
            bp_idx = self.bodyparts.index(bp)
            mach_x, mach_y = self.coordinates[vid][frame, bp_idx]
            mach_conf = self.confidences[vid][frame, bp_idx]
            
            # Error = log10( dist + 1 )
            dist = np.sqrt((mach_x - user_x)**2 + (mach_y - user_y)**2)
            errors.append(np.log10(dist + 1))
            
            # Conf = log10( conf + pseudocount )
            log_confs.append(np.log10(mach_conf + 1e-3))
            
        # Linear Regression
        slope, intercept = np.polyfit(log_confs, errors, 1)
        
        # Update Config
        if 'error_estimator' not in self.config:
            self.config['error_estimator'] = {}
            
        self.config['error_estimator']['slope'] = float(slope)
        self.config['error_estimator']['intercept'] = float(intercept)
        
        # Save config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        return f"Calibration Saved!\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}\nSaved to {self.config_path}"

# --- Gradio App ---

def create_app(project_dir):
    calibrator = NoiseCalibrator(project_dir)
    calibrator.load_data()
    calibrator.sample_frames()

    with gr.Blocks(title="KPMS Noise Calibrator") as app:
        gr.Markdown("## KPMS Noise Calibrator (Custom)")
        
        with gr.Row():
            with gr.Column(scale=3):
                img_display = gr.Image(label="Click True Position", type="numpy", interactive=True)
            with gr.Column(scale=1):
                info_text = gr.Textbox(label="Info", lines=4)
                status_text = gr.Textbox(label="Status")
        
        with gr.Row():
            btn_prev = gr.Button("Previous")
            btn_next = gr.Button("Skip / Next")
            btn_save = gr.Button("Finish & Save Config", variant="primary")
            
        # States
        state_idx = gr.State(0)

        def refresh_ui():
            frame, info = calibrator.get_current_frame()
            if frame is None:
                if info == "Done":
                    return None, "All frames annotated!", "Ready to Save"
                else: 
                    return None, info, "Error"
            return frame, info, f"Annotating {calibrator.current_idx + 1}"

        def on_select(evt: gr.SelectData):
            # Gradio select returns [x, y] in evt.index
            coords = (evt.index[0], evt.index[1]) # x, y
            msg = calibrator.save_annotation(coords)
            calibrator.next_frame()
            frame, info, _ = refresh_ui()
            return frame, info, msg

        def on_next():
            calibrator.next_frame()
            return refresh_ui()
            
        def on_prev():
            calibrator.prev_frame()
            return refresh_ui()

        def on_finish():
            return calibrator.compute_and_save()

        # Wiring
        img_display.select(on_select, None, [img_display, info_text, status_text])
        btn_next.click(on_next, None, [img_display, info_text, status_text])
        btn_prev.click(on_prev, None, [img_display, info_text, status_text])
        btn_save.click(on_finish, None, [status_text])
        
        # Init
        app.load(refresh_ui, None, [img_display, info_text, status_text])
        
    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python noise_calibrator.py <project_folder_or_config>")
        # Default for debugging if running locally
        project_dir = "/home/isonaei/ABVFM_benchmark/KPMS/results/current_project"
    else:
        project_dir = sys.argv[1]
        
    app = create_app(project_dir)
    app.launch(server_name="127.0.0.1", server_port=7860)
