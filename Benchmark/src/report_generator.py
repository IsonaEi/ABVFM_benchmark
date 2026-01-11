
import os
import markdown
import base64
import cv2
import glob
import numpy as np
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.report_path = os.path.join(output_dir, "benchmark_report.html")
        self.md_content = ""
        
    def add_header(self, title):
        self.md_content += f"# {title}\n\n"
        self.md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
    def add_section(self, title, content):
        self.md_content += f"## {title}\n\n"
        self.md_content += f"{content}\n\n"
        
    def add_image(self, image_path, caption):
        """Embeds image as Base64 string."""
        if not os.path.exists(image_path):
            self.md_content += f"**[Image Missing: {os.path.basename(image_path)}]**\n\n"
            return
            
        try:
            # Resize/Compress to keep size down
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Could not read image")
            
            # Helper to resize if width > 1000
            h, w = img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                img = cv2.resize(img, (1000, int(h * scale)))
                
            # Encode to JPG
            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            self.md_content += f'<figure><img src="data:image/jpeg;base64,{b64_str}" alt="{caption}" style="max-width:100%;"><figcaption>{caption}</figcaption></figure>\n\n'
            
        except Exception as e:
            self.md_content += f"**[Error embedding image: {e}]**\n\n"

    def add_video_clips_section(self, clips_dir):
        self.md_content += "## Killer Case Visualization (Top 5 Residuals)\n\n"
        clips = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
        if not clips:
            self.md_content += "No video clips generated.\n\n"
            return
            
        self.md_content += "These clips show moments with high **Optical Flow residual** (movements not explained by skeletal velocity).\n\n"
        
        # We can't easily embed videos in a single HTML without consistent bloat, 
        # but for small 3s clips it might be okay. Let's try or just link them.
        # User asked for single file. 3s * 5 clips * 1MB ~ 5MB. Acceptable.
        
        for clip in clips:
            name = os.path.basename(clip)
            try:
                with open(clip, "rb") as video_file:
                    b64_video = base64.b64encode(video_file.read()).decode('utf-8')
                
                self.md_content += f"### {name}\n"
                self.md_content += f'<video width="640" height="480" controls><source src="data:video/mp4;base64,{b64_video}" type="video/mp4">Your browser does not support the video tag.</video>\n\n'
            except Exception as e:
                self.md_content += f"**[Error embedding video {name}: {e}]**\n\n"

    def generate(self):
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max_width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                figure {{ margin: 20px 0; background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; }}
                img {{ box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                figcaption {{ margin-top: 10px; font-style: italic; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
        
        # Convert Markdown to HTML
        html_content = markdown.markdown(self.md_content, extensions=['tables', 'fenced_code'])
        
        full_html = html_template.format(content=html_content)
        
        print(f"Generating HTML report at {self.report_path}...")
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        return self.report_path
