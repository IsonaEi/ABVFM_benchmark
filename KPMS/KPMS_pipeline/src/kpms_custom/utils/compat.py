from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

def patch_matplotlib_compatibility():
    """
    Patches FigureCanvasAgg to ensure tostring_rgb exists.
    Newer matplotlib versions removed it in favor of buffer_rgba.
    KPMS internally calls tostring_rgb().
    """
    if not hasattr(FigureCanvasAgg, 'tostring_rgb'):
        def tostring_rgb_safe(self):
            self.draw()
            w, h = self.get_width_height()
            buf = self.buffer_rgba()
            # Convert RGBA to RGB
            return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].tobytes()
            
        FigureCanvasAgg.tostring_rgb = tostring_rgb_safe
