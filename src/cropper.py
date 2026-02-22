import math
from PIL import Image, ImageTk
import tkinter as tk
from utils import get_aspect_ratio

class Cropper:
    TARGET_DISPLAY_AREA = 1000 * 1000
    MIN_SCALE = 0.05
    MAX_SCALE = 8.0

    HANDLE_SIZE = 10

    def _canvas_to_px(self, x, y):
        return int(x / self.display_scale), int(y / self.display_scale)
    
    def _px_to_canvas(self, x, y):
        return float(x * self.display_scale), float(y * self.display_scale)
    
    def _unit_to_px(self, k):
        return int(k * self.base_dim[0]), int(k * self.base_dim[1])
    
    def _is_inside_crop_box(self, x, y):
        x1, y1, x2, y2 = self.canvas.coords(self.crop_box)
        return x1 < x < x2 and y1 < y < y2
    
    def _get_handle_hit(self, x, y):
        for i, handle in enumerate(self.handles):
            hx1, hy1, hx2, hy2 = self.canvas.coords(handle)
            if hx1 < x < hx2 and hy1 < y < hy2:
                return i
        return None

    def __init__(self, raw_img, w, h):
        self.raw_img = raw_img
        self.raw_dim = raw_img.size
        self.target_dim = (w, h)

        # Compute aspect ratio and base unit (for regularised scaling)
        self.aspect = get_aspect_ratio(self.target_dim[0], self.target_dim[1])
        wh_gcd = math.gcd(self.target_dim[0], self.target_dim[1])
        self.base_dim = (self.target_dim[0] // wh_gcd, self.target_dim[1] // wh_gcd)

        # UI and state variables
        self.mode = None # "move" or "resize" (or None)
        self.active_handle = None # 0: TL, 1: TR, 2: BL, 3: BR
        self.init_mouse_pos = (0,0)
        self.init_rect = None
        self.accepted = False
        self.final_crop_box = None

        self.build_ui()

    def _get_display(self):
        scale = (self.TARGET_DISPLAY_AREA / (self.raw_dim[0] * self.raw_dim[1])) ** 0.5
        scale = max(self.MIN_SCALE, min(self.MAX_SCALE, scale))

        display_dim = (int(self.raw_dim[0] * scale), int(self.raw_dim[1] * scale))
        res_algo = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.NEAREST

        display_img = self.raw_img.resize(display_dim, res_algo)
        return {"width": display_dim[0], "height": display_dim[1], "img": display_img, "scale": scale}
    
    def build_ui(self):
        self.root = tk.Tk()
        self.root.title("Crop Image")
        
        display_data = self._get_display()
        self.display_img = display_data["img"]
        self.display_scale = display_data["scale"]
        self.display_dim = (display_data["width"], display_data["height"])

        self.canvas = tk.Canvas(self.root, width=display_data["width"], height=display_data["height"], cursor="cross")
        self.canvas.pack()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.overlays = [ # For shading non-crop area
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
        ]

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=6)
        tk.Button(btn_frame, text="Accept", command=self.on_accept).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT)

        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)

        self.init_crop_box()

    def init_crop_box(self):
        k = max(1, min(self.raw_dim[0] // self.base_dim[0], self.raw_dim[1] // self.base_dim[1]))
        rect_w, rect_h = self._unit_to_px(k)

        left = (self.raw_dim[0] - rect_w) // 2
        top = (self.raw_dim[1] - rect_h) // 2
        right = left + rect_w
        bottom = top + rect_h

        left_c, top_c = self._px_to_canvas(left, top)
        right_c, bottom_c = self._px_to_canvas(right, bottom)
        self.crop_box = self.canvas.create_rectangle(left_c, top_c, right_c, bottom_c, outline="red", width=2)
        self.handles = []
        self.draw_handles()
        self.update_overlay()

    def draw_handles(self):
        for handle in getattr(self, "handles", []):
            try:
                self.canvas.delete(handle)
            except Exception:
                pass
        self.handles = []
        x1, y1, x2, y2 = self.canvas.coords(self.crop_box)
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            handle = self.canvas.create_rectangle(cx - self.HANDLE_SIZE, cy - self.HANDLE_SIZE,
                                                  cx + self.HANDLE_SIZE, cy + self.HANDLE_SIZE,
                                                  fill="white", outline="black")
            self.handles.append(handle)

    def update_overlay(self):
        x1, y1, x2, y2 = self.canvas.coords(self.crop_box)
        w, h = self.display_dim
        self.canvas.coords(self.overlays[0], 0, 0, w, y1) # top
        self.canvas.coords(self.overlays[1], 0, y2, w, h) # bottom
        self.canvas.coords(self.overlays[2], 0, y1, x1, y2) # left
        self.canvas.coords(self.overlays[3], x2, y1, w, y2) # right

        for overlay in self.overlays:
            self.canvas.tag_raise(overlay, self.crop_box)

    def press(self, event):
        self.init_mouse_pos = (event.x, event.y)
        self.init_rect = tuple(self.canvas.coords(self.crop_box))
        handle_hit = self._get_handle_hit(event.x, event.y)
        if handle_hit is not None:
            self.mode = "resize"
            self.active_handle = handle_hit
        elif self._is_inside_crop_box(event.x, event.y):
            self.mode = "move"
        else:
            self.mode = None
            self.active_handle = None

    def drag(self, event):
        if not self.mode:
            return
        if self.mode == "move":
            self.move(event)
        elif self.mode == "resize":
            self.resize(event)

    def move(self, event):
        x1, y1, x2, y2 = self.init_rect
        init_dim = (x2 - x1, y2 - y1)
        
        nx1 = x1 + (event.x - self.init_mouse_pos[0])
        ny1 = y1 + (event.y - self.init_mouse_pos[1])

        left, top = self._canvas_to_px(nx1, ny1)

        crop_box_w = int(init_dim[0] / self.display_scale)
        crop_box_h = int(init_dim[1] / self.display_scale)

        k = max(1, min(crop_box_w // self.base_dim[0], crop_box_h // self.base_dim[1]))
        crop_box_w = int(k * self.base_dim[0])
        crop_box_h = int(k * self.base_dim[1])

        left = max(0, min(left, self.raw_dim[0] - crop_box_w))
        top = max(0, min(top, self.raw_dim[1] - crop_box_h))

        nx1, ny1 = self._px_to_canvas(left, top)
        nx2, ny2 = self._px_to_canvas(left + crop_box_w, top + crop_box_h)
        self.canvas.coords(self.crop_box, nx1, ny1, nx2, ny2)
        self.draw_handles()
        self.update_overlay()

    def resize(self, event):
        x1, y1, x2, y2 = self.init_rect
        handle_idx = self.active_handle
        fixed_corner = [(x2, y2), (x1, y2), (x2, y1), (x1, y1)][handle_idx]

        mouse_pos = self._canvas_to_px(event.x, event.y)
        fixed_pos = self._canvas_to_px(*fixed_corner)
        raw_w = abs(mouse_pos[0] - fixed_pos[0])
        k = max(1, int(raw_w / self.base_dim[0]))
        crop_box_dim = self._unit_to_px(k)
        if handle_idx in [0, 3]: # TL or BR
            new_corner = (fixed_pos[0] + crop_box_dim[0], fixed_pos[1] + crop_box_dim[1])
        else: # TR or BL
            new_corner = (fixed_pos[0] - crop_box_dim[0], fixed_pos[1] + crop_box_dim[1])

        if new_corner[0] < 0 or new_corner[1] < 0 or new_corner[0] > self.raw_dim[0] or new_corner[1] > self.raw_dim[1]:
            return
        if crop_box_dim[0] < self.base_dim[0] or crop_box_dim[1] < self.base_dim[1]:
            return

        new_corner_c = self._px_to_canvas(*new_corner)
        self.canvas.coords(self.crop_box, fixed_corner[0], fixed_corner[1], new_corner_c[0], new_corner_c[1])
        self.draw_handles()
        self.update_overlay()        

    def release(self, event):
        self.mode = None
        self.active_handle = None

    def on_accept(self):
        x1, y1, x2, y2 = self.canvas.coords(self.crop_box)
        left, top = self._canvas_to_px(x1, y1)
        right, bottom = self._canvas_to_px(x2, y2)
        self.final_crop_box = (left, top, right, bottom)
        self.accepted = True
        self.root.destroy()

    def on_cancel(self):
        self.accepted = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.accepted, self.final_crop_box