import math
import tkinter as tk
from PIL import Image, ImageTk

class Cropper:
    HANDLE_SIZE = 10
    MIN_UNITS = 1   # minimum size is 1 base unit (k >= 1)

    def __init__(self, pil_image, target_w, target_h, max_display_size=(1000,800), target_display_area=1000_000):
        self.orig_image = pil_image
        self.ow, self.oh = pil_image.size
        if self.ow <= 0 or self.oh <= 0:
            raise ValueError("Original image has invalid size")
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.max_display_size = max_display_size
        self.TARGET_DISPLAY_AREA = target_display_area

        # compute exact aspect and base unit in original pixels
        g = math.gcd(self.target_w, self.target_h)
        self.base_w = self.target_w // g
        self.base_h = self.target_h // g
        # aspect as float for some calculations
        self.aspect = self.target_w / self.target_h

        # UI and internal state
        self.crop_box_original = None
        self._accepted = False

        # build UI
        self.root = tk.Tk()
        self.root.title("Crop Image")
        tk.Label(self.root, text="Drag corners to resize; Drag inside box to move").pack(anchor="w")

        self.display_image, self.scale = self._scaled_display_image()
        self.w, self.h = self.display_image.size

        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, cursor="cross")
        self.canvas.pack()

        self.tkimg = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        self.overlay_parts = [
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
            self.canvas.create_rectangle(0, 0, 0, 0, fill="black", stipple="gray50", width=0),
        ]

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=6)
        tk.Button(btn_frame, text="Accept", command=self._on_accept).pack(side="right", padx=6)
        tk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side="right")

        # initial rectangle (largest centered, in original pixels then converted to canvas)
        self._create_initial_rect()

        # interaction state
        self.mode = None            # "move" or "resize" or None
        self.active_corner = None   # 0:TL,1:TR,2:BL,3:BR
        self.start_mouse = (0,0)
        self.start_rect = None     # canvas coords at drag start

        # bind events
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
    def _scaled_display_image(self):
        """
        Scale image so its displayed area is ~ TARGET_DISPLAY_AREA (canvas pixels).
        Choose NEAREST when scaling up (big pixels), LANCZOS when scaling down.
        Return (display_image, scale) where scale = display_pixels_per_original_pixel.
        """
        ow, oh = self.orig_image.size
        max_w, max_h = self.max_display_size

        scale = (self.TARGET_DISPLAY_AREA / (ow * oh)) ** 0.5
        MIN_SCALE = 0.05
        MAX_SCALE = 8.0
        scale = max(MIN_SCALE, min(MAX_SCALE, scale))

        disp_w = max(1, int(round(ow * scale)))
        disp_h = max(1, int(round(oh * scale)))

        # ensure fits window
        fit_scale_x = max_w / disp_w if disp_w > 0 else 1.0
        fit_scale_y = max_h / disp_h if disp_h > 0 else 1.0
        fit_scale = min(1.0, fit_scale_x, fit_scale_y)
        if fit_scale < 1.0:
            disp_w = int(round(disp_w * fit_scale))
            disp_h = int(round(disp_h * fit_scale))
            scale *= fit_scale

        # choose resampling
        if scale >= 1.0:
            resample = Image.Resampling.NEAREST
        else:
            resample = Image.Resampling.LANCZOS

        disp_img = self.orig_image.resize((disp_w, disp_h), resample=resample)
        return disp_img, scale

    def canvas_to_img_px(self, cx, cy):
        """Convert canvas (display) coords to nearest original-image pixel coordinates (integers)."""
        # inverse scale: original_pixel = canvas_coord / scale
        px = int(round(cx / self.scale))
        py = int(round(cy / self.scale))
        # clamp
        px = max(0, min(self.ow - 1, px))
        py = max(0, min(self.oh - 1, py))
        return px, py

    def img_px_to_canvas(self, px, py):
        """Convert original-image integer pixel coords to canvas coords (float)."""
        return float(px * self.scale), float(py * self.scale)

    def units_to_pixels(self, k):
        """Width and height in original pixels for k base-units."""
        return k * self.base_w, k * self.base_h

    def _create_initial_rect(self):
        """
        Choose the largest possible centered rectangle in original-image pixels that:
          - fits in (original_w, original_h),
          - has width = k * base_w, height = k * base_h for some integer k >= 1
        Then convert to canvas coords and draw.
        """
        # maximum k allowed by original image bounds
        max_k_w = self.ow // self.base_w
        max_k_h = self.oh // self.base_h
        max_k = min(max_k_w, max_k_h)
        if max_k < 1:
            max_k = 1

        # choose k so resulting canvas area is reasonably large but not exceeding display
        # we'll pick largest k that fits in original; that gives largest rectangle
        k = max_k

        rect_w_px, rect_h_px = self.units_to_pixels(k)

        # center in original-image pixel coordinates
        left_px = (self.ow - rect_w_px) // 2
        top_px = (self.oh - rect_h_px) // 2
        right_px = left_px + rect_w_px
        bottom_px = top_px + rect_h_px

        # convert to canvas coordinates
        left_c, top_c = self.img_px_to_canvas(left_px, top_px)
        right_c, bottom_c = self.img_px_to_canvas(right_px, bottom_px)

        self.rect = self.canvas.create_rectangle(left_c, top_c, right_c, bottom_c, outline="red", width=2)
        self.handles = []
        self._draw_handles()
        self._update_overlay()

    def _draw_handles(self):
        for h in getattr(self, "handles", []):
            try:
                self.canvas.delete(h)
            except Exception:
                pass
        self.handles = []
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        pts = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
        for px, py in pts:
            h = self.canvas.create_rectangle(px - self.HANDLE_SIZE, py - self.HANDLE_SIZE,
                                             px + self.HANDLE_SIZE, py + self.HANDLE_SIZE,
                                             fill="white", outline="black")
            self.handles.append(h)

    def _inside_rect(self, x, y):
        x1,y1,x2,y2 = self.canvas.coords(self.rect)
        return x1 < x < x2 and y1 < y < y2

    def _handle_hit(self, x, y):
        for i, h in enumerate(self.handles):
            x1,y1,x2,y2 = self.canvas.coords(h)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None
    
    def _press(self, e):
        self.start_mouse = (e.x, e.y)
        self.start_rect = tuple(self.canvas.coords(self.rect))
        handle = self._handle_hit(e.x, e.y)
        if handle is not None:
            self.mode = "resize"
            self.active_corner = handle
        elif self._inside_rect(e.x, e.y):
            self.mode = "move"
        else:
            self.mode = None

    def _drag(self, e):
        if not self.mode:
            return
        if self.mode == "move":
            self._perform_move(e)
        else:
            self._perform_resize(e)

    def _perform_move(self, e):
        # start_rect are canvas coords
        x1_c, y1_c, x2_c, y2_c = self.start_rect
        width_c = x2_c - x1_c
        height_c = y2_c - y1_c

        # desired new top-left in canvas coords
        nx1_c = x1_c + (e.x - self.start_mouse[0])
        ny1_c = y1_c + (e.y - self.start_mouse[1])

        # Convert to original-image pixel coords (snap)
        nx1_px, ny1_px = self.canvas_to_img_px(nx1_c, ny1_c)

        # clamp in original pixel space so rectangle stays fully inside
        rect_w_px = int(round(width_c / self.scale))
        rect_h_px = int(round(height_c / self.scale))
        # ensure rect dims are multiples of base unit (they should be by construction)
        # but if rounding produced off-by-one, adjust:
        # compute k based on current rect size in px (rounded)
        k_w = max(1, rect_w_px // self.base_w)
        k_h = max(1, rect_h_px // self.base_h)
        # choose k that keeps aspect exact (they should be equal, but pick min)
        k = min(k_w, k_h)
        rect_w_px = k * self.base_w
        rect_h_px = k * self.base_h

        nx1_px = max(0, min(nx1_px, self.ow - rect_w_px))
        ny1_px = max(0, min(ny1_px, self.oh - rect_h_px))

        # convert back to canvas coordinates and apply
        nx1_c, ny1_c = self.img_px_to_canvas(nx1_px, ny1_px)
        nx2_c, ny2_c = self.img_px_to_canvas(nx1_px + rect_w_px, ny1_px + rect_h_px)

        self.canvas.coords(self.rect, nx1_c, ny1_c, nx2_c, ny2_c)
        self._draw_handles()
        self._update_overlay()

    def _perform_resize(self, e):
        # Work in original pixel coordinates for snapping and exact aspect preservation.
        # start_rect is canvas coords; convert the fixed corner to original pixels.
        x1_c, y1_c, x2_c, y2_c = self.start_rect

        # find fixed corner original pixel coords depending on active_corner
        fixed_cx, fixed_cy = (x2_c, y2_c) if self.active_corner == 0 else ((x1_c, y2_c) if self.active_corner == 1 else ((x2_c, y1_c) if self.active_corner == 2 else (x1_c, y1_c)))
        # mouse position maps to candidate opposite coordinate
        mouse_px, mouse_py = self.canvas_to_img_px(e.x, e.y)
        fixed_px, fixed_py = self.canvas_to_img_px(fixed_cx, fixed_cy)
        # raw width in px (original)
        raw_width = abs(fixed_px - mouse_px)
        # compute k (units) from raw_width, round to nearest but not below 1
        k = max(self.MIN_UNITS, int(round(raw_width / self.base_w)))
        candidate_w_px, candidate_h_px = self.units_to_pixels(k)
        left_px = fixed_px - candidate_w_px if self.active_corner in [0,2] else fixed_px
        top_px = fixed_py - candidate_h_px if self.active_corner in [0,1] else fixed_py
        right_px = fixed_px if self.active_corner in [0,2] else fixed_px + candidate_w_px
        bottom_px = fixed_py if self.active_corner in [0,1] else fixed_py + candidate_h_px

        # Strict check: must be fully inside original image bounds and be >= 1 unit
        if left_px < 0 or top_px < 0 or right_px > self.ow or bottom_px > self.oh:
            # ignore this resize candidate
            return
        if (right_px - left_px) < self.base_w or (bottom_px - top_px) < self.base_h:
            return

        # Convert candidate back to canvas coords and apply (these will snap because candidate was computed in px)
        left_c, top_c = self.img_px_to_canvas(left_px, top_px)
        right_c, bottom_c = self.img_px_to_canvas(right_px, bottom_px)
        self.canvas.coords(self.rect, left_c, top_c, right_c, bottom_c)
        self._draw_handles()
        self._update_overlay()
        
    def _update_overlay(self):
        """
        Darken everything outside the crop rectangle using 4 stippled rectangles.
        """
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        w, h = self.w, self.h

        self.canvas.coords(self.overlay_parts[0], 0, 0, w, y1)
        self.canvas.coords(self.overlay_parts[1], 0, y2, w, h)
        self.canvas.coords(self.overlay_parts[2], 0, y1, x1, y2)
        self.canvas.coords(self.overlay_parts[3], x2, y1, w, y2)

        # ensure overlays sit above image but below handles/rect border
        for r in self.overlay_parts:
            self.canvas.tag_raise(r, self.rect)

    def _release(self, e):
        self.mode = None
        self.active_corner = None

    def _on_accept(self):
        # Convert canvas coords to original image integer pixel coords and ensure integer bounds
        x1_c, y1_c, x2_c, y2_c = self.canvas.coords(self.rect)
        left_px, top_px = self.canvas_to_img_px(x1_c, y1_c)
        right_px, bottom_px = self.canvas_to_img_px(x2_c, y2_c)
        # Ensure left<right, top<bottom (and make right/bottom exclusive if you want)
        if right_px <= left_px:
            right_px = left_px + self.base_w
        if bottom_px <= top_px:
            bottom_px = top_px + self.base_h
        # final clamp
        left_px = max(0, min(self.ow - self.base_w, left_px))
        top_px = max(0, min(self.oh - self.base_h, top_px))
        right_px = max(left_px + self.base_w, min(self.ow, right_px))
        bottom_px = max(top_px + self.base_h, min(self.oh, bottom_px))

        self.crop_box_original = (int(left_px), int(top_px), int(right_px), int(bottom_px))
        self._accepted = True
        self.root.destroy()

    def _on_cancel(self):
        self._accepted = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self._accepted, self.crop_box_original