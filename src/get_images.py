import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from cropper import Cropper
from utils import get_aspect_ratio

def load_static(path, color_mode):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")
    img = Image.open(path).convert(color_mode)
    w, h = img.size
    pixels = np.array(img).reshape(-1, 3) if color_mode == "RGB" else np.array(img).reshape(-1)

    return {"width": w, "height": h, "image": img, "pixels": pixels}

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
    path = filedialog.askopenfilename(title="Select an image to upload", filetypes=filetypes, initialdir="../img")
    root.destroy()
    return path

def load_user_image(w, h, aspect):
    path = open_file_dialog()
    if not path:
        raise ValueError("No image selected.")
    raw_img = Image.open(path).convert("RGB")
    
    cropper = Cropper(raw_img, w, h)
    accepted, crop_box = cropper.run()
    if not accepted:
        raise RuntimeError("User canceled cropping or closed the window.")
    left, top, right, bottom = crop_box
    img = raw_img.crop((left, top, right, bottom))

    cw, ch = img.size

    res_algo = Image.Resampling.LANCZOS if (cw >= w and ch >= h) else Image.Resampling.NEAREST
    img = img.resize((w, h), res_algo)
    pixels = np.array(img).reshape(-1, 3)

    return {"width": w, "height": h, "image": img, "pixels": pixels}

def read_all_files(use_weights=False):
    # Load /data/target.png
    print("Loading /data/target.png...")
    try:
        target_data = load_static("../data/target.png", "RGB")
    except Exception as e:
        raise RuntimeError(f"Error loading /data/target.png: {e}")
    w, h = target_data["width"], target_data["height"]
    aspect = get_aspect_ratio(w, h)
    print("Target image loaded with dimensions", w, "x", h)

    # Load /data/weights.png if weights enabled
    if use_weights:
        print("Loading /data/weights.png...")
        try:
            weight_data = load_static("../data/weights.png", "L")
        except Exception as e:
            raise RuntimeError(f"Error loading /data/weights.png: {e}")
        if weight_data["width"] != w or weight_data["height"] != h:
            raise ValueError("weights.png dimensions do not match target.png")
    else:
        weight_data = {"width": w, "height": h, "image": Image.new("L", (w, h), color=255), "pixels": 255 * np.ones(w * h, dtype=np.float32)}

    if weight_data['width'] != w or weight_data['height'] != h:
        raise ValueError("weights.png dimensions do not match target.png")
    weight_data["pixels"] /= 255.0 # normalize weights to [0,1]
    
    # Load user provided image (initial image)
    try:
        user_data = load_user_image(w, h, aspect)
    except Exception as e:
        raise RuntimeError(f"Error loading user image: {e}")
    
    out_path = "../res/resized_result.png"
    user_data["image"].save(out_path)
    print(f"Resized user image saved to {out_path}")

    return target_data["pixels"], weight_data["pixels"], user_data["pixels"]