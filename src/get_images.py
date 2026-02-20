import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from cropper import Cropper
from utils import get_aspect_ratio

def load_target(path="target.png"):
    """
    Loads the target image from the given path.
    Returns a dictionary with width, height, image, and RGB pixel data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"target image not found at '{path}'")
    img = Image.open(path).convert("RGBA")
    w, h = img.size

    rgb = img.convert("RGB")
    pixels = np.array(rgb).reshape(-1, 3) # list of (R,G,B) tuples

    return {"width": w, "height": h, "image": img, "pixels": pixels}

def load_weights(path="weights.png"):
    """
    Loads the weight map from the given path.
    Returns a dictionary with width, height, image, and weight data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"weight map not found at '{path}'")
    img = Image.open(path).convert("L")
    w, h = img.size

    weights = np.array(img).reshape(-1) # list of weights
    weights = weights.astype(np.float32) / 255.0 # normalize to [0,1]
    return {"width": w, "height": h, "image": img, "pixels": weights}

def pick_file_via_dialog():
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
    path = filedialog.askopenfilename(title="Select an image to upload", filetypes=filetypes)
    root.destroy()
    return path

def load_uploaded(upload_path, target_w, target_h, target_aspect):
    """
    Loads and crops/resizes the uploaded image to match target dimensions.
    Returns a dictionary with the resized image and its RGB pixel data.
    """
    src = Image.open(upload_path).convert("RGB")

    cropper = Cropper(src, target_w, target_h)
    accepted, crop_box = cropper.run()
    if not accepted:
        raise RuntimeError("User canceled cropping or closed the window.")
    left, top, right, bottom = crop_box
    cropped = src.crop((left, top, right, bottom))

    cw, ch = cropped.size

    if cw >= target_w and ch >= target_h:
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.Resampling.NEAREST

    img = cropped.resize((target_w, target_h), resample=resample)

    pixels = np.array(img).reshape(-1, 3) # list of (R,G,B) tuples
    return {"image": img, "pixels": pixels}

def read_all_files(use_weights=True):
    print("Loading target.png...")
    try:
        target = load_target("../data/target.png")
    except Exception as e:
        print("Error loading target.png:", e)
        return
    
    if use_weights:
        print("Loading weights.png...")
        try:
            weights = load_weights("../data/weights.png")
        except Exception as e:
            print("Error loading weights.png:", e)
            return
    else:
        weights = {"pixels": np.ones_like(target["pixels"][:,0])} # dummy weights

    target_w = target["width"]
    target_h = target["height"]
    target_pixels = target["pixels"]

    weight_pixels = weights["pixels"]
    if weight_pixels.shape[0] != target_pixels.shape[0]:
        print("Warning: weight map size does not match target image size. Ignoring weights.")
        weight_pixels = np.ones_like(target_pixels[:,0])

    print(f"Target image: {target_w} x {target_h}")
    print("Please select an image file to upload in the dialog.")
    upload_path = pick_file_via_dialog()
    if not upload_path:
        print("No file selected. Exiting.")
        return

    print("Processing uploaded image:", upload_path)
    target_aspect = get_aspect_ratio(target_w, target_h)
    try:
        result = load_uploaded(upload_path, target_w, target_h, target_aspect)
        resized_img = result["image"]
        result_pixels = result["pixels"]
    except Exception as e:
        print("Operation canceled or failed:", e)
        return
    
    out_path = "../res/resized_result.png"
    resized_img.save(out_path)
    print(f"Note: resized image saved to {out_path}")

    return target_pixels, weight_pixels, result_pixels