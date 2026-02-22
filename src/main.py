import math
import numpy as np
from PIL import Image
from get_images import read_all_files
from utils import get_coords
import subprocess

def render_animation(assignment, init_pixels, width, num_steps=50):
    frames = []
    UPSCALE = 4
    for i in range(10):
        frames.append(init_pixels.reshape((width, width, 3)).repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1))
        
    final_frame = np.zeros((width * UPSCALE, width * UPSCALE, 3), dtype=np.uint8)
    for step in range(1, num_steps + 1):
        intermediate_image = np.zeros((width * UPSCALE, width * UPSCALE, 3), dtype=np.uint8)
        for i in range(init_pixels.shape[0]):
            target_idx = assignment[i]
            target_x, target_y = get_coords(target_idx, width)
            result_x, result_y = get_coords(i, width)

            interp_x = result_x + (target_x - result_x) * step / num_steps
            interp_y = result_y + (target_y - result_y) * step / num_steps

            x_start = int(interp_x * UPSCALE)
            y_start = int(interp_y * UPSCALE)
            intermediate_image[y_start:y_start+UPSCALE, x_start:x_start+UPSCALE] = init_pixels[i]

        intermediate_image = intermediate_image.reshape((width * UPSCALE, width * UPSCALE, 3))
        frames.append(intermediate_image)
        if (step == num_steps):
            final_frame = intermediate_image

    for i in range(10):
        frames.append(final_frame)
    return frames

def compute_hard_image(assignment, init_pixels, width):
    hard_image = np.zeros_like(init_pixels)
    for i in range(init_pixels.shape[0]):
        target_idx = assignment[i]
        hard_image[target_idx] = init_pixels[i]
    return hard_image.reshape((width, width, 3))

def send_to_cpp(target_pixels, weight_pixels, init_pixels):
    width = int(math.sqrt(target_pixels.shape[0]))
    
    with open("../data/transport_input.txt", "w") as f:
        f.write(f"{width}\n")
        for pixel in target_pixels:
            f.write(f"{pixel[0]} {pixel[1]} {pixel[2]}\n")
        for pixel in init_pixels:
            f.write(f"{pixel[0]} {pixel[1]} {pixel[2]}\n")
        for weight in weight_pixels:
            f.write(f"{weight}\n")
    
    subprocess.run(["../bin/transport"], check=True)

    assignment = []
    with open("../data/transport_output.txt", "r") as f:
        line = f.readline().strip()
        assignment = list(map(int, line.split()))
    return assignment

def main():
    use_weights = input("Use weights? (y/n): ").strip().lower() == 'y'
    target_pixels, weight_pixels, init_pixels = read_all_files(use_weights=use_weights)

    assignment = send_to_cpp(target_pixels, weight_pixels, init_pixels)
    final_pixels = compute_hard_image(assignment, init_pixels, int(math.sqrt(target_pixels.shape[0])))

    out_path = "../res/final_result.png"
    img = Image.fromarray(final_pixels.astype(np.uint8))
    img.save(out_path)
    print(f"Final transported image saved to {out_path}")
    
    frames = render_animation(assignment, init_pixels, int(math.sqrt(target_pixels.shape[0])))
    
    out_path_anim = "../res/animation.gif"
    frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    frames[0].save(out_path_anim, save_all=True, append_images=frames[1:], duration=len(frames), loop=0)
    print(f"Animation saved to {out_path_anim}")

if __name__ == "__main__":
    main()