# When choosing image sizes, we advise the following:
# Make sure height and width are both multiples of 8.
# Going below 512 might result in lower quality images.
# Going over 512 in both directions will repeat image areas (global coherence is lost).
# The best way to create non-square images is to use 512 in one dimension, and a value larger than that in the other one.
# INFO: https://huggingface.co/blog/stable_diffusion
# INFO: https://huggingface.co/CompVis/stable-diffusion
# PARAMETERS: https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(1024)

num_images = 3
prompt = ["a steampunk ambiented machine doing artificial intelligence"] * num_images

images = pipe(prompt,height=512, width=768, guidance_scale=7.5, num_inference_steps=5, generator=generator).images

# Function to create a grid of images
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

grid = image_grid(images, rows=1, cols=3)
grid.save(f"astronaut_rides_horse.png")

# Tkinter GUI for displaying images and saving selected images
import tkinter as tk
from tkinter import simpledialog

class ImageWindow(tk.Tk):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.bind("<Key>", self.on_key)
        self.show_images(0)

    def show_images(self, index):
        self.current_index = index
        self.photo = tk.PhotoImage(file=self.images[index])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def on_key(self, event):
        key = event.char
        if key in '123456789' and int(key) <= len(self.images):
            selected_index = int(key) - 1
            selected_image = self.images[selected_index]
            file_name = simpledialog.askstring("Input", "Enter file name:")
            if file_name:
                selected_image.save(f"{file_name}.png")
        elif key == 'n' and self.current_index + 1 < len(self.images):
            self.show_images(self.current_index + 1)
        elif key == 'p' and self.current_index - 1 >= 0:
            self.show_images(self.current_index - 1)

def main(images):
    window = ImageWindow(images)
    window.mainloop()

# Convert the PIL Images to temporary files so they can be loaded by Tkinter's PhotoImage
import tempfile
temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix='.png') for _ in images]
for img, temp_file in zip(images, temp_files):
    img.save(temp_file)
temp_file_paths = [temp_file.name for temp_file in temp_files]

# Call the main function with the paths to the temporary image files
main(temp_file_paths)
