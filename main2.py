# Import necessary libraries
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox
import tempfile

# Function to create a grid of images
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Tkinter GUI for displaying images and saving selected images
class ImageWindow(tk.Toplevel):  # Changed from tk.Tk to tk.Toplevel
    def __init__(self, master, images=None):  # Added master parameter
        super().__init__(master)
        self.images = images or []
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.bind("<Key>", self.on_key)
        self.photo = None  # Keep a reference to the photo object
        if self.images:
            self.show_images(0)

    def show_images(self, index):
        self.current_index = index
        pil_image = Image.open(self.images[index])  # Open the image file with Pillow
        self.photo = ImageTk.PhotoImage(pil_image)  # Convert to Tkinter-compatible image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def on_key(self, event):
        if event.keysym == 'Right' and self.current_index + 1 < len(self.images):
            self.show_images(self.current_index + 1)
        elif event.keysym == 'Left' and self.current_index - 1 >= 0:
            self.show_images(self.current_index - 1)
        elif event.char == 's':
            selected_image = self.images[self.current_index]
            file_name = simpledialog.askstring("Input", "Enter file name:")
            if file_name:
                Image.open(selected_image).save(f"{file_name}.png")


class ParameterForm(tk.Toplevel):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.prompt_label = tk.Label(self, text="Prompt:")
        self.prompt_entry = tk.Entry(self)
        self.prompt_entry.insert(0, "a steampunk ambiented machine doing artificial intelligence")

        self.height_label = tk.Label(self, text="Height:")
        self.height_entry = tk.Entry(self)
        self.height_entry.insert(0, "512")

        self.width_label = tk.Label(self, text="Width:")
        self.width_entry = tk.Entry(self)
        self.width_entry.insert(0, "768")

        self.guidance_scale_label = tk.Label(self, text="Guidance Scale:")
        self.guidance_scale_entry = tk.Entry(self)
        self.guidance_scale_entry.insert(0, "7.5")

        self.num_inference_steps_label = tk.Label(self, text="Number of Inference Steps:")
        self.num_inference_steps_entry = tk.Entry(self)
        self.num_inference_steps_entry.insert(0, "50")

        self.submit_button = tk.Button(self, text="Submit", command=self.submit)

        self.prompt_label.pack()
        self.prompt_entry.pack()
        self.height_label.pack()
        self.height_entry.pack()
        self.width_label.pack()
        self.width_entry.pack()
        self.guidance_scale_label.pack()
        self.guidance_scale_entry.pack()
        self.num_inference_steps_label.pack()
        self.num_inference_steps_entry.pack()
        self.submit_button.pack()

    def submit(self):
        self.withdraw()  # Hide this window
        messagebox.showinfo("Processing", "Running the pipeline...")  # Show a message
        params = {
            "prompt": self.prompt_entry.get(),
            "height": int(self.height_entry.get()),
            "width": int(self.width_entry.get()),
            "guidance_scale": float(self.guidance_scale_entry.get()),
            "num_inference_steps": int(self.num_inference_steps_entry.get())
        }
        self.callback(params)
        self.destroy()  # Close this window

window = None  # Declare window as a global variable outside of the main function

def main():
    global window  # Access the global window variable inside the main function
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    window = ImageWindow(root)  # Create the ImageWindow instance here, outside the generate_images function

    def generate_images(params):
        global window
        # Image generation part
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.to("cuda")

        generator = torch.Generator("cuda").manual_seed(1024)

        images = pipe(
            [params['prompt']] * 3,
            height=params['height'],
            width=params['width'],
            guidance_scale=params['guidance_scale'],
            num_inference_steps=params['num_inference_steps'],
            generator=generator
        ).images

        grid = image_grid(images, rows=1, cols=3)
        grid.save(f"output_grid.png")

        # Convert the PIL Images to temporary files so they can be loaded by Tkinter's PhotoImage
        temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix='.png') for _ in images]
        for img, temp_file in zip(images, temp_files):
            img.save(temp_file)
        temp_file_paths = [temp_file.name for temp_file in temp_files]

        # Update the images of the existing ImageWindow instance
        window.images = temp_file_paths
        window.show_images(0)

    form = ParameterForm(generate_images)
    form.mainloop()

# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    main()
