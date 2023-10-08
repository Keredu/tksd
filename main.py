# Import necessary libraries
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
import tempfile
import os

# Tkinter GUI for displaying images and saving selected images
class ImageWindow(tk.Toplevel):
    def __init__(self, master, output_dir, images=None, on_close=None):
        self.output_dir = output_dir  # Save the output_dir argument
        super().__init__(master)
        self.images = images or []
        self.on_close = on_close
        self.protocol("WM_DELETE_WINDOW", self.handle_close)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.bind("<Key>", self.on_key)
        self.photo = None  # Keep a reference to the photo object
        if self.images:
            self.show_images(0)

    def handle_close(self):
        if self.on_close:
            self.on_close()
        self.destroy()

    def show_images(self, index):
        if not self.winfo_exists():  # Check if window still exists
            self.recreate_window()
        self.current_index = index
        pil_image = Image.open(self.images[index])
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def recreate_window(self):
        self.__init__(self.master)  # Recreate the window

    def on_key(self, event):
        if event.keysym == 'Right' and self.current_index + 1 < len(self.images):
            self.show_images(self.current_index + 1)
        elif event.keysym == 'Left' and self.current_index - 1 >= 0:
            self.show_images(self.current_index - 1)
        elif event.char == 's':
            selected_image = self.images[self.current_index]
            file_name = simpledialog.askstring("Input", "Enter file name:")
            if file_name:
                Image.open(selected_image).save(f"{self.output_dir}/{file_name}.png")  # Use self.output_dir


class ParameterForm(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback
        self.protocol("WM_DELETE_WINDOW", self.handle_close)
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

        self.num_images_label = tk.Label(self, text="Number of Images:")
        self.num_images_entry = tk.Entry(self)
        self.num_images_entry.insert(0, "3")  # Set default value to 3

        self.submit_button = tk.Button(self, text="Submit", command=self.submit)

        self.status_label = tk.Label(self, text="")

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
        self.num_images_label.pack()
        self.num_images_entry.pack()
        self.submit_button.pack()
        self.status_label.pack()

    def update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()  # Ensure the label text is updated immediately
    
    def handle_close(self):
        self.master.quit()  # This will terminate the main event loop
        self.destroy()

    def enable_widgets(self):
        for widget in (self.prompt_entry, self.height_entry, self.width_entry,
                       self.guidance_scale_entry, self.num_inference_steps_entry,
                       self.submit_button):
            widget.config(state=tk.NORMAL)
        self.status_label.config(text="")

    def submit(self):
        for widget in (self.prompt_entry, self.height_entry, self.width_entry,
                       self.guidance_scale_entry, self.num_inference_steps_entry,
                       self.submit_button):
            widget.config(state=tk.DISABLED)

        params = {
            "prompt": self.prompt_entry.get(),
            "height": int(self.height_entry.get()),
            "width": int(self.width_entry.get()),
            "guidance_scale": float(self.guidance_scale_entry.get()),
            "num_inference_steps": int(self.num_inference_steps_entry.get()),
            "num_images": int(self.num_images_entry.get())
        }
        self.callback(params)


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.handle_close)  
        self.root.withdraw()  # Hide the root window
        self.window = None
        self.form = None
        self.output_dir = './images'  # Define the output directory for saved images
        self.temp_dir = './tmp_images'  # Define the directory for temporary images

    def handle_close(self):  # Add this method
        self.root.quit()
        self.root.destroy()

    def generate_images(self, params):
        # Create the output directory and temp directory if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)  # Ensure the temp directory exists
        self.form.update_status("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.to("cuda")

        generator = torch.Generator("cuda").manual_seed(1024)

        # Update the status message to indicate that the pipeline is running
        self.form.update_status("Running the pipeline...")
        images = pipe(
            [params['prompt']] * params['num_images'],
            height=params['height'],
            width=params['width'],
            guidance_scale=params['guidance_scale'],
            num_inference_steps=params['num_inference_steps'],
            generator=generator
        ).images
        
        # Update the status message to indicate that the pipeline has finished
        self.form.update_status("Pipeline finished. Displaying images.")

        # Update the following line to save the images in the temp directory
        temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=self.temp_dir) for _ in images]
        for img, temp_file in zip(images, temp_files):
            img.save(temp_file)
        temp_file_paths = [temp_file.name for temp_file in temp_files]

        if self.window is None or not self.window.winfo_exists():
            self.window = ImageWindow(self.root, self.output_dir, images=temp_file_paths, on_close=self.form.enable_widgets)
        self.window.images = temp_file_paths
        self.window.show_images(0)


    def run(self):
        self.form = ParameterForm(self.root, self.generate_images)
        self.form.mainloop()


if __name__ == "__main__":
    app = Application()
    app.run()