# Import necessary libraries
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
import tempfile


# Tkinter GUI for displaying images and saving selected images
class ImageWindow(tk.Toplevel):
    def __init__(self, master, images=None):
        super().__init__(master)
        self.images = images or []
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.bind("<Key>", self.on_key)
        self.photo = None  # Keep a reference to the photo object
        if self.images:
            self.show_images(0)

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
                Image.open(selected_image).save(f"{file_name}.png")


class ParameterForm(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
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
        self.submit_button.pack()
        self.status_label.pack()

    def submit(self):
        # Disable all input widgets and the submit button
        for widget in (self.prompt_entry, self.height_entry, self.width_entry,
                       self.guidance_scale_entry, self.num_inference_steps_entry,
                       self.submit_button):
            widget.config(state=tk.DISABLED)

        # Update the status label to indicate that the pipeline is running
        self.status_label.config(text="Running the pipeline...")
        self.update_idletasks()  # Ensure the label text is updated immediately

        params = {
            "prompt": self.prompt_entry.get(),
            "height": int(self.height_entry.get()),
            "width": int(self.width_entry.get()),
            "guidance_scale": float(self.guidance_scale_entry.get()),
            "num_inference_steps": int(self.num_inference_steps_entry.get())
        }
        self.callback(params)


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        self.window = None

    def generate_images(self, params):
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

        # Convert the PIL Images to temporary files so they can be loaded by Tkinter's PhotoImage
        temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix='.png') for _ in images]
        for img, temp_file in zip(images, temp_files):
            img.save(temp_file)
        temp_file_paths = [temp_file.name for temp_file in temp_files]

        # Check if the ImageWindow instance is still valid
        if self.window is None or not self.window.winfo_exists():
            self.window = ImageWindow(self.root)
        self.window.images = temp_file_paths
        self.window.show_images(0)

    def run(self):
        form = ParameterForm(self.root, self.generate_images)
        form.mainloop()


# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    app = Application()
    app.run()
