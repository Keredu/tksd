
markdown
Copy code
# TKSD: TKinter and Stable Diffusion

TKSD is a graphical user interface (GUI) application built with TKinter for generating images using the Stable Diffusion method.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Get Started

### Clone the repository
```bash
git clone https://github.com/your-username/tksd.git
cd tksd
```

### Create a virtual environment
```bash
python -m venv env
source env/bin/activate
```

### Install the required dependencies
```bash
pip install -r requirements.txt
sudo apt-get install python3-tk
```

### File Structure
- main.py : Main script to run the application.
- images/ : Directory where saved images are stored.
- tmp_images/ : Temporary directory for storing generated images before they are saved.

### Usage
Run the main script:
```
python main.py
```
A form window will pop up. Enter the desired parameters and click "Submit".

The application will generate images based on the provided prompt and display them in a new window. Navigate through images using the left and right arrow keys. Press 's' to save the currently displayed image, and provide a name when prompted.



## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

