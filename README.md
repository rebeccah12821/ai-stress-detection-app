# AI Stress Detection App

## App Description

This Python application detects stress from facial images using deep learning. The core functionality leverages a pre-trained convolutional neural network (CNN) model to analyze facial expressions and classify them as "stressed" or "not stressed." The app processes static images provided by the user and outputs both the predicted class and the model's confidence score.

## Technologies Used

- **Python 3**
- **Keras** (for deep learning model loading and inference)
- **OpenCV** (for image processing)
- **NumPy** (for array manipulation)
- **TensorFlow** (backend for Keras)
- **Matplotlib** (for optional visualization)

## User Guide

### 1. Clone the Repository


### 2. Install Dependencies

It is recommended to use a virtual environment. Then install requirements:
pip install -r requirements.txt

### 3. Prepare the Model

Ensure the pre-trained model file (e.g., `model.h5`) is present in the project directory. If not, follow any instructions in the repo for obtaining or training the model.

### 4. Run the App

The main script (`main.py`) processes a facial image to detect stress. To use it:

python main.py --image path/to/your/image.jpg


- Replace `path/to/your/image.jpg` with the path to your own facial image file.

### 5. Output

The script will print the predicted stress status (e.g., "Stressed" or "Not Stressed") and the confidence score to the console.

### 6. Notes

- The app works best with clear, front-facing facial images.
- All processing is local; no images are uploaded or stored externally.
- For best results, use images with good lighting and minimal obstructions.

---

For questions or feedback, please open an issue in this repository.
