import tensorflow as tf
import numpy as np
import cv2
import os
import json
import matplotlib

# Try to set an interactive backend for matplotlib.
# TkAgg is a good default as Tkinter is often included with Python.
try:
    matplotlib.use('TkAgg')
    print("Attempting to use TkAgg backend for matplotlib.")
except Exception as e:
    print(f"Could not set TkAgg backend: {e}. Falling back to default (non-interactive) backend.")
    print("If you want interactive plots, ensure Tkinter is installed or try 'pip install pyqt5' and change backend to 'Qt5Agg'.")
    # No explicit matplotlib.use() here, it will use the default (often 'Agg')

import matplotlib.pyplot as plt


# --- Configuration (must match your training setup) ---
# Load default parameters if config file is not found, or directly copy values if you know them.
# Ensure these match the values used during model training, especially IMAGE_SIZE and CLASSES.
CONFIG_FILE = 'model_config.json' # Path to your model configuration file

# Default parameters (should match what was used for training if config file is not found)
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CLASSES = ['normal', 'abnormal']

try:
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)
    print(f"Loaded configuration from {CONFIG_FILE}")

    model_params = config_data.get('model_parameters', {})
    IMAGE_SIZE = tuple(model_params.get('IMAGE_SIZE', DEFAULT_IMAGE_SIZE))
    CLASSES = model_params.get('CLASSES', DEFAULT_CLASSES)
    NUM_CLASSES = len(CLASSES)

except FileNotFoundError:
    print(f"Configuration file '{CONFIG_FILE}' not found. Using default parameters for inference.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from '{CONFIG_FILE}': {e}. Using default parameters for inference.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)
except Exception as e:
    print(f"An unexpected error occurred while loading config for inference: {e}. Using default parameters.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)


MODEL_PATH = 'best_forearm_xray_model.keras' # Path to your saved model

def preprocess_image(image_path, target_size):
    """
    Loads, resizes, and normalizes an image for model prediction.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load image (cv2.IMREAD_COLOR loads as BGR, cv2.IMREAD_GRAYSCALE loads grayscale)
    # The training used color_mode='rgb', so the grayscale X-rays were converted to 3 channels.
    # We should do the same here.
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}. Check file integrity.")

    # If image is grayscale, convert to 3 channels to match model input
    if len(img.shape) == 2: # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert to BGR (OpenCV default)
    
    # Ensure it's RGB for consistency with Matplotlib later if needed, and model expects RGB-like
    # ImageDataGenerator flow_from_directory with color_mode='rgb' handles this
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB if it was BGR

    # Resize image
    img = cv2.resize(img, target_size)

    # Normalize pixel values to 0-1
    img = img / 255.0

    # Add batch dimension (1, height, width, channels)
    img = np.expand_dims(img, axis=0) # This makes it (1, 224, 224, 3)

    return img

def predict_image_condition(model, image_path, target_size, classes):
    """
    Preprocesses an image, makes a prediction, and returns the predicted class
    and probabilities.
    """
    preprocessed_img = preprocess_image(image_path, target_size)

    # Make prediction
    predictions = model.predict(preprocessed_img)

    # Get predicted class (index of the highest probability)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_idx]

    # Get probabilities for all classes
    probabilities = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}

    return predicted_class, probabilities

# --- Main Execution ---
if __name__ == "__main__":
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and the model file exists.")
        exit()

    # --- Example Usage: Robustly find the dataset path ---
    # Define a list of potential paths for the processed dataset
    possible_dataset_paths = [
        'your_processed_mura_forearm_dataset', # Common folder name
        os.path.join(os.path.dirname(__file__), 'your_processed_mura_forearm_dataset'), # Relative to script
        os.path.join(os.path.expanduser("~"), 'Documents', 'your_processed_mura_forearm_dataset'), # Common Documents folder
        os.path.join(os.path.expanduser("~"), 'Downloads', 'your_processed_mura_forearm_dataset'), # Common Downloads folder
        os.path.join(os.path.dirname(__file__), '..', 'your_processed_mura_forearm_dataset'), # One level up
        # Add any other common locations where you might store your data
    ]
    
    YOUR_PROCESSED_DATASET_PATH = None

    def is_valid_dataset_path(path):
        """Checks if a given path looks like a valid processed dataset root."""
        print(f"--- Validating dataset path: {path} ---")
        
        # Check for train, valid, test directories and their class subdirectories
        splits = ['train', 'valid', 'test']
        for split in splits:
            for cls in CLASSES:
                check_path = os.path.join(path, split, cls)
                if not os.path.exists(check_path):
                    print(f"Validation FAILED: Directory '{check_path}' does not exist.")
                    return False
                else:
                    print(f"Validation OK: Directory '{check_path}' exists.")
                    
        # Check if there's at least one image in a representative test class folder
        # We choose the first class for this check, assuming other class folders will also be populated
        test_class_path = os.path.join(path, 'test', CLASSES[0])
        if not os.path.exists(test_class_path): # Redundant check but good for clarity
            print(f"Validation FAILED: Test class directory '{test_class_path}' does not exist (should have been caught earlier).")
            return False
            
        if len(os.listdir(test_class_path)) == 0:
            print(f"Validation FAILED: Test class directory '{test_class_path}' is empty. No images found.")
            return False
        else:
            print(f"Validation OK: Test class directory '{test_class_path}' contains images.")
            
        print(f"--- Dataset path '{path}' is VALID. ---")
        return True


    # 1. Try predefined paths
    for path in possible_dataset_paths:
        if is_valid_dataset_path(path):
            YOUR_PROCESSED_DATASET_PATH = path
            print(f"Dataset found at predefined path: {YOUR_PROCESSED_DATASET_PATH}")
            break
    
    # 2. If not found, check current working directory
    if YOUR_PROCESSED_DATASET_PATH is None:
        current_dir = os.getcwd()
        if is_valid_dataset_path(current_dir):
            YOUR_PROCESSED_DATASET_PATH = current_dir
            print(f"Dataset found in current directory: {YOUR_PROCESSED_DATASET_PATH}")
        else:
            # 3. If still not found, prompt user
            print("\nProcessed dataset not found in common locations.")
            while YOUR_PROCESSED_DATASET_PATH is None:
                user_input_path = input("Please enter the full path to your processed MURA forearm dataset (e.g., 'C:\\Users\\YourName\\my_dataset'): ")
                if is_valid_dataset_path(user_input_path):
                    YOUR_PROCESSED_DATASET_PATH = user_input_path
                    print(f"Dataset found at user-provided path: {YOUR_PROCESSED_DATASET_PATH}")
                else:
                    print("Invalid path or dataset structure not found. Please ensure the path is correct and contains 'train', 'valid', 'test' with 'normal' and 'abnormal' subfolders, and that these folders are not empty. Try again.")
    
    if YOUR_PROCESSED_DATASET_PATH is None:
        print("Failed to locate the processed dataset. Exiting.")
        exit()


    example_image_path = None

    # Try to find an example image from your test set (now that YOUR_PROCESSED_DATASET_PATH is confirmed)
    test_normal_dir = os.path.join(YOUR_PROCESSED_DATASET_PATH, 'test', 'normal')
    test_abnormal_dir = os.path.join(YOUR_PROCESSED_DATASET_PATH, 'test', 'abnormal')

    if os.path.exists(test_abnormal_dir) and len(os.listdir(test_abnormal_dir)) > 0:
        example_image_path = os.path.join(test_abnormal_dir, os.listdir(test_abnormal_dir)[0])
        print(f"\nUsing an example image from the 'abnormal' test set: {example_image_path}")
    elif os.path.exists(test_normal_dir) and len(os.listdir(test_normal_dir)) > 0:
        example_image_path = os.path.join(test_normal_dir, os.listdir(test_normal_dir)[0])
        print(f"\nUsing an example image from the 'normal' test set: {example_image_path}")
    else:
        print("\nCould not find example images in your processed test dataset.")
        print("Please ensure 'YOUR_PROCESSED_DATASET_PATH/test' exists and contains 'normal'/'abnormal' subfolders with images.")
        # As a last resort for testing the script structure, create a dummy image if no real image found
        dummy_image_filename = "dummy_xray_for_inference.png"
        dummy_image = np.random.randint(0, 256, size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
        cv2.imwrite(dummy_image_filename, dummy_image)
        example_image_path = dummy_image_filename
        print(f"Created a dummy image '{dummy_image_filename}' for demonstration.")


    if example_image_path:
        predicted_condition, probabilities = predict_image_condition(model, example_image_path, IMAGE_SIZE, CLASSES)

        print(f"\n--- Prediction Results for: {os.path.basename(example_image_path)} ---")
        print(f"Predicted Condition: {predicted_condition}")
        print("Probabilities:")
        for cls, prob in probabilities.items():
            print(f"  {cls}: {prob:.4f}")

        # Optional: Display the image with its prediction
        img_display = cv2.imread(example_image_path, cv2.IMREAD_GRAYSCALE)
        if img_display is not None:
            plt.imshow(img_display, cmap='gray')
            plt.title(f"Predicted: {predicted_condition} (Prob: {probabilities.get(predicted_condition, 0):.2f})")
            plt.axis('off')
            plt.show()
        else:
            print(f"Could not display image {example_image_path}.")
            
    # Clean up dummy image if it was created
    if 'dummy_image_filename' in locals() and os.path.exists(dummy_image_filename):
        os.remove(dummy_image_filename)
        print(f"Cleaned up dummy image '{dummy_image_filename}'.")
