Forearm X-ray Analysis Deep Learning Project
1. Project Description
This project aims to develop a deep learning model capable of analyzing forearm X-ray images to identify and classify various conditions (e.g., normal vs. abnormal). It leverages state-of-the-art image processing and deep learning techniques, specifically employing a pre-trained Convolutional Neural Network (CNN) architecture (ResNet50) with transfer learning. The project includes modules for robust dataset organization, configurable model training, and efficient inference, designed to run on GPU-accelerated environments for optimal performance.

2. Core Features
Dataset Organization: A Python script to preprocess and organize the raw MURA (Musculoskeletal Radiographs) dataset into a structured format suitable for deep learning (train/valid/test splits with 'normal'/'abnormal' classes).

Model Configuration Frontend: A static HTML page (deployable on GitHub Pages) allowing users to easily configure model hyperparameters (e.g., image size, batch size, epochs, learning rate, classes, data augmentation parameters) and download a configuration file.

Deep Learning Model Training: A Python script utilizing TensorFlow/Keras and ResNet50 for transfer learning. It reads configuration from the generated JSON file, performs data augmentation, trains the model, and saves the best-performing weights. It also includes robust GPU detection and utilization.

Model Evaluation: Comprehensive evaluation of the trained model on a test set, including accuracy, classification reports, confusion matrices, and ROC curves (for binary classification).

Model Inference: A Python script for making predictions on new, unseen forearm X-ray images using the trained model, with robust path finding for the dataset.

3. Dataset
The primary dataset recommended and used for this project is the MURA (Musculoskeletal Radiographs) Dataset from Stanford University.

Description: A large collection of musculoskeletal X-ray images, including forearms, primarily labeled as 'normal' or 'abnormal'.

Acquisition: The dataset is typically accessed via Shared Access Signature (SAS) URLs provided by Kaggle or Stanford's official website. Tools like AzCopy are required for efficient download of the entire dataset from Azure Blob Storage.

Structure Requirement: The project scripts expect the dataset to be organized into the following structure:

your_processed_mura_forearm_dataset/
├── train/
│   ├── normal/
│   └── abnormal/
├── valid/
│   ├── normal/
│   └── abnormal/
└── test/
    ├── normal/
    └── abnormal/

4. Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+: Recommended Python version.

Virtual Environment (recommended): venv or conda for dependency management.

Required Python Libraries:

tensorflow (with GPU support if available)

numpy

matplotlib

scikit-learn

seaborn

tqdm (for progress bars during data organization)

opencv-python (for image processing)

AzCopy (for MURA dataset download): A command-line utility by Microsoft for high-performance data transfer with Azure Storage.

5. Setup and Installation
Clone the Repository:

git clone https://github.com/YourUsername/YourProjectName.git
cd YourProjectName

(Replace YourUsername and YourProjectName with your actual GitHub details)

Create and Activate a Virtual Environment:

python -m venv env
# On Windows:
.\env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

Install Python Dependencies:

pip install tensorflow numpy matplotlib scikit-learn seaborn tqdm opencv-python

If you plan to use a GPU, ensure you have the correct CUDA Toolkit and cuDNN libraries installed compatible with your TensorFlow version before installing tensorflow.

Install AzCopy:

Download AzCopy from the official Microsoft documentation (search "azcopy download").

Extract the downloaded .zip file (e.g., to C:\AzCopy on Windows or /usr/local/azcopy on Linux/macOS).

Add AzCopy to your system's PATH environment variable to run it from any directory. Instructions vary by OS (Windows Environment Variables, ~/.bashrc or ~/.zshrc for Linux/macOS). Close and reopen your terminal after updating PATH.

6. Dataset Preparation
This step involves downloading the raw MURA dataset and organizing it into the required structure.

Obtain a Fresh SAS URL:

Go to the MURA dataset page on Kaggle or the Stanford ML Group website.

Look for a "Download" or "Data Access" section and generate a new, time-limited Shared Access Signature (SAS) URL. This URL is crucial and expires, so always get a fresh one if you encounter authentication errors.

Copy the entire URL provided.

Download the MURA Dataset using AzCopy:

Open your terminal/command prompt.

Navigate to a directory where you want to store the raw MURA dataset (e.g., outside your project folder to keep it clean).

Execute the azcopy command with your fresh SAS URL:

azcopy copy "YOUR_NEW_FRESH_SAS_URL_HERE" . --recursive

(Replace the placeholder with the full URL you copied. The . means current directory, --recursive downloads all subfolders).

This download may take a significant amount of time due to the dataset size.

Organize the Downloaded Data:

Once the download is complete, locate the downloaded MURA-v1.1 folder.

Open the mura_data_organizer.py script.

Edit MURA_RAW_BASE_DIR to point to the path containing MURA-v1.1.

Edit MURA_PROCESSED_BASE_DIR to specify where you want the new, organized dataset to be created (e.g., .\your_processed_mura_forearm_dataset).

Run the script:

python mura_data_organizer.py

This script will create the train/valid/test and normal/abnormal folder structure and copy/move the forearm X-ray images into place. It will print its progress.

7. Model Configuration (Frontend)
The frontend provides an interactive way to define model parameters.

Access the Frontend:

Open the forearm_xray_frontend.html file in your web browser (or deploy it to GitHub Pages for easy access).

Configure Parameters:

Adjust values for Image Size, Batch Size, Number of Epochs, Learning Rate, and Classes (e.g., normal,abnormal).

Modify Augmentation Parameters as desired.

Generate Configuration:

Click the "Generate Configuration File" button.

This will display the generated JSON in a textarea and automatically prompt you to download a model_config.json file.

Save this model_config.json file in the same directory as your forearm_xray_analysis_model.py and forearm_inference.py scripts.

8. Model Training
This step trains the deep learning model using your prepared dataset and chosen configuration.

Update forearm_xray_analysis_model.py:

Ensure the forearm_xray_analysis_model.py script has the robust path-finding logic (as provided in the latest version of the code in the Canvas). This script will automatically try to find your your_processed_mura_forearm_dataset or prompt you for its path if needed.

Run the Training Script:

Open your terminal/command prompt (with your virtual environment activated).

Navigate to the directory containing forearm_xray_analysis_model.py and model_config.json.

Execute the script:

python forearm_xray_analysis_model.py

GPU Usage: The script will automatically detect and utilize available GPUs. Ensure your TensorFlow installation is configured for GPU if you have one.

Monitoring: The console will display training progress (loss, accuracy for training and validation sets per epoch). A best_forearm_xray_model.keras file will be saved when validation accuracy improves.

Evaluation: After training, it will evaluate the best model on the test set and display detailed metrics (classification report, confusion matrix, ROC curve plots).

9. Model Inference
Once trained, you can use the forearm_inference.py script to classify new X-ray images.

Ensure model_config.json and best_forearm_xray_model.keras are present in the same directory as forearm_inference.py.

Run the Inference Script:

Open your terminal/command prompt (with your virtual environment activated).

Navigate to the directory containing forearm_inference.py.

Execute the script:

python forearm_inference.py

The script will attempt to find your processed test dataset to pick an example image. If not found, it will prompt you for the path or create a dummy image.

It will print the predicted condition and probabilities for the sample image.

If Matplotlib's interactive backend (like TkAgg) is successfully configured, it will display the image with its prediction.

10. Project Structure
YourProjectName/
├── forearm_xray_frontend.html  # Frontend for model configuration
├── mura_data_organizer.py      # Script to organize raw MURA data
├── forearm_xray_analysis_model.py # Script for model training and evaluation
├── forearm_inference.py        # Script for single image inference
├── model_config.json           # Generated configuration file (from frontend)
├── best_forearm_xray_model.keras # Saved best trained model weights
└── env/                        # Python Virtual Environment
└── your_processed_mura_forearm_dataset/ # Organized dataset (created by organizer script)
    ├── train/
    ├── valid/
    └── test/

11. Future Work / Improvements
DICOM Support: Implement direct handling of DICOM images in preprocessing without requiring prior conversion to PNG/JPG.

Specific Condition Classification: Further refine the model to classify specific types of fractures or conditions beyond just 'normal'/'abnormal' if a more granularly labeled dataset becomes available.

Web API for Inference: Develop a simple Flask/FastAPI backend to expose the model as a web API for remote inference.

Interactive Frontend for Inference: Extend the frontend to allow users to upload an X-ray image and get an immediate prediction.

Advanced Data Augmentation: Explore more sophisticated medical image augmentation techniques.

12. License
This project is licensed under the [MIT License / Apache 2.0 License / Your Chosen License] - see the LICENSE file for details.

13. Contact
For any questions or collaborations, please contact [Your Name] at [Your Email Address].
