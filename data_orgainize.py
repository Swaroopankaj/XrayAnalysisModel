import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bars, install with pip install tqdm

# --- Configuration ---
# Path to your downloaded and unzipped MURA dataset (e.g., where 'MURA-v1.1' is located)
MURA_RAW_BASE_DIR = 'C:/Users/PS/Downloads/azcopy_windows_amd64_10.29.1/azcopy_windows_amd64_10.29.1/muramskxrays/MURA-v1.1' # <--- CHANGE THIS PATH

# Path where the organized dataset will be created
MURA_PROCESSED_BASE_DIR = 'D:/Git_Workspace/Python/Hyperspectral Imaging/TrainingData' # <--- CHANGE THIS PATH

# Body part we are interested in
BODY_PART = 'XR_FOREARM'

# Test set size for the split from the original 'valid' set (e.g., 20% of original 'valid' for test)
TEST_SIZE_FROM_VALID = 0.5 # Using 50% of original 'valid' for new 'test', rest for new 'valid'

# --- Create Target Directories ---
def create_dirs(base_dir, splits=['train', 'valid', 'test'], classes=['normal', 'abnormal']):
    """Creates the necessary directory structure for the processed dataset."""
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

# --- Process and Copy Images ---
def process_mura_split(source_split_dir, target_base_dir, split_name):
    """
    Processes images for a given MURA split (e.g., 'train' or 'valid')
    and copies them to the target 'normal'/'abnormal' subfolders.
    Returns a list of (image_path, label) for further splitting if needed.
    """
    print(f"\nProcessing MURA raw '{split_name}' split for {BODY_PART}...")
    forearm_path = os.path.join(source_split_dir, BODY_PART)
    
    if not os.path.exists(forearm_path):
        print(f"Warning: {forearm_path} not found. Skipping {split_name} for {BODY_PART}.")
        return []

    image_paths_with_labels = []

    # Iterate through patients
    for patient_folder in tqdm(os.listdir(forearm_path), desc=f"Collecting {BODY_PART} images from {split_name}"):
        patient_path = os.path.join(forearm_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        # Iterate through studies (positive or negative)
        for study_folder in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_folder)
            if not os.path.isdir(study_path):
                continue

            # Determine label based on study folder name
            label = 'abnormal' if '_POSITIVE' in study_folder.upper() else 'normal'

            # Iterate through images in the study
            for image_file in os.listdir(study_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')): # Include .dcm if you plan to convert
                    src_image_path = os.path.join(study_path, image_file)
                    
                    # For simplicity, we'll only copy PNG/JPG for now,
                    # assuming DICOM images might need external conversion or specific loading
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Construct relative path to maintain uniqueness in new structure
                        relative_path = os.path.relpath(src_image_path, source_split_dir)
                        
                        # Use new_image_filename logic from the previous iteration,
                        # this combines patient/study/image_file into a single filename for the flat structure
                        new_image_filename = relative_path.replace(os.sep, '__') 
                        
                        target_class_dir = os.path.join(target_base_dir, split_name, label)
                        
                        # --- FIX: Ensure the target directory exists before copying ---
                        os.makedirs(target_class_dir, exist_ok=True) # Create the directory if it doesn't exist
                        
                        dst_image_path = os.path.join(target_class_dir, new_image_filename)
                        
                        shutil.copy(src_image_path, dst_image_path)
                        image_paths_with_labels.append((dst_image_path, label))
                    # else: # If you have DICOMs and want to handle them
                        # print(f"Skipping DICOM/unsupported image: {src_image_path}. Convert DICOMs to PNG first if needed.")
    print(f"Finished collecting images for '{split_name}'. Total: {len(image_paths_with_labels)} images.")
    return image_paths_with_labels

# --- Main Script ---
if __name__ == "__main__":
    if not os.path.exists(MURA_RAW_BASE_DIR):
        print(f"Error: Raw MURA dataset base directory '{MURA_RAW_BASE_DIR}' not found.")
        print("Please download and extract the MURA dataset and update 'MURA_RAW_BASE_DIR'.")
        print("You also need to install scikit-learn: pip install scikit-learn")
        exit()

    # Create processed directories (this creates the top-level train/valid/test and normal/abnormal folders)
    create_dirs(MURA_PROCESSED_BASE_DIR)

    # 1. Process original MURA 'train' split
    raw_train_dir = os.path.join(MURA_RAW_BASE_DIR, 'train')
    _ = process_mura_split(raw_train_dir, MURA_PROCESSED_BASE_DIR, 'train')

    # 2. Process original MURA 'valid' split and split into new 'valid' and 'test'
    raw_valid_dir = os.path.join(MURA_RAW_BASE_DIR, 'valid')
    # For splitting, we copy to a temp_valid folder first. We must create this temp_valid folder's class subdirs.
    create_dirs(MURA_PROCESSED_BASE_DIR, splits=['temp_valid'], classes=['normal', 'abnormal']) # Create temp_valid subdirs
    all_valid_images = process_mura_split(raw_valid_dir, MURA_PROCESSED_BASE_DIR, 'temp_valid') 

    if all_valid_images:
        # Separate image paths and labels
        images = [item[0] for item in all_valid_images]
        labels = [item[1] for item in all_valid_images]

        # Perform stratified split to maintain class distribution
        val_images, test_images, val_labels, test_labels = train_test_split(
            images, labels, test_size=TEST_SIZE_FROM_VALID, stratify=labels, random_state=42
        )

        print(f"\nSplitting original 'valid' into new 'valid' and 'test':")
        print(f"New 'valid' set size: {len(val_images)}")
        print(f"New 'test' set size: {len(test_images)}")

        # Move images to their final destinations
        # First, move temp_valid images to the new 'valid' structure
        print("Moving images to final 'valid' and 'test' directories...")
        for img_path, label in tqdm(zip(val_images, val_labels), total=len(val_images), desc="Moving to new 'valid'"):
            src_filename = os.path.basename(img_path)
            # Ensure target directory exists for this specific image within the new 'valid' split
            final_valid_target_dir = os.path.join(MURA_PROCESSED_BASE_DIR, 'valid', label)
            os.makedirs(final_valid_target_dir, exist_ok=True) # Ensure dir exists
            target_path = os.path.join(final_valid_target_dir, src_filename)
            shutil.move(img_path, target_path)
        
        # Then, move test images to the 'test' structure
        for img_path, label in tqdm(zip(test_images, test_labels), total=len(test_images), desc="Moving to new 'test'"):
            src_filename = os.path.basename(img_path)
            # Ensure target directory exists for this specific image within the new 'test' split
            final_test_target_dir = os.path.join(MURA_PROCESSED_BASE_DIR, 'test', label)
            os.makedirs(final_test_target_dir, exist_ok=True) # Ensure dir exists
            target_path = os.path.join(final_test_target_dir, src_filename)
            shutil.move(img_path, target_path)
        
        # Clean up the temporary directory created during initial copy
        # This needs to be done carefully if other files might be there that are not images
        # The current implementation copies all images into temp_valid/normal or temp_valid/abnormal
        # So removing the entire 'temp_valid' sub-structure is appropriate after moving files.
        shutil.rmtree(os.path.join(MURA_PROCESSED_BASE_DIR, 'temp_valid'))
        print("Temporary 'temp_valid' directory removed.")
    else:
        print("No images found in the original 'valid' set for splitting.")

    print(f"\nDataset organization complete! Organized images are in: {MURA_PROCESSED_BASE_DIR}")
    print("You can now update 'DATASET_BASE_DIR' in your model training script to this path.")

