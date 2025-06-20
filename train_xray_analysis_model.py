import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import json

# --- Configuration ---
# --- Load Configuration from JSON (Generated by Frontend) ---
CONFIG_FILE = 'model_config.json' # Expects this file in the same directory as the script

# Default parameters (used if CONFIG_FILE is not found or has errors)
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_CLASSES = ['normal', 'abnormal']

# Default augmentation parameters
DEFAULT_AUG_PARAMS = {
    'ROTATION_RANGE': 15,
    'WIDTH_SHIFT_RANGE': 0.1,
    'HEIGHT_SHIFT_RANGE': 0.1,
    'ZOOM_RANGE': 0.1,
    'HORIZONTAL_FLIP': True,
}

try:
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)
    print(f"Loaded configuration from {CONFIG_FILE}")

    # Extract model parameters
    model_params = config_data.get('model_parameters', {})
    IMAGE_SIZE = tuple(model_params.get('IMAGE_SIZE', DEFAULT_IMAGE_SIZE))
    BATCH_SIZE = model_params.get('BATCH_SIZE', DEFAULT_BATCH_SIZE)
    NUM_EPOCHS = model_params.get('NUM_EPOCHS', DEFAULT_NUM_EPOCHS)
    LEARNING_RATE = model_params.get('LEARNING_RATE', DEFAULT_LEARNING_RATE)
    CLASSES = model_params.get('CLASSES', DEFAULT_CLASSES)
    NUM_CLASSES = len(CLASSES)

    # Extract augmentation parameters
    aug_params = config_data.get('augmentation_parameters', {})
    ROTATION_RANGE = aug_params.get('ROTATION_RANGE', DEFAULT_AUG_PARAMS['ROTATION_RANGE'])
    WIDTH_SHIFT_RANGE = aug_params.get('WIDTH_SHIFT_RANGE', DEFAULT_AUG_PARAMS['WIDTH_SHIFT_RANGE'])
    HEIGHT_SHIFT_RANGE = aug_params.get('HEIGHT_SHIFT_RANGE', DEFAULT_AUG_PARAMS['HEIGHT_SHIFT_RANGE'])
    ZOOM_RANGE = aug_params.get('ZOOM_RANGE', DEFAULT_AUG_PARAMS['ZOOM_RANGE'])
    HORIZONTAL_FLIP = aug_params.get('HORIZONTAL_FLIP', DEFAULT_AUG_PARAMS['HORIZONTAL_FLIP'])

except FileNotFoundError:
    print(f"Configuration file '{CONFIG_FILE}' not found. Using default model and augmentation parameters.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    NUM_EPOCHS = DEFAULT_NUM_EPOCHS
    LEARNING_RATE = DEFAULT_LEARNING_RATE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)
    ROTATION_RANGE = DEFAULT_AUG_PARAMS['ROTATION_RANGE']
    WIDTH_SHIFT_RANGE = DEFAULT_AUG_PARAMS['WIDTH_SHIFT_RANGE']
    HEIGHT_SHIFT_RANGE = DEFAULT_AUG_PARAMS['HEIGHT_SHIFT_RANGE']
    ZOOM_RANGE = DEFAULT_AUG_PARAMS['ZOOM_RANGE']
    HORIZONTAL_FLIP = DEFAULT_AUG_PARAMS['HORIZONTAL_FLIP']
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from '{CONFIG_FILE}': {e}. Using default model and augmentation parameters.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    NUM_EPOCHS = DEFAULT_NUM_EPOCHS
    LEARNING_RATE = DEFAULT_LEARNING_RATE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)
    ROTATION_RANGE = DEFAULT_AUG_PARAMS['ROTATION_RANGE']
    WIDTH_SHIFT_RANGE = DEFAULT_AUG_PARAMS['WIDTH_SHIFT_RANGE']
    HEIGHT_SHIFT_RANGE = DEFAULT_AUG_PARAMS['HEIGHT_SHIFT_RANGE']
    ZOOM_RANGE = DEFAULT_AUG_PARAMS['ZOOM_RANGE']
    HORIZONTAL_FLIP = DEFAULT_AUG_PARAMS['HORIZONTAL_FLIP']
except Exception as e:
    print(f"An unexpected error occurred while loading config: {e}. Using default model and augmentation parameters.")
    IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    NUM_EPOCHS = DEFAULT_NUM_EPOCHS
    LEARNING_RATE = DEFAULT_LEARNING_RATE
    CLASSES = DEFAULT_CLASSES
    NUM_CLASSES = len(CLASSES)
    ROTATION_RANGE = DEFAULT_AUG_PARAMS['ROTATION_RANGE']
    WIDTH_SHIFT_RANGE = DEFAULT_AUG_PARAMS['WIDTH_SHIFT_RANGE']
    HEIGHT_SHIFT_RANGE = DEFAULT_AUG_PARAMS['HEIGHT_SHIFT_RANGE']
    ZOOM_RANGE = DEFAULT_AUG_PARAMS['ZOOM_RANGE']
    HORIZONTAL_FLIP = DEFAULT_AUG_PARAMS['HORIZONTAL_FLIP']


# --- Robust Dataset Path Finding ---
# Define a list of potential paths for the processed dataset
possible_dataset_paths = [
    'Dataset', # Common folder name
    os.path.join(os.path.dirname(__file__), 'Dataset'), # Relative to script
    os.path.join(os.path.expanduser("~"), 'Documents', 'Dataset'), # Common Documents folder
    os.path.join(os.path.expanduser("~"), 'Downloads', 'Dataset'), # Common Downloads folder
    os.path.join(os.path.dirname(__file__), '..', 'Dataset'), # One level up
    # Add any other common locations where you might store your data
]

DATASET_BASE_DIR = None

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
        DATASET_BASE_DIR = path
        print(f"Dataset found at predefined path: {DATASET_BASE_DIR}")
        break

# 2. If not found, check current working directory
if DATASET_BASE_DIR is None:
    current_dir = os.getcwd()
    if is_valid_dataset_path(current_dir):
        DATASET_BASE_DIR = current_dir
        print(f"Dataset found in current directory: {DATASET_BASE_DIR}")
    else:
        # 3. If still not found, prompt user
        print("\nProcessed dataset not found in common locations.")
        while DATASET_BASE_DIR is None:
            user_input_path = input("Please enter the full path to your processed MURA forearm dataset (e.g., 'C:\\Users\\YourName\\my_dataset'): ")
            if is_valid_dataset_path(user_input_path):
                DATASET_BASE_DIR = user_input_path
                print(f"Dataset found at user-provided path: {DATASET_BASE_DIR}")
            else:
                print("Invalid path or dataset structure not found. Please ensure the path is correct and contains 'train', 'valid', 'test' with 'normal' and 'abnormal' subfolders, and that these folders are not empty. Try again.")

if DATASET_BASE_DIR is None:
    print("Failed to locate the processed dataset. Exiting.")
    exit()

# Set up the specific train, validation, and test directories based on the found base path
TRAIN_DIR = os.path.join(DATASET_BASE_DIR, 'train')
VAL_DIR = os.path.join(DATASET_BASE_DIR, 'valid')
TEST_DIR = os.path.join(DATASET_BASE_DIR, 'test')


# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of GPUs available: {len(gpus)}. Using MirroredStrategy for distributed training.")
    except RuntimeError as e:
        print(f"MirroredStrategy could not be initialized: {e}")
        print("Falling back to default strategy (single GPU or CPU).")
        strategy = tf.distribute.get_strategy()
else:
    print("No GPUs found. Using default strategy (CPU).")
    strategy = tf.distribute.get_strategy()

# --- 1. Data Generators with Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    shear_range=0.1, # Shear is not configurable in frontend for simplicity, keep default or add
    zoom_range=ZOOM_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading training data from:", TRAIN_DIR)
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
        classes=CLASSES,
        color_mode='rgb'
    )
except Exception as e:
    print(f"Error loading training data: {e}")
    print(f"Please ensure '{TRAIN_DIR}' exists and contains '{CLASSES}' subfolders.")
    exit()

print("Loading validation data from:", VAL_DIR)
try:
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
        classes=CLASSES,
        color_mode='rgb',
        shuffle=False
    )
except Exception as e:
    print(f"Error loading validation data: {e}")
    print(f"Please ensure '{VAL_DIR}' exists and contains '{CLASSES}' subfolders.")
    exit()

print("Loading test data from:", TEST_DIR)
try:
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
        classes=CLASSES,
        color_mode='rgb',
        shuffle=False
    )
except Exception as e:
    print(f"Error loading test data: {e}")
    print(f"Please ensure '{TEST_DIR}' exists and contains '{CLASSES}' subfolders.")
    exit()

# --- 2. Model Definition (Transfer Learning with ResNet50) ---
with strategy.scope():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    
    # Fix for ValueError: arguments 'target' and 'output' must have the same rank
    # If binary classification, output layer should have 1 unit with sigmoid activation
    # If multi-class classification, output layer should have NUM_CLASSES units with softmax activation
    if NUM_CLASSES == 2:
        predictions = Dense(1, activation='sigmoid')(x) # Output layer for binary classification
    else:
        predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer for multi-class classification


    model = Model(inputs=base_model.input, outputs=predictions)

    # --- 3. Compile the Model ---
    # Loss function also depends on binary vs. categorical classification
    if NUM_CLASSES == 2:
        loss_function = 'binary_crossentropy'
    else:
        loss_function = 'categorical_crossentropy'

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss=loss_function,
                  metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# --- 4. Callbacks for Training ---
checkpoint = ModelCheckpoint('best_forearm_xray_model.keras',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              factor=0.5,
                              patience=5,
                              min_lr=0.00001,
                              verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               mode='max',
                               verbose=1,
                               restore_best_weights=True)

callbacks = [checkpoint, reduce_lr, early_stopping]

# --- 5. Train the Model ---
print("\nStarting model training with configured parameters...")
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)
print("Model training complete.")

# --- 6. Evaluate the Model on Test Set ---
print("\nEvaluating model on test set...")
try:
    model = tf.keras.models.load_model('best_forearm_xray_model.keras')
except Exception as e:
    print(f"Error loading the best model: {e}")
    print("Ensure 'best_forearm_xray_model.keras' was successfully saved during training.")
    exit()

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on test data
test_generator.reset()
predictions = model.predict(test_generator)

# Convert probabilities to predicted class labels based on classification type
if NUM_CLASSES == 2:
    y_pred_classes = (predictions > 0.5).astype(int).flatten() # For sigmoid output, threshold at 0.5
else:
    y_pred_classes = np.argmax(predictions, axis=1) # For softmax output

# Get true class labels from the generator.
y_true_classes = test_generator.classes

# --- 7. Performance Metrics and Visualization ---
print("\n--- Classification Report ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve and AUC
if NUM_CLASSES == 2:
    # For binary classification, y_pred_proba is the probability of the positive class
    y_pred_proba = predictions.flatten() # Sigmoid output is already the probability of class 1
    fpr, tpr, thresholds = roc_curve(y_true_classes, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("\nROC Curve and AUC are typically for binary classification. For multi-class, consider macro/micro AUC.")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
