"""
GrainPalette Rice Classification Model Training Script
This script trains a CNN model using MobileNetV2 transfer learning
for rice type classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5
LEARNING_RATE = 0.0001

# Rice classes
RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def create_data_generators(data_dir):
    """Create data generators for training and validation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

def create_model():
    """Create CNN model with MobileNetV2 transfer learning"""
    
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def compile_model(model):
    """Compile the model"""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_rice_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def fine_tune_model(model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some layers"""
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards...")
    
    # Continue training
    fine_tune_epochs = 10
    total_epochs = EPOCHS + fine_tune_epochs
    
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=EPOCHS,
        validation_data=validation_generator,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    return history_fine

def evaluate_model(model, validation_generator):
    """Evaluate the model"""
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return loss, accuracy

def download_kaggle_dataset():
    """Download Rice Image Dataset from Kaggle"""
    print("📥 Downloading Rice Image Dataset from Kaggle...")
    
    try:
        # Check if kaggle is installed
        import kaggle
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'muratkokludataset/rice-image-dataset',
            path='.',
            unzip=True
        )
        
        # Rename the extracted folder to rice_dataset
        import shutil
        if os.path.exists('Rice_Image_Dataset'):
            if os.path.exists('rice_dataset'):
                shutil.rmtree('rice_dataset')
            shutil.move('Rice_Image_Dataset', 'rice_dataset')
            print("✅ Dataset downloaded and extracted successfully!")
            return True
            
    except ImportError:
        print("❌ Kaggle API not installed. Please install it using:")
        print("pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset")
        print("2. Download the dataset")
        print("3. Extract it and rename the folder to 'rice_dataset'")
        return False

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔑 Setting up Kaggle API...")
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API credentials found!")
    return True

def main():
    """Main training function"""
    print("🌾 GrainPalette Rice Classification Model Training")
    print("Using Kaggle Rice Image Dataset")
    print("=" * 50)
    
    # Setup Kaggle API
    if not setup_kaggle_api():
        print("Please set up Kaggle API credentials first.")
        return
    
    # Check if dataset exists, if not download it
    # data_dir = "rice_dataset"
    data_dir = r"E:\228X1A0106 ADI\4-1 IIC Project\rice_dataset\Rice_Image_Dataset"
    if not os.path.exists(data_dir):
        print(f"📂 Dataset directory '{data_dir}' not found.")
        if not download_kaggle_dataset():
            print("❌ Failed to download dataset. Please download manually.")
            return
    else:
        print(f"✅ Dataset directory '{data_dir}' found.")
    
    # Verify dataset structure
    expected_classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    missing_classes = []
    
    for class_name in expected_classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"❌ Missing rice classes: {missing_classes}")
        print("Please ensure the dataset has the correct structure.")
        return
    
    # Count images in each class
    print("\n📊 Dataset Statistics:")
    total_images = 0
    for class_name in expected_classes:
        class_path = os.path.join(data_dir, class_name)
        image_count = len([f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {class_name}: {image_count} images")
        total_images += image_count
    
    print(f"  Total: {total_images} images")
    
    if total_images == 0:
        print("❌ No images found in dataset!")
        return
    
    # Rest of the training code remains the same...
    # Create data generators
    print("\n📊 Creating data generators...")
    train_generator, validation_generator = create_data_generators(data_dir)
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    # Create and compile model
    print("\n🧠 Creating model...")
    model = create_model()
    model = compile_model(model)
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Display model summary
    model.summary()
    
    # Train the model
    print("\n🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(history)
    
    # Fine-tune the model (optional)
    print("\n🔧 Fine-tuning model...")
    history_fine = fine_tune_model(model, train_generator, validation_generator)
    
    # Evaluate final model
    evaluate_model(model, validation_generator)
    
    # Save the final model
    model.save('rice_model.h5')
    print("\n✅ Model saved as 'rice_model.h5'")
    
    # Save model in TensorFlow SavedModel format as well
    model.save('rice_model_savedmodel')
    print("✅ Model also saved in SavedModel format")
    
    print("\n🎉 Training completed successfully!")
    print("You can now use the trained model in the Flask application.")

if __name__ == "__main__":
    # Set memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()
