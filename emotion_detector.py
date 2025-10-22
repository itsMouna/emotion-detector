from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths to your dataset
train_dir = "fer2013/train"
test_dir = "fer2013/test"

# Create data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normalize pixel values
    rotation_range=20,       # random rotations for augmentation
    zoom_range=0.2,          # random zoom
    horizontal_flip=True     # flip images for diversity
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),    # resize all images to 48x48
    batch_size=32,
    color_mode='grayscale',  # emotion recognition usually works better in grayscale
    class_mode='categorical' # since we have multiple classes
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)


# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 output classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,             # you can start with 10 if you want faster testing
    batch_size=32
)

# Save the trained model for later use
model.save("emotion_detector_model.h5")
print("✅ Model trained and saved successfully!")
