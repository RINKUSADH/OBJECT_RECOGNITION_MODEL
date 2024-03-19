import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import datasets, layers, models

from sklearn.metrics import classification_report, confusion_matrix

# # Load CIFAR-10 dataset
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# # Normalize pixel values
# train_images, test_images = train_images / 255.0, test_images / 255.0

# # Define the CNN model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10)
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Train the model
# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# # Save the model
# model.save("object_recognition_model.h5")

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# # Make predictions
# predictions = model.predict(test_images)
# predicted_labels = np.argmax(predictions, axis=1)

# # Generate classification report and confusion matrix
# print(classification_report(test_labels, predicted_labels))
# cm = confusion_matrix(test_labels, predicted_labels)
# plt.figure(figsize=(8, 6))
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


# Streamlit interface
st.title("Object Recognition Model")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    model = tf.keras.models.load_model("object_recognition_model.h5")
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(32, 32))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0

    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions[0])

    st.write(f"IT IS : {class_names[predicted_class]}")
