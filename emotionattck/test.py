# Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import mediapipe as mp

# Load the pre-trained expression prediction model from the drive
def initialize_model():
    # Open and read the model file
    with open('E:/pythonproject/model/mp_model.p', 'rb') as f:
        model_data = pickle.load(f)
    # Return the loaded model
    return model_data['model']

# Set up the MediaPipe FaceMesh model with static_image_mode set to True
def initialize_mediapipe_facemesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# Predict the expression using an image and overlay landmarks on the image
def predict_expression(img_path, clf, facemesh, label_classes):
    # Read the image from the path
    img = cv2.imread(img_path)
    # Convert the image color space to RGB and process it with FaceMesh
    results = facemesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Check if any faces are detected in the image
    if not results.multi_face_landmarks:
        print(f"No face detected in {img_path}.")
        return

    # Create a copy of the image to overlay landmarks on it
    img_with_landmarks = img.copy()

    # List to store the extracted landmarks
    landmark_data = []
    for landmarks in results.multi_face_landmarks:
        for landmark in landmarks.landmark:
            # Calculate the landmark's x and y coordinates on the image
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            # Draw the landmark as a green circle on the image copy
            cv2.circle(img_with_landmarks, (x, y), 1, (0, 255, 0), -1)
            # Append the x, y, and z values of the landmark to the list
            landmark_data.extend([landmark.x, landmark.y, landmark.z])

    # Use the trained model to predict the expression using the landmarks
    prediction_index = clf.predict([landmark_data])[0]
    # Get the predicted label string using the label_classes list
    prediction_label = label_classes[prediction_index]
    # Calculate the confidence of the prediction
    confidence = np.max(clf.predict_proba([landmark_data]))
    # Map the prediction label to its corresponding expression name
    prediction_expression = prediction_label.capitalize()

    # Display the image with landmarks and prediction details
    display_result(img_with_landmarks, prediction_expression, confidence)

# Display the image with landmarks and expression prediction using matplotlib
def display_result(img_with_landmarks, prediction_expression, confidence):
    # Calculate dpi for a desired width of 500 pixels
    dpi = 550 / 6
    # Create a new figure with calculated dpi
    plt.figure(dpi=dpi)
    plt.imshow(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
    # Set the title to show the predicted expression and confidence
    plt.title(f"{prediction_expression} ({confidence*100:.2f}%)")
    # Hide the axis values
    plt.axis('off')
    # Show the image
    plt.show()

# Initialize the expression prediction model and the MediaPipe FaceMesh model
clf = initialize_model()
facemesh = initialize_mediapipe_facemesh()
label_classes = ['happy', 'sad', 'surprise']

# Test the model using an image from the drive
predict_expression('E:/pythonproject/dataset/Face Expressions (Happy, Sad, Surprise)/test/happy/happy (4).jpg', clf, facemesh, label_classes)
predict_expression('E:/pythonproject/dataset/Face Expressions (Happy, Sad, Surprise)/test/sad/sad (7).jpg', clf, facemesh, label_classes)
predict_expression('E:/pythonproject/dataset/Face Expressions (Happy, Sad, Surprise)/test/surprise/surprise (3).jpg', clf, facemesh, label_classes)