import cv2

# Load the classifier for cat detection
classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# Capture video from webcam
video_capture = cv2.VideoCapture(1)

while True:
    # Read each frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using the classifier
    objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("image.jpg", frame)
        print ("image saved")

    # Show the video feed
    cv2.imshow('Animal Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the window
video_capture.release()
cv2.destroyAllWindows()
