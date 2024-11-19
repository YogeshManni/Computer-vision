import cv2
import mediapipe as mp

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initial size of the square
square_size = 100

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the coordinates of the thumb tip and index finger tip
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Convert normalized coordinates to pixel coordinates
            ix, iy = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
            tx, ty = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)

            # Draw circles on the detected landmarks
            cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)
            cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)

            # Calculate the distance between the thumb and index finger tips
            distance = ((ix - tx)**2 + (iy - ty)**2) ** 0.5

            # Adjust the size of the square based on the distance
            if distance < 50:
                square_size = max(50, square_size - 5)  # Zoom out
            elif distance > 100:
                square_size = min(frame_height - 10, square_size + 5)  # Zoom in

    # Draw the square in the center of the frame
    top_left = (frame_width // 2 - square_size // 2, frame_height // 2 - square_size // 2)
    bottom_right = (frame_width // 2 + square_size // 2, frame_height // 2 + square_size // 2)
    
    # to fill rectangle - change thickness to -1
    
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)



    # Display the frame
    cv2.imshow('Gesture Zoom App', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
