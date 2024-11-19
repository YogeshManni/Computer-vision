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

# Variable to store the previous position of the index finger
prev_x, prev_y = None, None

drawing_color = (0, 0, 255)  # Color of the drawing (red)

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
            
            # Get the coordinates of the index finger tip
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]

            # Convert normalized coordinates to pixel coordinates
            ix, iy = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)

            # Draw a circle at the index finger tip
            cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)

            # Draw on the screen if a previous position exists
            if prev_x is not None and prev_y is not None:
                cv2.line(frame, (prev_x, prev_y), (ix, iy), drawing_color, 5)

            # Update the previous position
            prev_x, prev_y = ix, iy
    else:
        # Reset previous position if no hand is detected
        prev_x, prev_y = None, None

    # Display the frame
    cv2.imshow('Gesture Drawing App', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
