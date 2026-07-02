import cv2
import mediapipe as mp

# Initializing the Model
mpHands = mp.solutions.hands
# MediaPipe utility to overlay landmarks on frames
mp_drawing = mp.solutions.drawing_utils
Hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

# Setting up camera
cap = cv2.VideoCapture(0)

# Setting up finger detection
def finger_detection(frame, hand, label):

    # Fingers Detection
    mp_drawing.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    # Count fingers
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_base = [2, 6, 10, 14, 18]  # Base joints for fingers
    count = 0

    # Identify left or right hand (assuming one hand for now)
    is_right_hand = (label) == "Right"

    # Adjust thumb detection based on hand side
    thumb_tip = hand.landmark[4]
    thumb_mcp = hand.landmark[2]
    thumb_ip = hand.landmark[3]  # Thumb intermediate joint

    # Count the thumb if it is extended, using different x-coordinate checks
    # for right and left hands because the thumb points in opposite directions.
    if is_right_hand:
        if thumb_tip.x < thumb_ip.x:  # Ensure correct order
            count += 1
    else:
        if thumb_tip.x > thumb_ip.x:
            count += 1

    # Check other four fingers using Y-coordinates (add margin for stability)
    for tip, base in zip(finger_tips[1:], finger_base[1:]):  # Skip thumb
        if hand.landmark[tip].y < hand.landmark[base].y:  # Add small threshold
           count += 1
    
    return count
                

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image(frame)
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = Hands.process(imgRGB)

    # A list to captures all fingers from both hands from the variable 'count'
    hand_counts = []    
    
    if results.multi_hand_landmarks and results.multi_handedness:

        for i, hand in enumerate (results.multi_hand_landmarks):

            label = results.multi_handedness[i].classification[0].label  # "Left" or "Right"

            # Counting fingers
            count = finger_detection(frame, hand, label)

            # Captures all fingers from both hands
            hand_counts.append(count)   

        if len(results.multi_handedness) == 2:

            # Total fingers across both hands
            total_fingers = sum(hand_counts)

            cv2.putText(frame, 
                        "Both Hands are present", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2)
            
            # Display the number of fingers
            cv2.putText(frame, f'Fingers: {total_fingers}', (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        else:   
                # We also need this to make sure that we detect each finger from each hand
                total_fingers = sum(hand_counts)

                if label == 'Left':
                    # Display 'Left Hand' on left side of window
                    cv2.putText(frame, label+' Hand', (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)
                # Display the number of fingers
                cv2.putText(frame, f'Fingers: {total_fingers}', (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

                if label == 'Right': 
                    # Display 'Left Hand' on left side of window
                    cv2.putText(frame, label+' Hand', (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

                    # Display the number of fingers
                    cv2.putText(frame, f'Fingers: {total_fingers}', (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Video Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
