import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def hand_detect(image,hands):
    hand_results = hands.process(image)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                             circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                               circle_radius=2))
            # Coordinates for the index finger landmarks (base to tip: 5, 6, 7, 8).
            index_finger_coordinates = [(hand_landmarks.landmark[point].x, hand_landmarks.landmark[point].y) for point in [5, 6, 7, 8]]
            for i in range(len(index_finger_coordinates) - 1):
                pt1 = (int(index_finger_coordinates[i][0] * image.shape[1]), int(index_finger_coordinates[i][1] * image.shape[0]))
                pt2 = (int(index_finger_coordinates[i+1][0] * image.shape[1]), int(index_finger_coordinates[i+1][1] * image.shape[0]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)

    return image,hand_results