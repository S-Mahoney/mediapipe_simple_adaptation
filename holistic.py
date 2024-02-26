import mediapipe as mp
import cv2
from serialize_landmarks import serialize_holistic_landmarks
from face_detect import face_detect

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

def holistic_detect(image,holistic,improved_face = None,use_pose = False, use_face = False, use_hands = False ):

    # Process the image and get the results
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if use_face:

        if improved_face:
            image, face_results = face_detect(image, improved_face)

        # Draw the face landmarks on the image.
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.face_landmarks,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    if use_pose:
        # Draw pose annotations on the image.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,  # Use the default connections
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if use_hands:
        # Drawing the hand landmarks for both hands
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.left_hand_landmarks,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.right_hand_landmarks,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

    return image, results

def main_holistic(print_json = False,use_pose = False, use_face = False, use_hands = False):

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = True
            image, results = holistic_detect(image, holistic,use_pose, use_face, use_hands )

            cv2.imshow('MediaPipe Holistic', image)

            if print_json:
                # Serialize the landmarks
                serialized_data = serialize_holistic_landmarks(results)
                print(serialized_data)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def main_holistic_improved_face(use_pose = False, use_face = False, use_hands = False ,print_json = False):

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic,\
            mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
            ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = True
            image, results = holistic_detect(image, holistic,face_mesh,use_pose, use_face, use_hands )

            cv2.imshow('MediaPipe Holistic', cv2.flip(image,1))

            if print_json:
                # Serialize the landmarks
                serialized_data = serialize_holistic_landmarks(results)
                print(serialized_data)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
