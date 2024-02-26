import numpy as np
import mediapipe as mp
import cv2
from face_detect import face_detect
from pose_detect import pose_detect
from hand_detect import hand_detect
from serialize_landmarks import serialize_combo_landmarks



def combo_detect(print_json = False,use_pose = False, use_face = False, use_hands = False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
            static_image_mode=False, max_num_faces=1,) as face_mesh,\
            mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands,\
            mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Instead of converting the image to RGB,
            # create a black image of the same size
            height, width, _ = image.shape
            black_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if use_face:
                image,face_results = face_detect(image,face_mesh)
            if use_hands:
                image,hand_results = hand_detect(image,hands)
            if use_pose:
                image, pose_results = pose_detect(image,pose)

            cv2.imshow('MediaPipe FaceMesh', cv2.flip(image,1))

            if print_json:
                # Serialize the landmarks
                serialized_data = serialize_combo_landmarks(
                    face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                    hand_results.multi_hand_landmarks,
                    pose_results.pose_landmarks)
                print(serialized_data)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()