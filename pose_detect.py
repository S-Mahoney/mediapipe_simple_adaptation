import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def pose_detect(image, pose):
    pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return image, pose_results

