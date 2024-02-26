"""
Program to detect hand, face and pose position in live camera footage
Written: 25/02/2024
Script Author: Sam Mahoney
Integrated Face, pose and hand detection originally created by MediaPipe ML solutions

"""

"""Main functions"""
from holistic import holistic_detect, main_holistic,main_holistic_improved_face
from combo_detect import combo_detect

"""Sub functions of combo_detect"""
# from hand_detect import hand_detect
# from face_detect import face_detect
# from pose_detect import pose_detect

"""Functions to convert landmarks to JSON format"""
# from serialize_landmarks import serialize_combo_landmarks,serialize_holistic_landmarks


if __name__ == "__main__":
    print("Starting..")
    main_holistic_improved_face(use_pose = False, use_face = False, use_hands = True ,print_json=False)
    # main_holistic(use_pose = False, use_face = False, use_hands = True ,print_json=False)
    # combo_detect(use_pose = False, use_face = False, use_hands = True ,print_json=False)