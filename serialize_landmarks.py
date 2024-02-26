import json


def serialize_combo_landmarks(face_landmarks,hand_landmarks,pose_landmarks):
    """Function takes in landmarks from individual main components"""
    # Prepare data structure for landmarks
    data = {
        "face": [],
        "hands": []
    }

    # Serialize face landmarks
    if face_landmarks is not None:
        for landmark in face_landmarks.landmark:
            data["face"].append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })

    # Serialize hand landmarks
    if hand_landmarks is not None and len(hand_landmarks) > 0:
        for hand_landmark in hand_landmarks:
            hand_data = []
            for landmark in hand_landmark.landmark:
                hand_data.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
            data["hands"].append(hand_data)

    return json.dumps(data, indent=4)

def serialize_holistic_landmarks(results):
    """Function takes in landmarks from combined holistic main"""
    # Prepare data structure for landmarks
    data = {
        "face": [],
        "hands": {
            "left": [],
            "right": []
        },
        "pose": []
    }

    # Serialize face landmarks
    if results.face_landmarks is not None:
        for landmark in results.face_landmarks.landmark:
            data["face"].append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })

    # Serialize hand landmarks for both left and right hands
    # Left hand
    if results.left_hand_landmarks is not None:
        for landmark in results.left_hand_landmarks.landmark:
            data["hands"]["left"].append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
    # Right hand
    if results.right_hand_landmarks is not None:
        for landmark in results.right_hand_landmarks.landmark:
            data["hands"]["right"].append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })

    # Serialize pose landmarks
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            data["pose"].append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                # Including visibility and presence for pose landmarks, if needed
                "visibility": landmark.visibility if hasattr(landmark, 'visibility') else None,
                "presence": landmark.presence if hasattr(landmark, 'presence') else None
            })

    return json.dumps(data, indent=4)