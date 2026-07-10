import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh

def run_head_pose():
    cap = cv2.VideoCapture(0)
    
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3D model points for head pose
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth
        (150.0, -150.0, -125.0)   # Right mouth
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]

            # Extract key facial landmarks
            landmarks = face.landmark
            
            image_points = np.array([
                (landmarks[1].x * w, landmarks[1].y * h),   # Nose tip
                (landmarks[152].x * w, landmarks[152].y * h),  # Chin
                (landmarks[33].x * w, landmarks[33].y * h),    # Left eye corner
                (landmarks[263].x * w, landmarks[263].y * h),  # Right eye corner
                (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth
                (landmarks[291].x * w, landmarks[291].y * h),  # Right mouth
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # Project nose direction
            nose_end_3d = np.array([[0, 0, 1000.0]])
            nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            # Convert rotation vector to angles
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, *_ = cv2.RQDecomp3x3(rmat)

            yaw, pitch, roll = angles * 180/np.pi

            # Display text
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # Direction classification
            direction = "Center"
            if yaw > 15:
                direction = "Looking Right"
            elif yaw < -15:
                direction = "Looking Left"
            elif pitch > 10:
                direction = "Looking Down"
            elif pitch < -10:
                direction = "Looking Up"

            cv2.putText(frame, direction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.imshow("Head Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_head_pose()
