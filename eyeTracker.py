import cv2
import mediapipe as mp

class EyeTracker:
    def __init__(self):
        #load mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  #for video
            max_num_faces=1,
            refine_landmarks = True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    #take fram from web cam and make it to rgb since mediapipe expects that
    def get_landmarks(self,frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #process the frame and detect facial landmarks
        result = self.face_mesh.process(rgb_frame)

        #if landmarks are found, return them (only the first face)
        if result.multi_face_landmarks:
            return result.multi_face_landmarks[0]

            # If no face is detected, return None
        return None


    def draw_landmarks(self, frame, landmarks):
        #use mediapipes drawing function to draw the landmarks
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,  #predefined mesh connections
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),  #green lines
                thickness=1,
                circle_radius=1
            )
        )

    def get_eye_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape  #get height and width of the frame

        #dfine landmark indices for left and right eyes
        #outer corner,inner corner, top eyelid,bottom eyelid
        left_eye_indices = [33, 133, 159, 145]
        right_eye_indices = [362, 263, 386, 374]

        #get coordinates of the 4 poins from the eye(both eyes)
        left_eye = []
        for i in left_eye_indices:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            left_eye.append((x, y))

        right_eye = []
        for i in right_eye_indices:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            right_eye.append((x, y))

        return left_eye, right_eye

    def computeEAR(self,eye_points):
        #eye points=[corner1, corner2,top,bottom]
        leftCorner,rightCorner = eye_points[0],eye_points[1]
        top,bottom = eye_points[2],eye_points[3]

        #horizontal distance between corners,
        hor_dist = ((rightCorner[0] - leftCorner[0]) ** 2 + (rightCorner[1] - leftCorner[1]) ** 2) ** 0.5
        #vertical distance between top and bottom eyelid
        ver_dist = ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5

        if hor_dist == 0:
            return 0
        #Eye Aspect Ratio --- how open or closed an eye is, based on facial landmark positions
        ear = ver_dist / hor_dist  #ratio:vertical / horizontal
        return ear

