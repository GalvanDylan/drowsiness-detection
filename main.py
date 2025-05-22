import cv2
from eyeTracker import EyeTracker  #Import our custom class
from scipy import ndimage as nd
import pygame

pygame.mixer.init()
pygame.mixer.music.load("recording.mp3")  #alarm sound

def main():
    cap = cv2.VideoCapture(0)  #use default webcam
    tracker = EyeTracker()
    #thresholds
    blink_threshold = 0.25
    blink_frame_count = 0
    drowsy_frame_count = 0

    BLINK_FRAME_MIN = 3
    DROWSY_FRAME_MIN = 80
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #apply Gaussian filter ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = nd.gaussian_filter(gray, sigma=1)
        blurred_3ch = cv2.merge([blurred] * 3)

        landmarks = tracker.get_landmarks(blurred_3ch)
        if landmarks:
            #tracker.draw_landmarks(frame,landmarks)

            #Get eye points
            left_eye, right_eye = tracker.get_eye_landmarks(frame, landmarks)

            #draw circles on each eye point
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # red dot

            #compute EAR (Eye Aspect Ratio)
            left_ear = tracker.computeEAR(left_eye)
            right_ear = tracker.computeEAR(right_eye)
            avgEAR = (left_ear + right_ear)/2.0

            #show EAR value on screen
            cv2.putText(frame, f"EAR: {avgEAR:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            #threshold for blink detection
            if avgEAR < blink_threshold:
                blink_frame_count += 1
                drowsy_frame_count += 1
            else:
                if blink_frame_count >= BLINK_FRAME_MIN:
                    cv2.putText(frame, "BLINK", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                blink_frame_count = 0
                drowsy_frame_count = 0
            #show DROWSY alert
            if drowsy_frame_count >= DROWSY_FRAME_MIN:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                cv2.putText(frame, "Wake Up!!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)


        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:  #eSC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




