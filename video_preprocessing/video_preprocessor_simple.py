import cv2
import numpy as np


class VideoPreprocessorSimple:

    def crop_video_96_96(self, video_path):
        cap = cv2.VideoCapture(video_path)

        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped_frame = self._crop_center(gray, 96, 96)
            sequence.append(cropped_frame)

        cap.release()
        cv2.destroyAllWindows()

        return np.asarray(sequence)

    @staticmethod
    def _crop_center(frame, crop_width, crop_height):
        height, width = frame.shape
        start_x = width//2 - (crop_width//2)
        start_y = height//2 - (crop_height//2)
        return frame[start_y:start_y+crop_height, start_x:start_x+crop_width]




