import cv2
import numpy as np


class VideoPreprocessorSimple:

    def __init__(self):
        pass

    def crop_video_96_96(self, video_path):
        cap = cv2.VideoCapture(video_path)

        sequence = []
        #fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        #out = cv2.VideoWriter("test_done.mp4", fourcc, fps, (96, 96), isColor=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped_frame = self._crop_center(gray, 96, 96)
            sequence.append(cropped_frame)

            #out.write(cropped_frame)

        cap.release()
        #out.release()
        cv2.destroyAllWindows()

        return np.asarray(sequence)

    def _crop_center(self, frame, crop_width, crop_height):
        height, width = frame.shape
        start_x = width//2 - (crop_width//2)
        start_y = height//2 - (crop_height//2)
        return frame[start_y:start_y+crop_height, start_x:start_x+crop_width]




