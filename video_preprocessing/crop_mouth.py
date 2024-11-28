import os

import cv2
import numpy as np
import dlib
import cv2
import numpy as np
from skimage import transform as tf

class VideoCrop:

    def __init__(self):
        # Load the video
        self.root_dir = 'data'
        self.output_root_dir = 'data_preprocessed'
        # Initialize the Dlib face detector and predictor
        self. detector = dlib.get_frontal_face_detector()
        self.predictor_path = 'shape_predictor_68_face_landmarks.dat'  # path to the landmark predictor
        self.predictor = dlib.shape_predictor(self.predictor_path)


    # Define function to extract the mouth region
    def extract_mouth(landmarks):
        # Mouth landmarks correspond to indices 48 to 67
        mouth_points = landmarks.parts()[48:68]
        mouth_points = np.array([(point.x, point.y) for point in mouth_points])

        # Calculate the bounding box for the mouth
        x_min = min(mouth_points[:, 0])
        x_max = max(mouth_points[:, 0])
        y_min = min(mouth_points[:, 1])
        y_max = max(mouth_points[:, 1])

        # Return the coordinates to crop
        return (x_min, y_min, x_max, y_max)

    def run(self):
        wav_files = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mp4'):

                    video_path = os.path.join(subdir, file)
                    print(f"Input Path: {video_path}")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    output_path = os.path.join(output_root_dir, subdir.split('/', 1)[1], file)
                    output_dir = os.path.dirname(output_path)

                    frame_idx = 0
                    frame_gen = self.load_video(video_path)
                    while True:
                        try:
                            frame = frame_gen.__next__()  ## -- BGR
                        except StopIteration:
                            break
                        if frame_idx == 0:
                            sequence = []

                            sequence_frame = []
                            sequence_landmarks = []

                        window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
                        smoothed_landmarks = np.mean(
                            [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)],
                            axis=0)
                        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
                        transformed_frame, transformed_landmarks = self.affine_transform(frame, smoothed_landmarks,
                                                                                         self._reference,
                                                                                         grayscale=self._convert_gray)
                        sequence.append(self.cut_patch(transformed_frame,
                                                       transformed_landmarks[self._start_idx:self._stop_idx],
                                                       self._crop_height // 2,
                                                       self._crop_width // 2, ))

                        sequence_frame.append(transformed_frame)
                        sequence_landmarks.append(transformed_landmarks)

                        frame_idx += 1





                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(f"OutputPath: {output_path}")
                    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height),isColor = False)
                    # Process video frames
                    i = 0
                    while cap.isOpened():
                        print(f"Frame: {i}")
                        i = i+1
                        ret, frame = cap.read()
                        if not ret:
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        out.write(gray)

                        # Detect faces
                        faces = detector(gray)

                        for face in faces:
                            landmarks = predictor(gray, face)
                            # preprocessed_landmarks = self.landmarks_interpolate(landmarks)
                            # -- Step 3, exclude corner case: 1) no landmark in all frames; 2) number of frames is less than window length.
                            #if not preprocessed_landmarks or len(preprocessed_landmarks) < self._window_margin: return
                            # -- Step 4, affine transformation and crop patch
                            sequence, transformed_frame, transformed_landmarks = self.crop_patch(video_pathname,
                                                                                                 preprocessed_landmarks)
                            assert sequence is not None, "cannot crop from {}.".format(filename)
                            return sequence

                            # Press 'q' to exit
                            out.write(mouth_crop)


                    # Release the video capture object
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()


    def landmarks_interpolate(self, landmarks):
        """landmarks_interpolate.

        :param landmarks: List, the raw landmark (in-place)

        """
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx:
            return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = self.linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        # -- Corner case: keep frames at the beginning or at the end failed to be detected.
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
        return landmarks

    def crop_patch(self, video_pathname, landmarks):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """

        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            try:
                frame = frame_gen.__next__()  ## -- BGR
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []

            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(frame, smoothed_landmarks, self._reference,
                                                                             grayscale=self._convert_gray)
            sequence.append(self.cut_patch(transformed_frame,
                                      transformed_landmarks[self._start_idx:self._stop_idx],
                                      self._crop_height // 2,
                                      self._crop_width // 2, ))


            frame_idx += 1
        return np.array(sequence)


    def affine_transform(self, frame, landmarks, reference, grayscale=False, target_size=(256, 256),
                         reference_size=(256, 256), stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                         border_value=0):
        """affine_transform.

        :param frame:
        :param landmarks:
        :param reference: ndarray, the neutral reference frame.
        :param grayscale: bool, save as grayscale if set as True.
        :param target_size: tuple, size of the output image.
        :param reference_size: tuple, size of the neural reference frame.
        :param stable_points: tuple, landmark idx for the stable points.
        :param interpolation: interpolation method to be used.
        :param border_mode: Pixel extrapolation method .
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(frame,
                                           transform,
                                           dsize=(target_size[0], target_size[1]),
                                           flags=interpolation,
                                           borderMode=border_mode,
                                           borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks

    def linear_interpolate(self, landmarks, start_idx, stop_idx):
        """linear_interpolate.

        :param landmarks: ndarray, input landmarks to be interpolated.
        :param start_idx: int, the start index for linear interpolation.
        :param stop_idx: int, the stop for linear interpolation.
        """
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks
        for idx in range(1, stop_idx - start_idx):
            landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
        return landmarks

    def warp_img(self, src, dst, img, std_size):
        """warp_img.

        :param src: ndarray, source coordinates.
        :param dst: ndarray, destination coordinates.
        :param img: ndarray, an input image.
        :param std_size: tuple (rows, cols), shape of the output image generated.
        """
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    def apply_transform(self, transform, img, std_size):
        """apply_transform.

        :param transform: Transform object, containing the transformation parameters \
                          and providing access to forward and inverse transformation functions.
        :param img: ndarray, an input image.
        :param std_size: tuple (rows, cols), shape of the output image generated.
        """
        warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped

    def cut_patch(self, img, landmarks, height, width, threshold=5):
        """cut_patch.

        :param img: ndarray, an input image.
        :param landmarks: ndarray, the corresponding landmarks for the input image.
        :param height: int, the distance from the centre to the side of of a bounding box.
        :param width: int, the distance from the centre to the side of of a bounding box.
        :param threshold: int, the threshold from the centre of a bounding box to the side of image.
        """
        center_x, center_y = np.mean(landmarks, axis=0)

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')

        cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                             int(round(center_x) - round(width)): int(round(center_x) + round(width))])
        return cutted_img

    def convert_bgr2gray(self, sequence):
        """convert_bgr2gray.

        :param sequence: ndarray, the RGB image sequence.
        """
        return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in sequence], axis=0)

    def load_video(self, filename):
        """load_video.

        :param filename: str, the fileanme for a video sequence.
        """
        cap = cv2.VideoCapture(filename)
        while(cap.isOpened()):
            ret, frame = cap.read() # BGR
            if ret:
                yield frame
            else:
                break
        cap.release()