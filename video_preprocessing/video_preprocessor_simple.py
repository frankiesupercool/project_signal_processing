import cv2
import numpy as np
import random

class VideoPreprocessorSimple:

    def __init__(self, target_frames=100, fps=25.0):
        """
        Initializes the video preprocessor.

        Args:
            target_frames (int): Number of frames corresponding to the fixed audio duration.
            fps (float): Frames per second of the videos.
        """
        self.target_frames = target_frames  # 4 seconds × 25 FPS = 100 frames
        self.fps = fps

    def crop_video_96_96(self, video_path):
        """
        Crops the video to a fixed number of frames and resizes each frame to 96x96 pixels.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Array of cropped frames with shape [target_frames, 96, 96].
        """
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps != self.fps:
            #print(f"Warning: Video {video_path} has FPS {actual_fps}, expected {self.fps}. Adjusting accordingly.")
            self.fps = actual_fps  # Update fps to actual if different

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        required_frames = self.target_frames

        if total_frames < required_frames:
            #print(f"Warning: Video {video_path} has only {total_frames} frames. Padding to reach {required_frames} frames.")
            frames = self._read_all_frames(cap)
            frames = self._pad_frames(frames, required_frames)
        else:
            # Option 1: Use the first 4 seconds (first 100 frames)
            frames = self._read_first_n_frames(cap, required_frames)
            # Option 2: Uncomment below to use a random 4-second window
            # frames = self._read_random_n_frames(cap, required_frames, total_frames)

        cap.release()
        cv2.destroyAllWindows()

        frames = self._crop_and_resize(frames, 96, 96)
        return np.asarray(frames)

    def _read_all_frames(self, cap):
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _read_first_n_frames(self, cap, n):
        frames = []
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _read_random_n_frames(self, cap, n, total_frames):
        """
        Reads a contiguous random window of n frames from the video.

        Args:
            cap (cv2.VideoCapture): Opened video capture object.
            n (int): Number of frames to read.
            total_frames (int): Total number of frames in the video.

        Returns:
            list: List of frames.
        """
        max_start = total_frames - n
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = self._read_first_n_frames(cap, n)
        return frames

    def _pad_frames(self, frames, target_count):
        """
        Pads the list of frames to reach the target count by repeating the last frame.

        Args:
            frames (list): List of frames.
            target_count (int): Desired number of frames.

        Returns:
            list: Padded list of frames.
        """
        pad_length = target_count - len(frames)
        if not frames:
            # If no frames were read, create blank frames
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Assuming default frame size
            frames = [blank_frame for _ in range(target_count)]
        else:
            last_frame = frames[-1]
            frames += [last_frame for _ in range(pad_length)]
        return frames

    def _crop_and_resize(self, frames, crop_width, crop_height):
        """
        Converts frames to grayscale, crops the center, and resizes to the desired dimensions.

        Args:
            frames (list): List of BGR frames.
            crop_width (int): Width of the cropped frame.
            crop_height (int): Height of the cropped frame.

        Returns:
            list: List of processed frames.
        """
        processed_frames = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped = self._crop_center(gray, crop_width, crop_height)
            processed_frames.append(cropped)
        return processed_frames

    @staticmethod
    def _crop_center(frame, crop_width, crop_height):
        """
        Crops the center of the frame.

        Args:
            frame (np.ndarray): Grayscale frame.
            crop_width (int): Width of the crop.
            crop_height (int): Height of the crop.

        Returns:
            np.ndarray: Cropped frame.
        """
        height, width = frame.shape
        start_x = max(width // 2 - (crop_width // 2), 0)
        start_y = max(height // 2 - (crop_height // 2), 0)
        return frame[start_y:start_y + crop_height, start_x:start_x + crop_width]





