import os
import cv2
import config


def test_videos_in_directory(folder):
    """
    Check for corrupted MP4 video files
    """
    cap = None
    for folder_name, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(folder_name, filename)
                try:
                    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        raise IOError(f"Cannot open video file: {video_path}")
                except Exception as e:
                    print(f"Error with video {video_path}: {e}")
                finally:
                    cap.release()
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    root_folders = {config.PRETRAIN_DATA_PATH, config.TEST_DATA_PATH, config.TRAINVAL_DATA_PATH}
    for root_folder in root_folders:
        test_videos_in_directory(root_folder)
