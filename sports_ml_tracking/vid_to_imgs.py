import cv2
import os

def save_frames(video_path, output_folder):
    vid = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    success, image = vid.read()
    count = 0
    while success:
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, image)
        print(f"Saved frame {count}")
        
        success, image = vid.read()
        count += 1

    print("Finished saving frames")
    vid.release()

if __name__ == "__main__":
    video_path = "test_assets/cropped.mp4"
    output_folder = "/home/gilberto/Downloads/vid_to_imgs/cropped"
    save_frames(video_path, output_folder)