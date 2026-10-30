"""
Splice together videos using AIResQ and HIT UAV frames 
"""
import json
import cv2
from pathlib import Path

# define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# define the path to the dataset directory
DATASET_DIRS = [ROOT_DIR / "AIResQ" / "images", ROOT_DIR / "hit-uav" / "images"]

# define the path to the output directory
OUTPUT_DIR = ROOT_DIR / "videos"

# define the image types to pick up
IMAGE_TYPES = [".jpg", ".png"]

# define the frame rate of the output video
FRAME_RATE = 10

# define the total video length in seconds
VIDEO_LENGTH = 110

# define the total number of frames in the video
TOTAL_FRAMES = FRAME_RATE * VIDEO_LENGTH

def list_frames(image_dir: Path) -> list[Path]:
    """
    List all the frames in the given directory
    Return the image paths sorted by name/capture order
    """
    
    # store the frames in a list comprehension and filter by the image types defined above
    frames = [frame for frame in image_dir.rglob("*") if frame.suffix.lower() in IMAGE_TYPES]
    
    # return the sorted list of frames by name to ensure they are in capture order
    return sorted(frames, key=lambda x: x.name)

def make_video(frames: list[Path], output_path: Path) -> None:
    """
    Write the frames into an mp4 video file, in the given order
    Every frame is resized to the size of the first frame, and the video is written at the given frame rate
    """
    
    # ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # first frame is used to get the size of the video
    first_frame = cv2.imread(str(frames[0]))
    size = (first_frame.shape[1], first_frame.shape[0])
    
    # open the video file with output path, codec, frame rate and size
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, size)
    
    # iterate over the frames and write them to the video file
    for frame in frames:
        # read the frame and resize it to the size of the first frame
        image = cv2.imread(str(frame))
        
        # skip the frame if it is None (e.g. if the image file is corrupted)
        if image is None:
            print(f"Skipping frame {frame} because it is None")
            continue
        
        # resize the image to the size of the first frame
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size)
        
        # write the frame to the video file
        writer.write(image)

    # release the video writer
    writer.release()

def main() -> None:
    """
    Make a video from the frames in the dataset directories
    """
    
    # go over each dataset directory and make a video from the frames in that directory
    for dataset_dir in DATASET_DIRS:
        # list the frames in the dataset directory
        frames = list_frames(dataset_dir)
        
        # set the video name based on directory
        output_path = OUTPUT_DIR / f"{dataset_dir.parent.name}.mp4"
        
        # make the video from the sampled frames
        make_video(frames[:TOTAL_FRAMES], output_path)

if __name__ == "__main__":
    main()