"""
    Mock alert for testing purposes and designing the formatting of the alert message for future use
"""
import discord
import random
import json
import os
import io
from PIL import Image, ImageDraw
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv

# get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[3]

# join the root directory with the path to the image
HIT_UAV_IMAGE_PATH = ROOT_DIR / "clean_HITUAV/splits/test.txt"
AIRESQ_IMAGE_PATH = ROOT_DIR / "clean_AIResQ/splits/test.txt"

SPLITS = [HIT_UAV_IMAGE_PATH, AIRESQ_IMAGE_PATH]

# define a dataclass for the camera model and its FOVs
@dataclass
class Camera:
    model: str
    horizontal_fov_deg: float
    vertical_fov_deg: float

# reference the camera models based on the metadata json files and derive FOVs from the camera model's specifications
HIT_UAV_CAMERA = Camera("Zenmuse XT2 25mm", 24.6, 19.7)
AIRESQ_CAMERA = Camera("VarioCAM HD 30mm", 32.3, 24.6)

# create a dictionary to map the source to the camera
CAMERAS = {
    "HU": HIT_UAV_CAMERA,
    "AQ": AIRESQ_CAMERA
}

@dataclass
class Telemetry:
    latitude: float
    longitude: float
    altitude_agl_m: float = field(default_factory=lambda: random.randint(50, 120))
    camera_depression_deg: float = field(default_factory=lambda: random.randint(30, 60))
    heading_deg: float = field(default_factory=lambda: random.randint(0, 359))
    speed_mps: float = field(default_factory=lambda: round(random.uniform(3, 10), 1))
    battery_percent: int = field(default_factory=lambda: random.randint(20, 100))

def pick_frame() -> tuple[Path, Path]:
    """
    Pick a random frame from the given path
    Return a tuple of the frame path and the label path
    """
    
    # init the frames list
    frames = []
    
    # go over the 2 datasets and read the frames from the test.txt file
    for split in SPLITS:
        # use extend to keep the frames list flat instead of nested
        # .split() to split the text file into a list of frames
        frames.extend(split.read_text().split())
    
    while True:
        # pick a random frame from the frames list and keep it as a Path object
        frame = Path(random.choice(frames))
        
        # build the label path by going to the parent directory of the frame and then to the labels directory and replacing the extension with .txt
        label = (Path(frame).parent.parent / "labels" / frame.name).with_suffix(".txt")
        
        # check if the label file exists and if the label is a valid label (not empty)
        if label.exists() and label.read_text().strip():
            return frame, label

def load_metadata(frame: Path) -> dict:
    """
    Load the metadata for a given frame.
    """
    
    
    # get the metadata path then read the metadata json file and load it into a dictionary
    metadata_path = (Path(frame).resolve().parent.parent / "metadata").with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())

    # get the frame's metadata entry using the frame's name
    return metadata.get(frame.name, {})

def make_telemetry(frame_metadata: dict) -> Telemetry:
    """
    Create a Telemetry object based on the frame's metadata
    Update the telemetry data with the values from the metadata if the frame's metadata source is "HU"
    """
    
    
    # generate random telemetry data
    telemetry = Telemetry(latitude = 47.41, longitude = 10.28)

    # check if the frame's metadata source is "HU" and if it is, update the telemetry data with the values from the metadata
    if frame_metadata["source"] == "HU":
        telemetry.altitude_agl_m = frame_metadata["altitude_m"]
        telemetry.camera_depression_deg = frame_metadata["camera_angle_deg"]

    return telemetry

def mock_event() -> dict:
    """
    Create a mock event for testing purposes
    """

    # pick a random frame and label and load the metadata for the frame
    frame, label = pick_frame()
    frame_metadata = load_metadata(frame)

    # derive the camera from the frame's metadata source
    camera = CAMERAS[frame_metadata["source"]]

    # get the frame's telemetry data
    telemetry = make_telemetry(frame_metadata)

    # get the image height and width using PIL
    width, height = Image.open(frame).size
    
    bounding_boxes = read_boxes(label, width, height)
    
    # return a dictionary with the compiled information for the mock event
    return {
        "frame": str(frame),
        "label": str(label),
        "metadata": frame_metadata,
        "camera": asdict(camera),
        "telemetry": asdict(telemetry),
        "image_size": {
            "width": width,
            "height": height
        },
        "bounding_boxes": bounding_boxes
    }

def read_boxes(label: Path, width: int, height: int) -> list:
    """
    Read the bounding boxes from the label file and return a list.
    Each dictionary entry will contain the bounding box coordinates in pixel values and the confidence score of the bounding box.
    """
    
    # store all the bounding boxes in a list with the bounding box coordinates in pixel values and the confidence score of the bounding box
    boxes = []

    # get all the bounding boxes from the label file and convert them to pixel coordinates
    for line in label.read_text().splitlines():
        # read the center x, center y, width, and height of the bounding box
        _, x_center, y_center, box_width, box_height = map(float, line.split())
        x1 = round((x_center - box_width / 2) * width)
        x2 = round((x_center + box_width / 2) * width)
        y1 = round((y_center - box_height / 2) * height)
        y2 = round((y_center + box_height / 2) * height)
        
        confidence = round(random.uniform(0.55, 0.95), 2)  # generate a random confidence score between 0.55 and 0.95
        boxes.append({"box_xyxy": [x1, y1, x2, y2], "confidence": confidence})

    return boxes

def render_boxes(event: dict) -> Image.Image:
    """
    Render the bounding boxes on the image and return the image with the bounding boxes.
    """

    # open the image using PIL
    image = Image.open(event["frame"]).convert("RGB")
    
    # create a draw object to draw on the image
    draw = ImageDraw.Draw(image)
    
    # scale the line width of the bounding boxes based on the image width
    line_width = max(3, image.width // 400)
    
    # draw each bounding box on the image
    for box in event["bounding_boxes"]:
        x1, y1, x2, y2 = box["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline = "red", width = line_width)

    return image

def create_mock_alert(event: dict) -> discord.Embed:
    """
    Create a mock alert for testing purposes
    """
    
    # get the person found count in the frame
    number_of_people = len(event["bounding_boxes"])
    
    # alter the title of the alert based on the number of people found in the frame
    if number_of_people == 1:
        title = "Person Detected"
    else:
        title = f"{number_of_people} People Detected"

    # pass in the event's telemetry data to the embed message
    telemetry = event["telemetry"]
    latitude = telemetry["latitude"]
    longitude = telemetry["longitude"]
    
    # create a google maps url with the latitude and longitude
    google_maps_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
    
    # create the description message with the latitude and longitude and the google maps url
    description_message = f"[{latitude:.5f}, {longitude:.5f}]({google_maps_url})\nDrone Position (approximate)"
    
    # create the embed message with the title, description, and color
    embed = discord.Embed(
        title = title,
        color = discord.Color.red(),
        description = description_message
    )
    
    # add the confidence score of the highest confidence bounding box to the embed message
    best_confidence = max(box["confidence"] for box in event["bounding_boxes"])
    embed.add_field(name = "Confidence Score", value = f"{best_confidence:.0%}", inline = True)
    
    # get the remainder of the event's telemetry data and add it to the embed message
    telemetry = event["telemetry"]
    embed.add_field(name = "Altitude (AGL)", value = f"{telemetry['altitude_agl_m']:.0f} m", inline = True)
    embed.add_field(name = "Camera Depression", value = f"{telemetry['camera_depression_deg']:.0f}°", inline = True)
    embed.add_field(name = "Heading", value = f"{telemetry['heading_deg']:.0f}°", inline = True)
    
    # add the image to the embed message
    embed.set_image(url = "attachment://image.png")
    
    return embed

if __name__ == "__main__":
    load_dotenv()
    
    # get the discord token and channel id from the .env file
    token = os.getenv("DISCORD_TOKEN")
    channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))
    
    # create a mock event for testing purposes
    event = mock_event()
    
    # save the image with the bounding boxes to buffer
    buffer = io.BytesIO()
    render_boxes(event).save(buffer, "PNG")
    buffer.seek(0)
    
    # wrap the buffer in a discord.File object to send it as an attachment
    file = discord.File(fp=buffer, filename="image.png")
    
    # create the embed message
    embed = create_mock_alert(event)
    
    # create a discord client and send the embed message to the channel
    client = discord.Client(intents=discord.Intents.default())
    
    @client.event
    async def on_ready():
        """
        Event handler for when the bot is ready
        """
    
        # get the channel object using the channel id
        channel = await client.fetch_channel(channel_id)
        await channel.send(embed=embed, file=file)
        await client.close()
        
    client.run(token)