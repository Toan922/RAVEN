import discord
import io
import time
import asyncio
from PIL import Image, ImageDraw

from raven.backend.scene_similarity import scenes_are_similar

# FR-12: minimum seconds between two alerts, so one sighting doesn't flood the channel
COOLDOWN_SECONDS = 10

# Allow a fresh alert after this long without any positive detection events.
# Brief missed detections should not make a stationary group alert again.
SCENE_RESET_SECONDS = 10

# FR-11.2: seconds to wait before trying a failed alert again
RETRY_SECONDS = 3

class DiscordSubscriber:
    """
    Alert Notification subsystem (architecture sections 2.2 and 6): subscribes to HumanDetected on the
    Detection Event Bus and sends alerts to a Discord webhook (FR-11.1, C-03).
    Keeps track of the Discord webhook, the last scene queued, and the queue of alerts waiting to be sent.
    Suppresses similar scenes as well as enforcing the alert cooldown (FR-12).
    Failed alerts are retried until delivered (FR-11.2, NFR-09). The queue is held in memory, so it covers
    network and Discord outages, but alerts still waiting are lost if RAVEN itself stops.
    """

    def __init__(self, webhook):
        self.webhook = webhook
        self.last_alert_time = None
        self.last_alert_boxes = None
        self.last_detection_time = None
        self.alerts = asyncio.Queue()

    async def on_human_detected(self, event):
        """
        Bus handler for HumanDetected, published only for detections at or above the threshold (FR-10.1).
        Checks scene similarity and the cooldown (FR-12), then queues changed scenes for deliver_alerts(). It never waits
        on Discord, so a slow or failed upload can't hold up the event loop or the pipeline (NFR-05, architecture section 5.2).
        """

        # Empty frames do not publish HumanDetected. A gap in positive events re-arms similarity
        # checking, so people returning to the same positions can trigger another alert.
        if self.last_detection_time is not None and event.time - self.last_detection_time > SCENE_RESET_SECONDS:
            self.last_alert_boxes = None
        self.last_detection_time = event.time

        # check if we should alert based on the cooldown (FR-12)
        if not should_alert(event, self.last_alert_time):
            return

        if self.last_alert_boxes is not None and scenes_are_similar(event.bounding_boxes, self.last_alert_boxes):
            return

        # add the event to the queue, where it is retried until delivered (FR-11.2)
        self.alerts.put_nowait(event)

        # start the cooldown when queued, not when sent, or every frame during an outage would queue another alert (FR-12)
        self.last_alert_time = event.time
        # Keep the last queued scene as the reference. Updating it for skipped events would hide
        # gradual movement or consume a changed scene while the cooldown is still active.
        self.last_alert_boxes = [
            {"box_xyxy": list(box["box_xyxy"])} for box in event.bounding_boxes
        ]

    async def deliver_alerts(self):
        """
        Continuously deliver alerts from the queue to the Discord webhook, retrying failed sends (FR-11.1, FR-11.2, NFR-09).
        Runs as its own asyncio task for the whole session. Sends in a separate thread so an upload never freezes the
        event loop (NFR-03, NFR-05). Alerts are sent in order, so one that keeps failing holds up those behind it.
        """
        while True:
            event = await self.alerts.get()

            # keep trying to send the alert until it succeeds (NFR-09)
            while True:
                try:
                    await asyncio.to_thread(
                        send_alert,
                        event,
                        self.webhook
                    )
                    break
                except Exception as e:
                    print(f"Failed to send alert: {e}. Retrying in {RETRY_SECONDS} seconds...")
                    await asyncio.sleep(RETRY_SECONDS)

            # mark this alert as finished, which is what alerts.join() waits for at shutdown
            self.alerts.task_done()


def render_boxes(event):
    """
    Draw the bounding boxes on the saved frame for the given event, for the alert's frame image (FR-10.2, NFR-07).
    Returns a PIL Image object.
    """

    # get the image from the event's saved frame and create a draw object for it
    image = Image.open(event.frame).convert("RGB")
    draw = ImageDraw.Draw(image)

    for bounding_box in event.bounding_boxes:
        box = bounding_box["box_xyxy"]
        x1, y1, x2, y2 = box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=max(3, image.width // 400)
        )

    return image

def create_discord_alert(event):
    """
    Create a Discord embed message for the given detection event (FR-10.2, NFR-07).
    Coordinates are in decimal degrees (NFR-06). If the frame has no telemetry, the alert says the location is
    not available and still sends (FR-09).
    Not implemented yet: per-person location (FR-08) and the approximate label (FR-09), see docs/TODO.md.
    The location shown is currently the drone's position.
    Returns a discord.Embed object.
    """
    
    # create the embed message for the alert (FR-10.1)
    embed = discord.Embed(
        title=f"{len(event.bounding_boxes)} Human(s) Detected",
        description=f"At Time: {time.strftime('%H:%M:%S', time.localtime(event.time))}",
        color=discord.Color.red()
    )

    # add the location information to the embed (FR-10.2)
    if event.telemetry:
        latitude = event.telemetry["latitude"]
        longitude = event.telemetry["longitude"]
        maps_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

        embed.add_field(
            name="Location",
            value=f"{latitude:.5f}, {longitude:.5f} [View on Maps]({maps_url})",
            inline=False
        )
    else:
        embed.add_field(
            name="Location",
            value="Location not available",
            inline=False
        )

    # add the highest confidence score in the frame to the embed (FR-10.2)
    best_confidence = max(bounding_box["confidence"] for bounding_box in event.bounding_boxes)
    embed.add_field(
        name="Confidence Score",
        value=f"{best_confidence:.2f}",
        inline=False
    )

    # attach the frame image to the embed (FR-10.2)
    embed.set_image(url="attachment://image.png")

    return embed

def should_alert(event, last_alert_time):
    """
    Decide whether this event should trigger an alert, based on the cooldown (FR-12).
    Always alerts on the first event.
    """

    # no alert queued yet
    if last_alert_time is None:
        return True

    # alert only once the cooldown has passed since the last alert
    return event.time - last_alert_time >= COOLDOWN_SECONDS

def send_alert(event, webhook):
    """
    Turn one detection event into a Discord alert (FR-10.2) and post it to the configured channel (FR-11.1).
    Draws the bounding boxes on the saved frame with render_boxes(), builds the embed,
    and sends both to the alerts channel over the HTTPS webhook (NFR-02, C-03).
    Raise an exception if the message fails to send, so the caller can retry it (FR-11.2, NFR-09).
    """
    
    # draw the bounding boxes on the saved frame and save it to a buffer
    buffer = io.BytesIO()
    render_boxes(event).save(buffer, "PNG")
    buffer.seek(0)
    
    # wrap the buffer in a discord.File object
    file = discord.File(fp=buffer, filename="image.png")
    
    # create the embed message for the alert
    embed = create_discord_alert(event)
    
    # send the alert to the Discord channel with the embed message and the image as an attachment
    webhook.send(embed=embed, file=file)
