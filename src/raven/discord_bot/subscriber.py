import discord
import io
import time
import asyncio
from PIL import Image, ImageDraw

from raven.backend.scene_similarity import RecentPeople, scenes_are_similar

# FR-12: a person already alerted isn't alerted again while they stay in view; they're forgotten once unseen
# for this long. Without ground positions (no telemetry) it's also the minimum time between two alerts.
DUPLICATE_WINDOW_SECONDS = 10

# default minimum seconds between two alerts from one drone when people can be told apart by ground position:
# only protects the Discord channel from bursts, so a new person waits at most this long (NFR-03: 5 s)
COOLDOWN_SECONDS = 2

# FR-11.2: seconds to wait before trying a failed alert again
RETRY_SECONDS = 3

# people listed in one alert: Discord rejects embeds over 25 fields, and a rejected alert would retry forever
MAX_PEOPLE_LISTED = 10

class DiscordSubscriber:
    """
    Alert Notification subsystem (architecture sections 2.2 and 6): subscribes to HumanDetected on the
    Detection Event Bus and sends alerts to a Discord webhook (FR-11.1, C-03).
    Keeps track of the Discord webhook, who was already alerted, and the queue of alerts waiting to be sent.
    Suppresses duplicate alerts for people already alerted (FR-12): by ground position when geotag estimated one
    for every person, otherwise by comparing person boxes with the last alert, behind a DUPLICATE_WINDOW_SECONDS
    cooldown. Each drone has its own state (events from a single feed all have drone ""), so one busy drone
    can't hide another drone's sightings.
    on_sent(event) and on_error(event, error) are optional callbacks (the operator console's status light uses them).
    Failed alerts are retried until delivered (FR-11.2, NFR-09). The queue is held in memory, so it covers
    network and Discord outages, but alerts still waiting are lost if RAVEN itself stops.
    """

    def __init__(self, webhook, cooldown=COOLDOWN_SECONDS, on_sent=None, on_error=None):
        self.webhook = webhook
        self.cooldown = cooldown
        self.on_sent = on_sent
        self.on_error = on_error

        # per drone: time of the last alert queued, boxes of the last alert queued, time of the last positive
        # detection, and the people already alerted (RecentPeople)
        self.last_alert_times = {}
        self.last_alert_boxes = {}
        self.last_detection_times = {}
        self.recent_people = {}
        self.alerts = asyncio.Queue()

    async def on_human_detected(self, event):
        """
        Bus handler for HumanDetected, published only for detections at or above the threshold (FR-10.1).
        Queues an alert for deliver_alerts() when the frame shows someone not yet alerted (FR-12). It never waits
        on Discord, so a slow or failed upload can't hold up the event loop or the pipeline (NFR-05, architecture section 5.2).
        """
        drone = getattr(event, "drone", "")

        # Empty frames do not publish HumanDetected. A gap in this drone's positive events re-arms the box
        # comparison, so people returning to the same positions can trigger another alert.
        last_detection = self.last_detection_times.get(drone)
        if last_detection is not None and event.time - last_detection > DUPLICATE_WINDOW_SECONDS:
            self.last_alert_boxes.pop(drone, None)
        self.last_detection_times[drone] = event.time

        locations = [box.get("location") for box in event.bounding_boxes]
        new = []
        if locations and all(locations):
            people = self.recent_people.setdefault(drone, RecentPeople(DUPLICATE_WINDOW_SECONDS))
            new = people.new_people(locations, event.time)
            # a new person held back by the cooldown stays new, so the next frame alerts for them
            if not new or not should_alert(event, self.last_alert_times.get(drone), self.cooldown):
                return
        elif not self._new_scene(event, drone):
            return

        # add the event to the queue, where it is retried until delivered (FR-11.2)
        self.queue_alert(event)
        if new:
            # only once queueing succeeded: if the outbox write raised, they stay new and the next frame retries (NFR-09)
            people.remember(new, event.time)

        # start the cooldown when queued, not when sent, or every frame during an outage would queue another alert (FR-12)
        self.last_alert_times[drone] = event.time
        # Keep the last queued scene as the reference. Updating it for skipped events would hide
        # gradual movement or consume a changed scene while the cooldown is still active.
        self.last_alert_boxes[drone] = [
            {"box_xyxy": list(box["box_xyxy"])} for box in event.bounding_boxes
        ]

    def _new_scene(self, event, drone):
        """
        Fallback when some person has no ground position: boxes in the image can't follow people while the camera
        moves, so FR-12's window is kept as a cooldown, and after it only a changed scene alerts.
        """
        if not should_alert(event, self.last_alert_times.get(drone), max(self.cooldown, DUPLICATE_WINDOW_SECONDS)):
            return False
        previous_boxes = self.last_alert_boxes.get(drone)
        return previous_boxes is None or not scenes_are_similar(event.bounding_boxes, previous_boxes)

    def queue_alert(self, event):
        """
        Put an alert that passed the cooldown and similarity checks on the queue. Subclasses override it to also
        keep a durable copy (the console's outbox), so only alerts that will really be sent are recorded.
        """
        self.alerts.put_nowait(event)

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
                    if self.on_error is not None:
                        self.on_error(event, e)
                    await asyncio.sleep(RETRY_SECONDS)

            if self.on_sent is not None:
                self.on_sent(event)

            # mark this alert as finished, which is what alerts.join() waits for at shutdown
            self.alerts.task_done()


def render_boxes(event):
    """
    Draw the bounding boxes on the saved frame for the given event, for the alert's frame image (FR-10.2, NFR-07).
    Returns a PIL Image object.
    """

    # get the image from the event's saved frame and create a draw object for it; the operator console saves
    # frames encrypted, so they're decrypted in memory here (plain files are read as they are)
    from raven.backend.vault import read_bytes

    image = Image.open(io.BytesIO(read_bytes(event.frame))).convert("RGB")
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

def alert_people(event):
    """
    The people in a detection event, numbered for the alert so the numbers on the frame image and the
    embed's fields match (FR-10.2, NFR-07). Numbered from 1, highest confidence first.
    Returns a list of dicts, one per bounding box:
    - number: the person's number in the alert
    - box: their [x1, y1, x2, y2] in the frame
    - confidence: the model's confidence score
    - location: {"latitude", "longitude"} in decimal degrees (NFR-06), or None
    - status: how location was found (FR-08, FR-09):
        "estimated" (the person's own ground position, from geotag),
        "approximate" (geotag couldn't place them, so the drone's position from the frame's telemetry), or
        "unavailable" (no position data, location is None; the alert still sends)
    """
    people = []

    # the drone's own position, the fallback for people geotag couldn't place: only the coordinates, and only
    # when the telemetry row has both (it also holds altitude, heading and the receiver's timestamp)
    telemetry = event.telemetry or {}
    drone_position = None
    if "latitude" in telemetry and "longitude" in telemetry:
        drone_position = {"latitude": telemetry["latitude"], "longitude": telemetry["longitude"]}

    # sort the bounding boxes by confidence, highest first
    sorted_boxes = sorted(event.bounding_boxes, key=lambda box: box["confidence"], reverse=True)

    for number, bounding_box in enumerate(sorted_boxes, start=1):
        box = bounding_box["box_xyxy"]
        confidence = bounding_box["confidence"]
        location = bounding_box.get("location")
        if location:
            status = "estimated"
        elif drone_position:
            location, status = drone_position, "approximate"
        else:
            location, status = None, "unavailable"
        people.append({
            "number": number,
            "box": box,
            "confidence": confidence,
            "location": location,
            "status": status
        })

    return people

def create_discord_alert(event):
    """
    Create a Discord embed message for the given detection event (FR-10.2, NFR-07).
    Each person gets their own line, numbered as in alert_people(): classification, confidence score and their own
    location in decimal degrees (NFR-06) with a Google Maps link. The location is the person's estimated ground
    position (FR-08); otherwise the drone's position, labelled approximate, or "Location unavailable" when the frame
    has no position data, and the alert still sends (FR-09). Crowded frames list the MAX_PEOPLE_LISTED most
    confident people, then how many more are in the image.
    Returns a discord.Embed object.
    """
    
    # create the embed message for the alert (FR-10.1), naming the drone when several feeds share the channel
    title = f"{len(event.bounding_boxes)} Human(s) Detected"
    if getattr(event, "drone", ""):
        title = f"{event.drone}: {title}"
    embed = discord.Embed(
        title=title,
        description=f"At Time: {time.strftime('%H:%M:%S', time.localtime(event.time))}",
        color=discord.Color.red()
    )

    # one line per person (FR-10.2), most confident first
    people = alert_people(event)
    for person in people[:MAX_PEOPLE_LISTED]:
        location = person["location"]
        if location is None:
            value = "Location unavailable"
        else:
            latitude, longitude = location["latitude"], location["longitude"]
            maps_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
            value = f"{latitude:.5f}, {longitude:.5f} · [Open in Google Maps]({maps_url})"
            if person["status"] == "approximate":
                value = f"Approximate (drone position): {value}"
        embed.add_field(name=f"Person {person['number']} · Human · {person['confidence']:.2f}", value=value, inline=False)
    if len(people) > MAX_PEOPLE_LISTED:
        embed.add_field(name="More people", value=f"+{len(people) - MAX_PEOPLE_LISTED} more in the image", inline=False)

    # attach the frame image to the embed (FR-10.2)
    embed.set_image(url="attachment://image.png")

    return embed

def should_alert(event, last_alert_time, cooldown=COOLDOWN_SECONDS):
    """
    Decide whether this event should trigger an alert, based on the cooldown in seconds (FR-12).
    Always alerts on the first event.
    """

    # no alert queued yet
    if last_alert_time is None:
        return True

    # alert only once the cooldown has passed since the last alert
    return event.time - last_alert_time >= cooldown

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
