import redis
import json
import os
import discord
import io
from dotenv import load_dotenv
from mock_alert import create_mock_alert, render_boxes

# setup the redis stream name
REDIS_STREAM_NAME = "raven:detections"

# the consumer group the Discord bot reads as, and this reader's name within it
GROUP_NAME = "discord"
CONSUMER_NAME = "bot-1"

# minimum video time between two alerts, so one sighting doesn't flood the channel
COOLDOWN_SECONDS = 10

def create_group(redis_client):
    """
    Create the consumer group on the detections stream so the bot has a bookmark of which events it has handled.
    Starts from the oldest message (id "0") and creates the stream if ingest hasn't published yet.
    Safe to call on every startup: if the group already exists, it does nothing.
    """
    
    # create the consumer group on the stream, starting from the oldest message (id "0")
    try:
        redis_client.xgroup_create(REDIS_STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass

def should_alert(event, last_alert_time):
    """
    Decide whether this event should trigger an alert, based on the cooldown.
    Always alerts on the first event, and when the video time goes backwards (a new video started).
    """

    # no alert sent yet
    if last_alert_time is None:
        return True

    # time went backwards, so this is a new video and the old cooldown doesn't apply
    if event["time"] < last_alert_time:
        return True

    # alert only once the cooldown has passed since the last alert
    return event["time"] - last_alert_time >= COOLDOWN_SECONDS

def send_alert(event, webhook):
    """
    Turn one detection event into a Discord alert.
    Draws the bounding boxes on the saved frame with render_boxes(), builds the embed with create_mock_alert(),
    and sends both to the alerts channel.
    Raises if the alert can't be sent, so listen() leaves the message unacknowledged and it can be retried.
    """
    
    # draw the bounding boxes on the saved frame and save it to a buffer
    buffer = io.BytesIO()
    render_boxes(event).save(buffer, "PNG")
    buffer.seek(0)
    
    # wrap the buffer in a discord.File object
    file = discord.File(fp=buffer, filename="image.png")
    
    # create the embed message for the alert
    embed = create_mock_alert(event)
    
    # send the alert to the Discord channel with the embed message and the image as an attachment
    webhook.send(embed=embed, file=file)

def listen(redis_client, webhook):
    """
    Read new events from the stream forever as the Discord consumer group.
    Blocks while waiting for new messages, decodes each event from JSON, sends an alert with send_alert()
    unless the event is inside the cooldown, then acknowledges the message.
    If send_alert() raises, the message is not acknowledged and stays pending for a retry.
    """

    # catch up on pending messages
    last_id = "0"

    # video time of the last alert sent, None until the first alert
    last_alert_time = None
    
    # loop forever, reading new messages from the stream
    while True:
        messages = redis_client.xreadgroup(
            GROUP_NAME, 
            CONSUMER_NAME,
            {REDIS_STREAM_NAME: last_id}, 
            count=10, 
            block=5000,
        )

        if last_id == "0" and not any(entries for _, entries in messages):
            # no pending messages left, switch to new ones
            last_id = ">"

        for _, entries in messages:
            for message_id, message_data in entries:
                # decode the event from JSON
                event = json.loads(message_data["event"])

                # send an alert unless we're still inside the cooldown of the last one
                if should_alert(event, last_alert_time):
                    send_alert(event, webhook)
                    last_alert_time = event["time"]

                # mark the event as handled, whether it was alerted or skipped by the cooldown
                redis_client.xack(REDIS_STREAM_NAME, GROUP_NAME, message_id)

if __name__ == "__main__":
    # connect to redis, create the consumer group, then listen for detection events
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    create_group(redis_client)
    
    # load the discord webhook from the .env file
    load_dotenv()
    webhook = discord.SyncWebhook.from_url(os.environ["DISCORD_WEBHOOK_URL"])
    
    listen(redis_client, webhook)
