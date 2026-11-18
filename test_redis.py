import os
import redis
import certifi
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("REDIS_URL")

try:
    client = redis.Redis.from_url(
        url,
        decode_responses=True,
        ssl_ca_certs=certifi.where(),
        socket_connect_timeout=15,
        socket_timeout=15
    )

    print("PING:", client.ping())
    print("REDIS CONNECTED SUCCESSFULLY!")

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR DETAILS:", str(e))