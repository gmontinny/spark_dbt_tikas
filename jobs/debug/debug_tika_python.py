import os
os.environ["TIKA_CLIENT_ONLY"] = "True"
os.environ["TIKA_SERVER_ENDPOINT"] = "http://tika-server:9998"

import tika
tika.TikaClientOnly = True

from tika import parser

result = parser.from_file(
    "/opt/spark/work-dir/datas/ipop-dezembro-2025.pdf",
    serverEndpoint="http://tika-server:9998",
    xmlContent=False,
    service="text",
)
print("STATUS:", result.get("status"))
print("ALL KEYS:", list(result.keys()))
print("CONTENT:", repr(result.get("content"))[:200])
print("TEXT:", repr(result.get("text"))[:200] if result.get("text") else "None")
print("FULL RESULT keys/values:")
for k, v in result.items():
    if k != "metadata":
        print(f"  {k!r}: {repr(v)[:200]}")
