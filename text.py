import requests
import json

payload = {
    "data": {"apple": 1}
}
print(payload)

# Convert payload to JSON format
# json_payload = json.dumps(payload)

response = requests.post("https://gemini.up.railway.app/api/gemini/realtimeupdate", json=payload)
print(response)

# Check for successful response (may not always be 200)
if response.status_code == 200:
    # Access the response data (assuming JSON format)
    data = response.json()
    print(response.status_code)
else:
    print(response.status_code)
