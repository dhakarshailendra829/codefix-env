import requests
import json

print("Testing /reset endpoint...")
response = requests.post(
    "http://127.0.0.1:8000/reset", json={}, headers={"Content-Type": "application/json"}
)
print("Status Code:", response.status_code)
if response.status_code == 200:
    data = response.json()
    print("Success! Reset response received.")
    print(json.dumps(data, indent=2))
else:
    print("Failed. Response text:")
    print(response.text)
