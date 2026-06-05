import requests
import time
import sys

url = "http://localhost:8000/predict"
# Passing a "High Risk" crude batch (Heavy API, High BSW, High Salt)
data = {
    "Crude_Blend": "Basrah Heavy",
    "API_Gravity": 21.0,
    "Inlet_BSW": 2.2,
    "Inlet_Salt_PTB": 58.0
}

print(f"Sending POST request to {url} with data:")
print(data)

# Retry loop in case server takes a moment to spin up
for i in range(15):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print("\n--- API RESPONSE ---")
        print(response.json())
        sys.exit(0)
    except requests.exceptions.ConnectionError:
        time.sleep(1)
        
print("Error: Could not connect to the API.")
sys.exit(1)
