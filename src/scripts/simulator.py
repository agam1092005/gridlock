import time
import requests
import random
import json
import uuid
import csv
import os
from datetime import datetime, timezone

# Configuration
API_URL = "http://localhost:8000/api/incidents/"
API_KEY = "test-key-12345"  # Valid key
DATASET_PATH = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"


def load_dataset():
    """Load the CSV dataset into a list of dictionaries."""
    rows = []
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Falling back to random mode.")
        return None

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Ensure lat/lon are valid floats
                float(row["latitude"])
                float(row["longitude"])
                rows.append(row)
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(rows)} valid records from dataset.")
    return rows


def generate_random_incident(dataset_rows=None):
    """Generate an incident from the dataset or randomly."""
    if dataset_rows:
        row = random.choice(dataset_rows)
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # Parse incident type and description
        cause = row.get("event_cause", "unknown")
        desc = row.get("description", "")
        if not desc or desc == "NULL":
            desc = f"Reported {cause} at {row.get('address', 'unknown location')}."

        incident_type = (
            cause
            if cause in ["accident", "congestion", "road_closure", "hazard", "event"]
            else "event"
        )

        # Priority mapping to severity
        priority = row.get("priority", "Low").lower()
        if priority == "high":
            severity = random.randint(70, 100)
            duration = random.randint(120, 240)
        elif priority == "medium":
            severity = random.randint(40, 69)
            duration = random.randint(60, 119)
        else:
            severity = random.randint(10, 39)
            duration = random.randint(15, 59)

        description = desc if len(desc) >= 10 else f"{desc} (padded desc)"
        metadata_dict = row
    else:
        # Fallback SF generation
        incident_types = ["accident", "congestion", "road_closure", "hazard", "event"]
        lat = random.uniform(37.70, 37.81)
        lon = random.uniform(-122.51, -122.38)
        severity = random.randint(10, 100)
        duration = random.randint(15, 180)
        incident_type = random.choice(incident_types)
        description = f"Simulated {incident_type} reported at coordinates {lat:.4f}, {lon:.4f}. Estimated delay: {duration} mins."
        metadata_dict = {}

    return {
        "incident_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": description,
        "location": {"latitude": lat, "longitude": lon},
        "incident_type": incident_type,
        "metadata": metadata_dict,
    }


def run_simulator(interval_seconds=3.0):
    print(f"Starting Gridlock 2.0 Incident Simulator...")
    print(f"Targeting API: {API_URL}")
    print(f"Interval: 1 incident every {interval_seconds} seconds")
    print("-" * 50)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    dataset_rows = load_dataset()

    while True:
        incident = generate_random_incident(dataset_rows)

        try:
            response = requests.post(API_URL, json=incident, headers=headers, timeout=5.0)

            if response.status_code == 202:
                print(
                    f"[+] Successfully generated incident: {incident['incident_id']} (Type: {incident['incident_type']})"
                )
            else:
                print(f"[-] Failed to submit incident: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"[!] Connection error: {e}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    try:
        run_simulator(interval_seconds=10.0)
    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")
