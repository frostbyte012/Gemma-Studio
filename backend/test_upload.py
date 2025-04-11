import os
import requests

# 1. Define paths
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')
TEST_FILE = os.path.join(TEST_DIR, 'test.txt')

# 2. Verify file exists
if not os.path.exists(TEST_FILE):
    print(f"Error: Test file not found at {TEST_FILE}")
    exit(1)

# 3. Prepare request
url = "http://localhost:8000/api/datasets/upload"
with open(TEST_FILE, 'rb') as f:
    # Include 'name' field as required by your API
    data = {'name': 'test_dataset'}  # Explicit name for the dataset
    files = {'file': (os.path.basename(TEST_FILE), f, 'text/plain')}  # Explicit MIME type
    response = requests.post(url, files=files, data=data)

# 4. Print results
print("\n=== Upload Test ===")
print(f"File: {TEST_FILE}")
print(f"Status: {response.status_code}")
print("Response:", response.json())