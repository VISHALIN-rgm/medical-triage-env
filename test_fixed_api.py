import requests
import json
import time

base_url = "http://localhost:8000"

print("=" * 50)
print("Testing Medical Triage Environment")
print("=" * 50)

# Give server time to start
time.sleep(1)

# 1. Health check
print("\n1. Health Check:")
try:
    response = requests.get(f"{base_url}/health", timeout=5)
    print(f"   Status: {response.json()}")
except Exception as e:
    print(f"   ✗ Server not running! Start the server first.")
    print(f"   Error: {e}")
    exit(1)

# 2. Reset environment
print("\n2. Reset Environment (Easy Task):")
reset_response = requests.post(
    f"{base_url}/reset",
    json={"task_id": "easy"}
)

if reset_response.status_code == 200:
    data = reset_response.json()
    print(f"   ✓ Reset successful")
    print(f"   Task: {data['observation']['task_id']}")
    print(f"   Patients: {len(data['observation']['patients'])}")
    
    for patient in data['observation']['patients']:
        print(f"   - {patient['id']}: {patient['name']}, Urgency: {patient['urgency_score']}")
else:
    print(f"   ✗ Reset failed: {reset_response.status_code}")
    exit(1)

# 3. Process patients with lowercase actions
print("\n3. Processing Patients:")

# Step 1: Escalate P1 (critical)
print("\n   Step 1: P1 (Critical Patient)")
step1 = requests.post(
    f"{base_url}/step",
    json={"action": {"type": "escalate", "patient_id": "P1"}}
)

if step1.status_code == 200:
    data = step1.json()
    print(f"   ✓ Action: escalate")
    print(f"     Reward: {data['reward']}")
    print(f"     Expected: {data['info'].get('expected_action', 'N/A')}")
else:
    print(f"   ✗ Failed: {step1.text}")

# Step 2: Discharge P2 (stable)
print("\n   Step 2: P2 (Stable Patient)")
step2 = requests.post(
    f"{base_url}/step",
    json={"action": {"type": "discharge", "patient_id": "P2"}}
)

if step2.status_code == 200:
    data = step2.json()
    print(f"   ✓ Action: discharge")
    print(f"     Reward: {data['reward']}")
    print(f"     Expected: {data['info'].get('expected_action', 'N/A')}")
else:
    print(f"   ✗ Failed: {step2.text}")

# Step 3: Discharge P3 (stable)
print("\n   Step 3: P3 (Stable Patient)")
step3 = requests.post(
    f"{base_url}/step",
    json={"action": {"type": "discharge", "patient_id": "P3"}}
)

if step3.status_code == 200:
    data = step3.json()
    print(f"   ✓ Action: discharge")
    print(f"     Reward: {data['reward']}")
    print(f"     Expected: {data['info'].get('expected_action', 'N/A')}")
    print(f"     Episode Complete: {data['done']}")
else:
    print(f"   ✗ Failed: {step3.text}")

# 4. Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

total_reward = 0
for i, resp in enumerate([step1, step2, step3], 1):
    if resp.status_code == 200:
        reward = resp.json()['reward']
        total_reward += reward
        print(f"Step {i}: {reward}")

print(f"Total Reward: {total_reward:.2f}")
print(f"Expected Max: 30.0 (3 patients * 10.0)")

if total_reward >= 29.0:
    print("\n🎉 PERFECT! Your environment is working correctly!")
elif total_reward > 0:
    print("\n⚠️  Partial success. Check individual rewards above.")
else:
    print("\n❌ Test failed. Check server logs.")