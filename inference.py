#!/usr/bin/env python3
"""
Simple inference script for OpenEnv validation
"""

import requests
import sys
import time

def main():
    """Main inference function"""
    print("=" * 50)
    print("Medical Triage Environment - Inference")
    print("=" * 50)
    
    # The automated system will set this URL
    env_url = "http://localhost:8000"
    
    print(f"Connecting to environment at {env_url}")
    
    # Wait for server to be ready
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get(f"{env_url}/health", timeout=5)
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        return 1
    
    try:
        # Reset environment
        print("\n🔄 Resetting environment...")
        reset_response = requests.post(
            f"{env_url}/reset",
            json={"task_id": "easy"},
            timeout=5
        )
        
        if reset_response.status_code != 200:
            print(f"❌ Reset failed: {reset_response.status_code}")
            return 1
            
        data = reset_response.json()
        patients = data['observation']['patients']
        print(f"✅ Reset successful. Got {len(patients)} patients.")
        
        # Process each patient
        total_reward = 0
        
        for patient in patients:
            # Decide action based on urgency
            if patient['urgency_score'] > 0.7:
                action = "escalate"
            elif patient['urgency_score'] > 0.3:
                action = "treat"
            else:
                action = "discharge"
            
            print(f"\n  Patient {patient['id']}: Urgency={patient['urgency_score']:.2f}")
            print(f"    Taking action: {action}")
            
            # Take step
            step_response = requests.post(
                f"{env_url}/step",
                json={"action": {"type": action, "patient_id": patient['id']}},
                timeout=5
            )
            
            if step_response.status_code != 200:
                print(f"    ❌ Step failed")
                return 1
                
            step_data = step_response.json()
            reward = step_data['reward']
            total_reward += reward
            print(f"    ✅ Reward: {reward:.2f}")
        
        print(f"\n{'='*50}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"{'='*50}")
        
        if total_reward > 0:
            print("\n✅ Inference completed successfully!")
            return 0
        else:
            print("\n❌ No rewards earned")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())