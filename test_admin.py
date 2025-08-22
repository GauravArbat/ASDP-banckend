#!/usr/bin/env python3
"""
Test script for admin functionality
"""

import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def test_admin_functionality():
    """Test admin functionality"""
    print("🧪 Testing Admin Functionality")
    print("=" * 50)
    
    session = requests.Session()
    
    # Test 1: Try to access admin dashboard without login
    print("\n1. Testing admin access without login...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/dashboard")
        if response.status_code == 401 or response.status_code == 403:
            print("✅ Correctly denied access without login")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Login as admin
    print("\n2. Logging in as admin...")
    try:
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = session.post(f"{BASE_URL}/api/auth/login", json=login_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Admin login successful")
                print(f"   User: {data['user']['username']}")
                print(f"   Role: {data['user']['role']}")
            else:
                print(f"❌ Login failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Login failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return False
    
    # Test 3: Access admin dashboard
    print("\n3. Testing admin dashboard access...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/dashboard")
        if response.status_code == 200:
            data = response.json()
            print("✅ Admin dashboard accessible")
            print(f"   Users: {data['stats']['users']}")
            print(f"   Datasets: {data['stats']['datasets']}")
            print(f"   Runs: {data['stats']['runs']}")
            print(f"   Reports: {data['stats']['reports']}")
        else:
            print(f"❌ Dashboard access failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Dashboard error: {str(e)}")
        return False
    
    # Test 4: Test user list
    print("\n4. Testing user list...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/dashboard")
        data = response.json()
        users = data.get('users', [])
        print(f"✅ Found {len(users)} users")
        for user in users:
            print(f"   - {user['username']} ({user['role']})")
    except Exception as e:
        print(f"❌ User list error: {str(e)}")
    
    # Test 5: Test role update (if there are other users)
    print("\n5. Testing role update...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/dashboard")
        data = response.json()
        users = data.get('users', [])
        
        # Find a non-admin user to test with
        test_user = None
        for user in users:
            if user['role'] == 'user':
                test_user = user
                break
        
        if test_user:
            print(f"   Testing with user: {test_user['username']}")
            role_data = {"role": "admin"}
            response = session.post(f"{BASE_URL}/api/admin/user/{test_user['id']}/role", json=role_data)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Role update successful: {data.get('message')}")
                
                # Change back to user
                role_data = {"role": "user"}
                response = session.post(f"{BASE_URL}/api/admin/user/{test_user['id']}/role", json=role_data)
                if response.status_code == 200:
                    print("✅ Role reverted back to user")
                else:
                    print(f"❌ Role revert failed: {response.status_code}")
            else:
                print(f"❌ Role update failed: {response.status_code}")
                print(f"   Response: {response.text}")
        else:
            print("   No non-admin users found to test with")
    except Exception as e:
        print(f"❌ Role update error: {str(e)}")
    
    print("\n🎉 Admin functionality tests completed!")
    return True

if __name__ == "__main__":
    try:
        success = test_admin_functionality()
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script error: {str(e)}")
        sys.exit(1)
