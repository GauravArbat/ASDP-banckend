#!/usr/bin/env python3
"""
API endpoint test script
"""
import requests
import json

BASE_URL = "https://asdp-banckend.onrender.com"

def test_endpoint(endpoint, method="GET", data=None, headers=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    if headers is None:
        headers = {}
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"🔍 Testing {method} {endpoint}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Success")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
        else:
            print(f"   ❌ Failed: {response.text[:200]}...")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Test all API endpoints"""
    print("🧪 Testing ASDP API Endpoints\n")
    
    # Test basic endpoints
    test_endpoint("/")
    test_endpoint("/health")
    
    # Test auth endpoints (without credentials)
    test_endpoint("/api/auth/me")
    
    # Test admin endpoint (should fail without auth)
    test_endpoint("/api/admin/dashboard")
    
    # Test data endpoints (should fail without auth)
    test_endpoint("/upload", method="POST", data={"test": "data"})
    
    print("\n✅ API testing completed")

if __name__ == "__main__":
    main()
