#!/usr/bin/env python3
"""
Simple test script to verify API endpoints are working
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_root():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_api_docs():
    """Test the API documentation endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ API docs endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'endpoints' in data:
                print(f"   Available endpoints: {list(data['endpoints'].keys())}")
        return True
    except Exception as e:
        print(f"❌ API docs endpoint failed: {e}")
        return False

def test_legacy_routes():
    """Test legacy route compatibility"""
    legacy_routes = [
        "/me",
        "/login", 
        "/register",
        "/logout",
        "/profile",
        "/admin",
        "/admin/summary"
    ]
    
    for route in legacy_routes:
        try:
            response = requests.get(f"{BASE_URL}{route}")
            print(f"✅ Legacy {route}: {response.status_code}")
        except Exception as e:
            print(f"❌ Legacy {route} failed: {e}")

def test_api_routes():
    """Test new API routes"""
    api_routes = [
        "/api/auth/me",
        "/api/admin/dashboard",
    ]
    
    for route in api_routes:
        try:
            response = requests.get(f"{BASE_URL}{route}")
            print(f"✅ API {route}: {response.status_code}")
        except Exception as e:
            print(f"❌ API {route} failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing ASDP Backend API...")
    print("=" * 50)
    
    # Test basic endpoints
    test_root()
    test_api_docs()
    
    print("\n🔗 Testing Legacy Routes (for backward compatibility):")
    test_legacy_routes()
    
    print("\n🔗 Testing New API Routes:")
    test_api_routes()
    
    print("\n✅ API testing completed!")
    print("\nTo start the server, run: python app.py")
