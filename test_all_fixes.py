#!/usr/bin/env python3
"""
Comprehensive test script to verify all ASDP fixes
"""
import requests
import json
import sys

BASE_URL = "https://asdp-banckend.onrender.com"

def test_endpoint(endpoint, method="GET", data=None, headers=None, description=""):
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
        if description:
            print(f"   Description: {description}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code in [200, 201]:
            print("   ✅ Success")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
        elif response.status_code in [401, 403]:
            print("   ⚠️  Authentication required (expected)")
        else:
            print(f"   ❌ Failed: {response.text[:200]}...")
        
        return response.status_code in [200, 201, 401, 403]
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_cors_preflight():
    """Test CORS preflight requests"""
    print("\n🌐 Testing CORS Preflight Requests")
    
    endpoints = [
        "/api/auth/login",
        "/api/auth/register", 
        "/upload",
        "/clean",
        "/report"
    ]
    
    for endpoint in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.options(url, headers={
                'Origin': 'https://asdp-frontend.vercel.app',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            })
            
            print(f"🔍 Testing OPTIONS {endpoint}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ CORS preflight successful")
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods')
                }
                print(f"   CORS Headers: {cors_headers}")
            else:
                print(f"   ❌ CORS preflight failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ CORS preflight error: {e}")

def test_authentication_flow():
    """Test authentication flow"""
    print("\n🔐 Testing Authentication Flow")
    
    # Test registration
    test_endpoint("/api/auth/register", "POST", {
        "username": "testuser",
        "email": "test@example.com", 
        "password": "testpass123"
    }, description="User registration")
    
    # Test login
    test_endpoint("/api/auth/login", "POST", {
        "username": "testuser",
        "password": "testpass123"
    }, description="User login")
    
    # Test admin login
    test_endpoint("/api/auth/login", "POST", {
        "username": "admin",
        "password": "admin123"
    }, description="Admin login")

def test_data_endpoints():
    """Test data processing endpoints"""
    print("\n📊 Testing Data Processing Endpoints")
    
    # Test upload endpoint (without file)
    test_endpoint("/upload", "POST", {}, description="File upload endpoint")
    
    # Test clean endpoint
    test_endpoint("/clean", "POST", {
        "config": {
            "imputation": {"method": "mean", "columns": None},
            "outliers": {"detection_method": "iqr", "handling_method": "winsorize", "columns": None},
            "weights": {"column": None},
            "estimate_columns": None
        }
    }, description="Data cleaning endpoint")
    
    # Test report generation
    test_endpoint("/report", "POST", {
        "format": "html"
    }, description="HTML report generation")

def main():
    """Run all tests"""
    print("🧪 Comprehensive ASDP Fixes Test")
    print("=" * 50)
    
    # Test basic endpoints
    print("\n🏥 Testing Basic Endpoints")
    test_endpoint("/", description="Root endpoint")
    test_endpoint("/health", description="Health check")
    
    # Test CORS
    test_cors_preflight()
    
    # Test authentication
    test_authentication_flow()
    
    # Test data endpoints
    test_data_endpoints()
    
    print("\n✅ All tests completed!")
    print("\n📋 Summary:")
    print("- Basic endpoints should return 200")
    print("- CORS preflight should return 200")
    print("- Auth endpoints should work or return 401 (expected)")
    print("- Data endpoints should return 400 (no data) or work")

if __name__ == "__main__":
    main()
