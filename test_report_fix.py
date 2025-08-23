#!/usr/bin/env python3
"""
Test script to verify report generation fix
"""
import requests
import json

BASE_URL = "https://asdp-banckend.onrender.com"

def test_report_generation():
    """Test report generation endpoints"""
    print("🧪 Testing Report Generation Fix")
    print("=" * 40)
    
    # Test report generation without data (should give proper error)
    print("\n📄 Testing PDF Report Generation (no data)")
    try:
        response = requests.post(f"{BASE_URL}/report", 
                               json={"format": "pdf"},
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        if response.status_code == 400:
            data = response.json()
            print(f"✅ Expected error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test HTML report generation without data
    print("\n🌐 Testing HTML Report Generation (no data)")
    try:
        response = requests.post(f"{BASE_URL}/report", 
                               json={"format": "html"},
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        if response.status_code == 400:
            data = response.json()
            print(f"✅ Expected error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test download data without data
    print("\n📥 Testing Data Download (no data)")
    try:
        response = requests.post(f"{BASE_URL}/download_data", 
                               json={},
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        if response.status_code == 400:
            data = response.json()
            print(f"✅ Expected error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test CORS preflight for report endpoints
    print("\n🌐 Testing CORS Preflight for Report Endpoints")
    endpoints = ["/report", "/download_data"]
    
    for endpoint in endpoints:
        try:
            response = requests.options(f"{BASE_URL}{endpoint}", 
                                      headers={
                                          'Origin': 'https://asdp-frontend.vercel.app',
                                          'Access-Control-Request-Method': 'POST',
                                          'Access-Control-Request-Headers': 'Content-Type'
                                      })
            
            print(f"OPTIONS {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ CORS preflight successful")
            else:
                print(f"   ❌ CORS preflight failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ CORS preflight error: {e}")

if __name__ == "__main__":
    test_report_generation()
    print("\n✅ Report generation fix test completed!")
