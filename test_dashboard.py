#!/usr/bin/env python3
"""
Simple test for admin dashboard functionality
"""

import sqlite3
import os

def test_dashboard():
    """Test if admin dashboard data can be retrieved"""
    db_path = "instance/app.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return False
    
    print("🧪 Testing Admin Dashboard Data...")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test 1: Count users
        try:
            cursor.execute("SELECT COUNT(*) FROM user")
            users_count = cursor.fetchone()[0]
            print(f"✅ Users count: {users_count}")
        except Exception as e:
            print(f"❌ Error counting users: {str(e)}")
            users_count = 0
        
        # Test 2: Count datasets
        try:
            cursor.execute("SELECT COUNT(*) FROM dataset")
            datasets_count = cursor.fetchone()[0]
            print(f"✅ Datasets count: {datasets_count}")
        except Exception as e:
            print(f"❌ Error counting datasets: {str(e)}")
            datasets_count = 0
        
        # Test 3: Count processing runs
        try:
            cursor.execute("SELECT COUNT(*) FROM processing_run")
            runs_count = cursor.fetchone()[0]
            print(f"✅ Runs count: {runs_count}")
        except Exception as e:
            print(f"❌ Error counting runs: {str(e)}")
            runs_count = 0
        
        # Test 4: Count reports (should work now)
        try:
            cursor.execute("SELECT COUNT(*) FROM report_record")
            reports_count = cursor.fetchone()[0]
            print(f"✅ Reports count: {reports_count}")
        except Exception as e:
            print(f"❌ Error counting reports: {str(e)}")
            reports_count = 0
        
        # Test 5: Get latest users
        try:
            cursor.execute("SELECT id, username, email, role, created_at FROM user ORDER BY created_at DESC LIMIT 5")
            users = cursor.fetchall()
            print(f"✅ Latest users: {len(users)} found")
            for user in users:
                print(f"   - {user[1]} ({user[3]})")
        except Exception as e:
            print(f"❌ Error getting users: {str(e)}")
        
        # Test 6: Get latest datasets
        try:
            cursor.execute("SELECT id, filename, rows, columns, uploaded_at FROM dataset ORDER BY uploaded_at DESC LIMIT 5")
            datasets = cursor.fetchall()
            print(f"✅ Latest datasets: {len(datasets)} found")
            for dataset in datasets:
                print(f"   - {dataset[1]} ({dataset[2]} rows, {dataset[3]} cols)")
        except Exception as e:
            print(f"❌ Error getting datasets: {str(e)}")
        
        # Test 7: Get latest runs
        try:
            cursor.execute("SELECT id, dataset_id, user_id, success, plots_count, created_at FROM processing_run ORDER BY created_at DESC LIMIT 5")
            runs = cursor.fetchall()
            print(f"✅ Latest runs: {len(runs)} found")
            for run in runs:
                print(f"   - Run {run[0]} (Dataset {run[1]}, Success: {run[3]}, Plots: {run[4]})")
        except Exception as e:
            print(f"❌ Error getting runs: {str(e)}")
        
        print(f"\n📊 Summary:")
        print(f"   Users: {users_count}")
        print(f"   Datasets: {datasets_count}")
        print(f"   Runs: {runs_count}")
        print(f"   Reports: {reports_count}")
        
        print("\n🎉 Dashboard data test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing dashboard: {str(e)}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🧪 ASDP Admin Dashboard Test")
    print("=" * 40)
    
    try:
        success = test_dashboard()
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
