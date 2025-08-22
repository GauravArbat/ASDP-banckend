#!/usr/bin/env python3
"""
Simple database fix script
"""

import sqlite3
import os

def fix_database():
    """Fix the database schema"""
    db_path = "instance/app.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return False
    
    print("🔧 Fixing database schema...")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if report_record table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='report_record'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check if run_id column exists
            cursor.execute("PRAGMA table_info(report_record)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'run_id' not in columns:
                print("❌ Missing run_id column in report_record table")
                print("🔄 Dropping and recreating report_record table...")
                
                # Drop the table
                cursor.execute("DROP TABLE report_record")
                print("✅ Dropped report_record table")
                
                # Create new table with correct schema
                cursor.execute("""
                    CREATE TABLE report_record (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id INTEGER,
                        user_id INTEGER,
                        report_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print("✅ Created report_record table with correct schema")
            else:
                print("✅ report_record table has correct schema")
        else:
            print("❌ report_record table doesn't exist")
            print("🔄 Creating report_record table...")
            
            cursor.execute("""
                CREATE TABLE report_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    user_id INTEGER,
                    report_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ Created report_record table")
        
        # Check other tables
        tables = ['user', 'dataset', 'processing_run']
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                print(f"✅ {table} table exists")
            else:
                print(f"❌ {table} table missing")
        
        # Commit changes
        conn.commit()
        print("\n🎉 Database schema fixed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {str(e)}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🔧 ASDP Database Schema Fix")
    print("=" * 40)
    
    try:
        success = fix_database()
        if success:
            print("\n✅ Database fixed successfully!")
        else:
            print("\n❌ Failed to fix database!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
