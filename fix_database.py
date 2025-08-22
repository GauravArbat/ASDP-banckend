#!/usr/bin/env python3
"""
Script to fix database schema issues
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User, Dataset, ProcessingRun, ReportRecord

def fix_database():
    """Fix database schema issues"""
    with app.app_context():
        print("🔧 Fixing Database Schema...")
        print("=" * 40)
        
        # Check if tables exist and have correct columns
        try:
            # Check if report_record table exists and has run_id column
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'report_record' in tables:
                columns = [col['name'] for col in inspector.get_columns('report_record')]
                print(f"Report record table columns: {columns}")
                
                if 'run_id' not in columns:
                    print("❌ Missing run_id column in report_record table")
                    print("🔄 Recreating report_record table...")
                    
                    # Drop the existing table
                    db.engine.execute('DROP TABLE IF EXISTS report_record')
                    print("✅ Dropped existing report_record table")
                    
                    # Recreate the table with correct schema
                    db.create_all()
                    print("✅ Recreated report_record table with correct schema")
                else:
                    print("✅ report_record table has correct schema")
            else:
                print("❌ report_record table doesn't exist")
                print("🔄 Creating all tables...")
                db.create_all()
                print("✅ Created all tables")
            
            # Check other tables
            for table_name in ['user', 'dataset', 'processing_run']:
                if table_name in tables:
                    columns = [col['name'] for col in inspector.get_columns(table_name)]
                    print(f"✅ {table_name} table: {columns}")
                else:
                    print(f"❌ {table_name} table missing")
            
            # Verify admin user exists
            admin_count = User.query.filter_by(role='admin').count()
            print(f"✅ Admin users: {admin_count}")
            
            if admin_count == 0:
                print("⚠️  No admin users found. Creating default admin...")
                admin_user = User(
                    username="admin",
                    email="admin@asdp.gov.in",
                    role="admin",
                    created_at=datetime.utcnow()
                )
                admin_user.set_password("admin123")
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Created default admin user")
            
            print("\n🎉 Database schema fixed successfully!")
            
        except Exception as e:
            print(f"❌ Error fixing database: {str(e)}")
            print("🔄 Attempting to recreate all tables...")
            
            try:
                # Drop all tables and recreate
                db.drop_all()
                db.create_all()
                print("✅ Recreated all tables")
                
                # Create admin user
                admin_user = User(
                    username="admin",
                    email="admin@asdp.gov.in",
                    role="admin",
                    created_at=datetime.utcnow()
                )
                admin_user.set_password("admin123")
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Created default admin user")
                
            except Exception as e2:
                print(f"❌ Failed to recreate tables: {str(e2)}")
                return False
        
        return True

if __name__ == "__main__":
    print("🔧 ASDP Database Fix Script")
    print("=" * 40)
    
    try:
        success = fix_database()
        if success:
            print("\n✅ Database fixed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Failed to fix database!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Database fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Database fix error: {str(e)}")
        sys.exit(1)
