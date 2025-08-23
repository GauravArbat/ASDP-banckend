#!/usr/bin/env python3
"""
Database check and admin user creation script
"""
import os
import sys
from app import app, db, User
from werkzeug.security import generate_password_hash

def check_database():
    """Check database status and create admin user if needed"""
    with app.app_context():
        try:
            # Check if database exists and can be accessed
            db.session.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            
            # Check if tables exist
            db.create_all()
            print("✅ Database tables created/verified")
            
            # Check for admin users
            admin_users = User.query.filter_by(role='admin').all()
            if admin_users:
                print(f"✅ Found {len(admin_users)} admin user(s):")
                for user in admin_users:
                    print(f"   - {user.username} ({user.email or 'no email'})")
            else:
                print("⚠️  No admin users found. Creating default admin user...")
                admin_user = User(
                    username='admin',
                    email='admin@asdp.gov.in',
                    role='admin'
                )
                admin_user.set_password('admin123')
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Created admin user: admin/admin123")
            
            # Check total users
            total_users = User.query.count()
            print(f"✅ Total users in database: {total_users}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    
    return True

if __name__ == "__main__":
    from sqlalchemy import text
    success = check_database()
    sys.exit(0 if success else 1)
