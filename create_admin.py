#!/usr/bin/env python3
"""
Script to create an admin user for the ASDP application.
Run this script to create the first admin user if none exists.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User

def create_admin_user():
    """Create an admin user if none exists"""
    with app.app_context():
        # Check if any admin users exist
        admin_users = User.query.filter_by(role='admin').count()
        
        if admin_users > 0:
            print("✅ Admin users already exist!")
            print(f"Found {admin_users} admin user(s)")
            return
        
        # Create default admin user
        admin_username = "admin"
        admin_email = "admin@asdp.gov.in"
        admin_password = "admin123"  # Change this in production!
        
        # Check if admin user already exists
        existing_user = User.query.filter_by(username=admin_username).first()
        if existing_user:
            print(f"✅ User '{admin_username}' already exists!")
            if existing_user.role != 'admin':
                existing_user.role = 'admin'
                db.session.commit()
                print(f"✅ Updated user '{admin_username}' to admin role")
            return
        
        # Create new admin user
        admin_user = User(
            username=admin_username,
            email=admin_email,
            role='admin',
            created_at=datetime.utcnow()
        )
        admin_user.set_password(admin_password)
        
        db.session.add(admin_user)
        db.session.commit()
        
        print("✅ Admin user created successfully!")
        print(f"Username: {admin_username}")
        print(f"Email: {admin_email}")
        print(f"Password: {admin_password}")
        print("\n⚠️  IMPORTANT: Change the password after first login!")
        print("You can change it in the profile page.")

if __name__ == "__main__":
    print("🔧 ASDP Admin User Setup")
    print("=" * 40)
    
    try:
        create_admin_user()
    except Exception as e:
        print(f"❌ Error creating admin user: {str(e)}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
