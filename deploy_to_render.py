#!/usr/bin/env python3
"""
Deployment Helper Script for Render
This script helps prepare the ASDP backend for deployment on Render
"""

import os
import shutil
import subprocess
import sys

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'render.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def check_git_status():
    """Check git status and suggest next steps"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("📝 You have uncommitted changes:")
            print(result.stdout)
            print("\n💡 Commit your changes before deploying:")
            print("   git add .")
            print("   git commit -m 'Prepare for Render deployment'")
            print("   git push origin main")
        else:
            print("✅ Working directory is clean")
            
    except subprocess.CalledProcessError:
        print("⚠️  Git not available or not a git repository")
    except FileNotFoundError:
        print("⚠️  Git not installed")

def create_deployment_checklist():
    """Create a deployment checklist"""
    print("\n📋 Render Deployment Checklist:")
    print("=" * 40)
    print("1. ✅ Code is committed and pushed to GitHub")
    print("2. ✅ render.yaml is configured for free tier")
    print("3. ✅ Dockerfile is optimized")
    print("4. ✅ All dependencies are in requirements.txt")
    print("5. 🔄 Deploy on Render dashboard")
    print("6. 🔄 Test the deployed application")
    print("7. 🔄 Change default admin password")

def main():
    print("🚀 ASDP Backend - Render Deployment Helper")
    print("=" * 50)
    
    if not check_requirements():
        sys.exit(1)
    
    check_git_status()
    create_deployment_checklist()
    
    print("\n🎯 Next Steps:")
    print("1. Push your code to GitHub")
    print("2. Go to https://dashboard.render.com/")
    print("3. Create a new Web Service")
    print("4. Connect your repository: GauravArbat/ASDP-banckend")
    print("5. Render will auto-detect your configuration")
    print("6. Click 'Create Web Service'")
    
    print("\n📚 For detailed instructions, see: RENDER_DEPLOYMENT.md")

if __name__ == "__main__":
    main()
