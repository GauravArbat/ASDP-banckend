#!/usr/bin/env python3
"""
ASDP (AI Survey Data Processor) Application
Ministry of Statistics and Programme Implementation (MoSPI)

Startup script for the application
"""

import os
import sys
import webbrowser
import threading
import time
from app import app

def open_browser():
    """Open the application in the default web browser after a short delay"""
    time.sleep(2)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

def main():
    """Main function to start the application"""
    print("=" * 60)
    print("ASDP (AI Survey Data Processor) Application")
    print("Ministry of Statistics and Programme Implementation (MoSPI)")
    print("=" * 60)
    print()
    print("Starting the application...")
    print("• Web interface will be available at: http://localhost:5000")
    print("• Press Ctrl+C to stop the application")
    print()
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Run the Flask application
        app.run(
            debug=True,  # Enable auto-reload for development
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Application stopped by user")
        print("Thank you for using the ASDP (AI Survey Data Processor) Application!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
