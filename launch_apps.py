#!/usr/bin/env python3
"""
App Launcher for Thermal Image and Escalator Safety Analysis
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 80)
    print("🔥 THERMAL IMAGE & ESCALATOR SAFETY ANALYSIS SUITE 🔥")
    print("=" * 80)
    print("📱 Available Applications:")
    print("  1. 🔥 Thermal Image Analyzer (Original)")
    print("  2. 🤖 VLM-Enhanced Escalator Safety Monitor")
    print("=" * 80)

def check_dependencies():
    """Check if required files exist"""
    apps = {
        "Thermal Image": "streamlit_app.py",
        "VLM Escalator": "streamlit_escalator_vlm.py"
    }
    
    missing = []
    for name, file in apps.items():
        if not os.path.exists(file):
            missing.append(f"{name} ({file})")
    
    if missing:
        print("❌ Missing application files:")
        for app in missing:
            print(f"   - {app}")
        return False
    
    print("✅ All application files found!")
    return True

def launch_app(app_name, app_file, port=None):
    """Launch a Streamlit app"""
    print(f"\n🚀 Launching {app_name}...")
    print(f"📁 App file: {app_file}")
    
    if port:
        print(f"🌐 Port: {port}")
        cmd = [sys.executable, "-m", "streamlit", "run", app_file, "--server.port", str(port)]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", app_file]
    
    try:
        print(f"⚡ Command: {' '.join(cmd)}")
        print("⏳ Starting app (this may take a few seconds)...")
        
        # Launch in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for app to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ {app_name} started successfully!")
            print(f"🌐 Check your browser for the app")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {app_name}")
            print(f"Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching {app_name}: {e}")
        return None

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please ensure all required files are present before launching apps.")
        return
    
    while True:
        print("\n🎯 Choose an app to launch:")
        print("  1. 🔥 Thermal Image Analyzer (Port 8501)")
        print("  2. 🤖 VLM-Enhanced Escalator Safety (Port 8502)")
        print("  0. ❌ Exit")
        
        try:
            choice = input("\n📝 Enter your choice (0-2): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                launch_app("Thermal Image Analyzer", "streamlit_app.py", 8501)
            elif choice == "2":
                launch_app("VLM-Enhanced Escalator Safety", "streamlit_escalator_vlm.py", 8502)
            else:
                print("❌ Invalid choice. Please enter a number between 0-2.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 