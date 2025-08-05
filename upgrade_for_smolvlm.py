#!/usr/bin/env python3
"""
Upgrade Transformers for SmolVLM
Script to upgrade transformers to a version compatible with SmolVLM models
"""

import subprocess
import sys

def upgrade_transformers():
    """Upgrade transformers to a version compatible with SmolVLM"""
    print("🔄 Upgrading transformers for SmolVLM compatibility...")
    print("=" * 50)
    
    try:
        # Check current version
        result = subprocess.run([sys.executable, "-m", "pip", "show", "transformers"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    current_version = line.split(':')[1].strip()
                    print(f"📦 Current transformers version: {current_version}")
                    break
        
        # Upgrade transformers
        print("🔄 Installing transformers >= 4.37.0...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.37.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Transformers upgraded successfully!")
            print("🎉 You can now use actual SmolVLM models!")
            print("\n💡 Next steps:")
            print("1. Restart your Python environment")
            print("2. Run your Streamlit app again")
            print("3. SmolVLM should now load the actual models instead of fallbacks")
        else:
            print("❌ Failed to upgrade transformers:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error during upgrade: {e}")

if __name__ == "__main__":
    upgrade_transformers() 