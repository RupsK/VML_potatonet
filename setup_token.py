#!/usr/bin/env python3
"""
Setup Hugging Face Token
Script to set up the token for persistent use
"""

import os
import sys

def setup_token():
    """Set up the token for persistent use"""
    token = "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"
    
    print("🔧 Setting up Hugging Face Token for persistent use...")
    print("=" * 60)
    
    # Method 1: Set environment variable for current session
    os.environ["HF_TOKEN"] = token
    print("✅ Token set as environment variable for current session")
    
    # Method 2: Create .env file
    with open(".env", "w") as f:
        f.write(f"HF_TOKEN={token}\n")
    print("✅ Created .env file with token")
    
    # Method 3: Create a token file for the app
    with open("hf_token.txt", "w") as f:
        f.write(token)
    print("✅ Created hf_token.txt file")
    
    print("\n🎉 Token setup complete!")
    print("\n📋 Usage options:")
    print("1. Environment variable: HF_TOKEN is now set for this session")
    print("2. .env file: Load automatically with python-dotenv")
    print("3. hf_token.txt: Read directly by the app")
    
    return True

if __name__ == "__main__":
    setup_token() 