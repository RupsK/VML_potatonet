#!/usr/bin/env python3
"""
Minimal Streamlit app to test VLM loading
"""

import streamlit as st
import os

def test_vlm_in_streamlit():
    st.title("VLM Loading Test")
    
    # Check token
    hf_token = None
    if os.path.exists("hf_token.txt"):
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
            st.success(f"✅ Token loaded: {hf_token[:10]}...")
        except Exception as e:
            st.error(f"❌ Token file error: {e}")
            return
    else:
        st.error("❌ No token file found")
        return
    
    # Test imports
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        st.success("✅ Transformers and torch imported successfully")
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        return
    
    # Test processor loading
    try:
        with st.spinner("Loading processor..."):
            processor = AutoProcessor.from_pretrained(
                "microsoft/git-base",
                token=hf_token,
                cache_dir="./model_cache",
                trust_remote_code=True
            )
        st.success("✅ Processor loaded successfully")
    except Exception as e:
        st.error(f"❌ Processor loading failed: {e}")
        return
    
    # Test model loading
    try:
        with st.spinner("Loading model..."):
            model = AutoModelForVision2Seq.from_pretrained(
                "microsoft/git-base",
                token=hf_token,
                cache_dir="./model_cache",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return
    
    st.success("🎉 All VLM components loaded successfully in Streamlit!")

if __name__ == "__main__":
    test_vlm_in_streamlit() 