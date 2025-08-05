#!/usr/bin/env python3
"""
Simple Streamlit test for VLM loading
"""
import streamlit as st
import os

st.set_page_config(page_title="VLM Test", page_icon="🤖")

st.title("🤖 VLM Loading Test")

# Get token
hf_token = None
if os.path.exists("hf_token.txt"):
    with open("hf_token.txt", "r") as f:
        hf_token = f.read().strip()
    st.success(f"✅ Token loaded: {hf_token[:10]}...")
else:
    st.error("❌ No token file found")

if hf_token:
    try:
        with st.spinner("🔄 Loading VLM..."):
            from escalator_vlm_analyzer import EscalatorVLMAnalyzer
            analyzer = EscalatorVLMAnalyzer(hf_token=hf_token)
        
        st.info(f"VLM Available: {analyzer.vlm_available}")
        st.info(f"VLM Processor: {analyzer.vlm_processor is not None}")
        st.info(f"VLM Model: {analyzer.vlm_model is not None}")
        
        if analyzer.vlm_available:
            st.success("✅ VLM is working in Streamlit!")
        else:
            st.error("❌ VLM is not available")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.warning("Please ensure hf_token.txt exists with your Hugging Face token") 