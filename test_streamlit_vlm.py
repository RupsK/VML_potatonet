import streamlit as st
import os
from escalator_vlm_analyzer import EscalatorVLMAnalyzer

st.title("VLM Loading Test")

# Check token
st.subheader("1. Token Check")
if os.path.exists("hf_token.txt"):
    try:
        with open("hf_token.txt", "r") as f:
            token = f.read().strip()
        st.success(f"✅ Token found: {token[:10]}...")
    except Exception as e:
        st.error(f"❌ Error reading token: {e}")
        token = None
else:
    st.error("❌ Token file not found")
    token = None

# Test VLM loading
if token:
    st.subheader("2. VLM Loading Test")
    try:
        with st.spinner("Loading VLM..."):
            analyzer = EscalatorVLMAnalyzer(hf_token=token)
        
        st.success("✅ Analyzer created!")
        st.info(f"VLM Available: {analyzer.vlm_available}")
        st.info(f"Processor: {analyzer.vlm_processor is not None}")
        st.info(f"Model: {analyzer.vlm_model is not None}")
        
        if analyzer.vlm_available:
            st.success("🎉 VLM is working in Streamlit!")
        else:
            st.error("❌ VLM failed to initialize in Streamlit")
            
    except Exception as e:
        st.error(f"❌ Error creating analyzer: {e}")
else:
    st.warning("No token available for testing") 