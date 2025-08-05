# 🔥 Manual Upload Instructions

Since the automated upload failed due to authentication issues, here are the manual steps to upload your project to the server.

## 📦 Step 1: Create Project Package

The upload script has already created a package for you. If you need to create it again, run:

```bash
python upload_to_server.py
```

This will create a file like: `Thermal_Image_AI_Analyzer_20250730_165736.zip`

## 🚀 Step 2: Manual Upload Options

### Option A: Using SCP (if you have SSH access)
```bash
scp -P 8024 Thermal_Image_AI_Analyzer_*.zip potatonet@112.149.254.77:/home/potatonet/workspace/
```

### Option B: Using SFTP Client
1. Connect to `112.149.254.77:8024` with username `potatonet`
2. Navigate to `/home/potatonet/workspace/`
3. Upload the zip file
4. Create folder `T` and extract the contents

### Option C: Using File Manager
1. Use WinSCP, FileZilla, or similar FTP client
2. Connect to `112.149.254.77:8024`
3. Upload the zip file to `/home/potatonet/workspace/`

## 📂 Step 3: Server Setup

Once you have access to the server, run these commands:

```bash
# SSH to the server
ssh -p 8024 potatonet@112.149.254.77

# Navigate to workspace
cd /home/potatonet/workspace

# Create T directory
mkdir -p T

# Move and extract the package
mv Thermal_Image_AI_Analyzer_*.zip T/
cd T
unzip Thermal_Image_AI_Analyzer_*.zip
rm Thermal_Image_AI_Analyzer_*.zip

# Set up the environment
conda create -n thermal_img python=3.9 -y
conda activate thermal_img

# Install dependencies
pip install -r requirements.txt

# Create startup script
cat > start_thermal_analyzer.sh << 'EOF'
#!/bin/bash
cd /home/potatonet/workspace/T
conda activate thermal_img
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
EOF

chmod +x start_thermal_analyzer.sh
```

## 🎯 Step 4: Start the Application

```bash
# Start the application
./start_thermal_analyzer.sh
```

## 🌐 Step 5: Access the Application

Open your browser and go to:
```
http://112.149.254.77:8501
```

## 📋 Project Files Included

The package contains:
- `streamlit_app.py` - Main web application
- `thermal_vlm_processor.py` - BLIP/GIT processor
- `thermal_vlm_comparison.py` - LLaVA-Next processor
- `thermal_vlm_ensemble.py` - Ensemble system
- `thermal_knowledge_injection.py` - Knowledge injection
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `test_image/` - Sample thermal images
- `test_ensemble.py` - Ensemble testing
- `test_knowledge_injection.py` - Knowledge injection testing
- `quick_test.py` - Quick verification
- `processimage.py` - Basic image processing

## 🔧 Troubleshooting

### Authentication Issues
- Ensure you have the correct username and password
- Check if SSH key authentication is required
- Verify the server is accessible from your network

### Port Issues
- Make sure port 8024 is open for SSH
- Ensure port 8501 is available for Streamlit

### Dependencies Issues
- Make sure conda/miniconda is installed on the server
- Check Python version compatibility
- Install system dependencies if needed (OpenCV, etc.)

## 📞 Support

If you encounter issues:
1. Check the server logs
2. Verify network connectivity
3. Ensure all dependencies are installed
4. Check file permissions on the server

---

**🎉 Your Thermal Image AI Analyzer will be ready to use once deployed!** 