# Streamlit Cloud Deployment Guide

## 🔐 Secure Token Management for Streamlit Cloud

### Why This Matters
Your previous token was compromised because it was hardcoded in a file that was pushed to GitHub. For Streamlit Cloud deployment, we use **Streamlit Secrets** which are encrypted and secure.

## 🚀 Step-by-Step Deployment

### 1. Get a New Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name like "streamlit-escalator-app"
4. Select "Read" permissions
5. Copy the new token (starts with `hf_`)

### 2. Deploy to Streamlit Cloud

1. **Push your code to GitHub** (make sure NO token files are included)
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Connect your GitHub repository
5. Set the main file path: `streamlit_escalator_vlm.py`

### 3. Add Your Token to Streamlit Secrets

**Method 1: Using Streamlit Cloud Dashboard**
1. In your Streamlit Cloud app dashboard
2. Go to "Settings" → "Secrets"
3. Add this configuration:

```toml
HF_TOKEN = "hf_your_new_token_here"
```

**Method 2: Using .streamlit/secrets.toml (for local testing)**
1. Create a `.streamlit` folder in your project
2. Create `secrets.toml` inside it:

```toml
HF_TOKEN = "hf_your_new_token_here"
```

**⚠️ IMPORTANT: Add `.streamlit/secrets.toml` to your `.gitignore` file!**

### 4. Update .gitignore

Make sure your `.gitignore` includes:

```gitignore
# Token files
hf_token.txt
test_token.py
.streamlit/secrets.toml

# Environment files
.env
*.env

# Python cache
__pycache__/
*.pyc
```

## 🔧 Local Development Setup

### Option 1: Environment Variable (Recommended)
```bash
# Windows PowerShell
$env:HF_TOKEN="hf_your_new_token_here"

# Windows Command Prompt
set HF_TOKEN=hf_your_new_token_here

# Linux/Mac
export HF_TOKEN=hf_your_new_token_here
```

### Option 2: .streamlit/secrets.toml (for testing)
Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_new_token_here"
```

## 🛡️ Security Best Practices

### ✅ DO:
- Use Streamlit secrets for production
- Use environment variables for local development
- Keep tokens out of version control
- Use read-only tokens when possible

### ❌ DON'T:
- Hardcode tokens in files
- Commit token files to GitHub
- Share tokens publicly
- Use write permissions unless needed

## 🔍 Verification

After deployment, your app should show:
- ✅ "🔐 Token loaded from Streamlit secrets (production)"
- ✅ VLM analysis working properly
- ✅ No token visible in the interface

## 🚨 Troubleshooting

### "No token found" error:
1. Check if token is added to Streamlit secrets
2. Verify token format (starts with `hf_`)
3. Ensure token has correct permissions

### VLM not working:
1. Verify token is valid at [Hugging Face](https://huggingface.co/settings/tokens)
2. Check if model access is granted
3. Try regenerating the token

### Local vs Cloud differences:
- Local: Uses environment variables or secrets.toml
- Cloud: Uses Streamlit secrets dashboard

## 📝 Example Configuration

### For Streamlit Cloud Secrets:
```toml
HF_TOKEN = "hf_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
```

### For Local Development (.streamlit/secrets.toml):
```toml
HF_TOKEN = "hf_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
```

### For Environment Variable:
```bash
export HF_TOKEN=hf_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```

## 🎯 Quick Deployment Checklist

- [ ] Get new Hugging Face token
- [ ] Remove old token files from repository
- [ ] Update .gitignore
- [ ] Push clean code to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Add token to Streamlit secrets
- [ ] Test VLM functionality
- [ ] Verify security (no tokens visible)

## 🔗 Useful Links

- [Streamlit Cloud](https://share.streamlit.io/)
- [Hugging Face Tokens](https://huggingface.co/settings/tokens)
- [Streamlit Secrets Documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management) 