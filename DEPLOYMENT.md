# Streamlit Deployment Guide

## Overview
This guide explains how to deploy the MSU Club Discovery RAG Assistant using Streamlit Cloud or a self-hosted Streamlit server.

## Prerequisites
- Python 3.8+
- Pinecone account with pre-populated vector database
- Groq API key (free tier available)
- Streamlit account (for cloud deployment)

## Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Secrets Locally
Create `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "msu-clubs-index"
PINECONE_NAMESPACE = "clubs"
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key"
LLM_MODEL = "llama-3.3-70b-versatile"
```

**Important:** Never commit `secrets.toml` to version control!

### 3. Run Locally
```bash
streamlit run app_streamlit.py
```

The app will be available at `http://localhost:8501`

---

## Streamlit Cloud Deployment

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Deploy streamlit version"
git push origin streamlit-deploy
```

### 2. Create Streamlit App on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository and `streamlit-deploy` branch
4. Select `app_streamlit.py` as the main file

### 3. Add Secrets
1. In Streamlit Cloud app settings, go to "Secrets"
2. Add your secrets in TOML format:
```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "msu-clubs-index"
PINECONE_NAMESPACE = "clubs"
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key"
LLM_MODEL = "llama-3.3-70b-versatile"
```

### 4. Deploy
Your app will automatically deploy and restart when you push changes to GitHub!

---

## Self-Hosted Deployment (Docker)

### Dockerfile
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      PINECONE_API_KEY: ${PINECONE_API_KEY}
      PINECONE_INDEX_NAME: msu-clubs-index
      PINECONE_NAMESPACE: clubs
      LLM_PROVIDER: groq
      GROQ_API_KEY: ${GROQ_API_KEY}
      LLM_MODEL: llama-3.3-70b-versatile
    volumes:
      - ./.streamlit:/app/.streamlit
```

### Run with Docker
```bash
docker-compose up -d
```

---

## Production Best Practices

### 1. Environment Variables
- Use Streamlit secrets for sensitive data
- Never hardcode API keys
- Use environment-specific configurations

### 2. Performance
- Cache RAG engine initialization
- Use Streamlit's caching decorators (`@st.cache_resource`)
- Limit number of results returned

### 3. Monitoring
- Monitor Streamlit logs for errors
- Set up alerts for API failures
- Track usage metrics

### 4. Security
- Use HTTPS for all connections
- Implement rate limiting if needed
- Validate user inputs
- Keep dependencies updated

---

## Troubleshooting

### Issue: "Secret not found" error
**Solution:** Ensure all required secrets are added to `.streamlit/secrets.toml` (local) or Streamlit Cloud settings (cloud)

### Issue: Slow search results
**Solution:** 
- Check Pinecone connection
- Verify embedding model is working
- Monitor API response times

### Issue: "Dimension mismatch" error
**Solution:**
- Ensure sentence-transformers is installed
- Verify Pinecone index has 1024 dimensions
- Check that embeddings are being generated correctly

### Issue: App crashes on startup
**Solution:**
- Check Streamlit logs: `streamlit run app_streamlit.py --logger.level=debug`
- Verify all dependencies are installed
- Check for missing or invalid secrets

---

## File Structure for Deployment

```
project/
├── app_streamlit.py          # Main Streamlit app (deployment version)
├── requirements.txt          # Streamlit deployment dependencies
├── config_streamlit.py       # Streamlit secrets configuration
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml.example # Secrets template
├── src/
│   ├── vector_store.py      # Pinecone integration
│   ├── rag_engine.py        # RAG logic
│   └── llm_client.py        # LLM integration
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── DEPLOYMENT.md           # This file
```

---

## Support & Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Groq API Docs](https://console.groq.com/docs)
