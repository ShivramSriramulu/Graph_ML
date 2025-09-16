# 🚀 Deployment Instructions

This document provides step-by-step instructions for deploying the CORD-19 GraphRAG Streamlit app to various platforms.

## 🌐 Streamlit Cloud (Recommended)

### 1. Prepare Repository
- Ensure all files are pushed to GitHub
- Add your OpenAI API key to Streamlit Cloud secrets

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `ShivramSriramulu/Graph_ML`
5. Set main file path: `streamlit_app.py`
6. Add secrets in the "Advanced settings":
   ```toml
   [openai]
   api_key = "your_openai_api_key_here"
   ```
7. Click "Deploy!"

### 3. Access Your App
- Your app will be available at: `https://your-app-name.streamlit.app`
- The deployment process takes 2-3 minutes

## 🐳 Docker Deployment

### 1. Build Docker Image
```bash
docker build -t cord19-graphrag .
```

### 2. Run Container
```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key cord19-graphrag
```

### 3. Access App
- Open browser to `http://localhost:8501`

## ☁️ Heroku Deployment

### 1. Install Heroku CLI
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli
```

### 2. Create Heroku App
```bash
heroku create your-app-name
```

### 3. Set Environment Variables
```bash
heroku config:set OPENAI_API_KEY=your_openai_api_key
```

### 4. Deploy
```bash
git push heroku main
```

### 5. Open App
```bash
heroku open
```

## 🔧 AWS/GCP/Azure Deployment

### 1. Container Registry
```bash
# Build and push to registry
docker build -t your-registry/cord19-graphrag .
docker push your-registry/cord19-graphrag
```

### 2. Deploy to Cloud
- Use your cloud provider's container service
- Set environment variables
- Configure load balancer for port 8501

## 📊 Performance Optimization

### 1. Memory Requirements
- **Minimum**: 2GB RAM
- **Recommended**: 4GB+ RAM
- **For large datasets**: 8GB+ RAM

### 2. CPU Requirements
- **Minimum**: 2 CPU cores
- **Recommended**: 4+ CPU cores

### 3. Storage Requirements
- **Minimum**: 5GB
- **With full dataset**: 20GB+

## 🔐 Security Considerations

### 1. API Keys
- Never commit API keys to repository
- Use environment variables or secrets management
- Rotate keys regularly

### 2. Data Privacy
- CORD-19 dataset is public
- No sensitive data in the application
- Follow data protection regulations

### 3. Access Control
- Consider authentication for production use
- Implement rate limiting
- Monitor usage and costs

## 🐛 Troubleshooting

### Common Issues

**App won't start**
- Check Python version (3.8+)
- Verify all dependencies installed
- Check environment variables

**Memory errors**
- Reduce dataset size
- Increase memory allocation
- Use smaller models

**API errors**
- Verify API key is correct
- Check API quota and billing
- Test API connectivity

**Slow performance**
- Use smaller graph samples
- Enable caching
- Optimize queries

### Debug Mode
```bash
# Run with debug information
streamlit run streamlit_app.py --logger.level=debug
```

## 📈 Monitoring

### 1. Application Metrics
- Response time
- Memory usage
- CPU utilization
- Error rates

### 2. Usage Analytics
- Number of queries
- Popular search terms
- User engagement
- Performance trends

### 3. Cost Monitoring
- API usage costs
- Infrastructure costs
- Storage costs

## 🔄 Updates and Maintenance

### 1. Code Updates
```bash
git pull origin main
# Restart application
```

### 2. Dependency Updates
```bash
pip install -r requirements.txt --upgrade
```

### 3. Data Updates
- Re-run pipeline with new data
- Update embeddings and indices
- Refresh knowledge graph

## 📞 Support

For deployment issues:
1. Check the logs
2. Review this documentation
3. Open an issue on GitHub
4. Contact the maintainer

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured
- [ ] Performance monitoring enabled
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Backup strategy in place
- [ ] Security measures applied
- [ ] Documentation updated
