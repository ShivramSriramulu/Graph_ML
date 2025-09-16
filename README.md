# 🧬 CORD-19 GraphRAG Pipeline

A comprehensive GraphRAG (Graph Retrieval-Augmented Generation) pipeline for exploring COVID-19 research papers from the CORD-19 dataset. This project combines knowledge graph construction, semantic search, and AI-powered summarization to enable interactive exploration of biomedical research.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.49+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### 🔍 **Semantic Search**
- Natural language queries powered by SciBERT embeddings
- FAISS vector search for fast retrieval
- Similarity scoring and ranking

### 🕸️ **Knowledge Graph**
- Automated entity extraction using spaCy
- NetworkX-based graph construction
- Interactive visualization with Pyvis

### 🤖 **AI-Powered Summarization**
- GPT-4 integration for intelligent summaries
- Citation-aware responses
- Context from knowledge graph

### 📊 **Interactive Web App**
- Streamlit-based dashboard
- Real-time visualizations
- Multiple analysis views

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ShivramSriramulu/Graph_ML.git
cd Graph_ML
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv cord_env
source cord_env/bin/activate  # On Windows: cord_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Download CORD-19 Dataset
```bash
# Download the CORD-19 dataset from:
# https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge
# Place the metadata.csv file in the 2020-04-10/ directory
```

### 4. Run the Pipeline
```bash
# Run the complete GraphRAG pipeline
python run_pipeline.py

# Or run the Jupyter notebook
jupyter notebook CORD19_GraphRAG_Pipeline.ipynb
```

### 5. Launch the Web App
```bash
# Launch Streamlit app
streamlit run streamlit_app.py

# Or use the smart launcher
python launch_app.py
```

## 📁 Project Structure

```
Graph_ML/
├── 📊 CORD19_GraphRAG_Pipeline.ipynb    # Jupyter notebook
├── 🚀 streamlit_app.py                  # Streamlit web app
├── 🔧 run_pipeline.py                   # Pipeline runner
├── 🎯 launch_app.py                     # Smart app launcher
├── 📋 requirements.txt                  # Dependencies
├── ⚙️ setup.py                          # Package setup
├── 📚 README.md                         # This file
├── 📖 STREAMLIT_README.md              # Streamlit documentation
├── 🗂️ .streamlit/                       # Streamlit config
├── 📊 results/                          # Generated data
└── 📁 2020-04-10/                      # CORD-19 dataset
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for processing large datasets)
- OpenAI API key (for GPT-4 summarization)

### Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **pandas**: Data manipulation
- **networkx**: Graph processing
- **transformers**: SciBERT embeddings
- **faiss-cpu**: Vector search
- **streamlit**: Web interface
- **plotly**: Interactive visualizations
- **pyvis**: Network visualization
- **openai**: GPT-4 integration

## 🔧 Usage

### Command Line Interface
```bash
# Run complete pipeline
python run_pipeline.py

# Launch web app
streamlit run streamlit_app.py

# Demo capabilities
python demo_streamlit.py
```

### Jupyter Notebook
```bash
jupyter notebook CORD19_GraphRAG_Pipeline.ipynb
```

### Web Interface
1. Launch the Streamlit app
2. Navigate to `http://localhost:8501`
3. Explore different sections:
   - **Overview**: Dataset statistics
   - **Query Interface**: Search papers
   - **Graph Visualization**: Interactive network
   - **Analytics**: Advanced analysis

## 📊 Dataset

The CORD-19 dataset contains research papers related to COVID-19, SARS-CoV-2, and related coronaviruses. This pipeline processes:

- **42,351 papers** with abstracts
- **1,213 nodes** in knowledge graph
- **1,319 edges** representing relationships
- **200 embeddings** for semantic search

## 🎯 Example Queries

Try these sample queries in the web interface:

- "What drugs are being tested for COVID-19 treatment?"
- "What are the symptoms of COVID-19?"
- "How effective are masks in preventing COVID-19?"
- "What vaccines are being developed for COVID-19?"
- "What are the long-term effects of COVID-19?"

## 🔬 Technical Details

### GraphRAG Pipeline
1. **Data Loading**: Load and preprocess CORD-19 metadata
2. **Entity Extraction**: Extract biomedical entities using spaCy
3. **Graph Construction**: Build knowledge graph with NetworkX
4. **Embeddings**: Generate SciBERT embeddings
5. **Indexing**: Create FAISS vector index
6. **Search**: Semantic search with similarity scoring
7. **Summarization**: GPT-4 powered summaries

### Architecture
- **Backend**: Python with pandas, NetworkX, transformers
- **Frontend**: Streamlit with Plotly and Pyvis
- **ML Models**: SciBERT, GPT-4, spaCy
- **Search**: FAISS vector database

## 📈 Performance

- **Processing Time**: ~5-10 minutes for full pipeline
- **Search Speed**: ~1-2 seconds per query
- **Memory Usage**: ~2-4 GB depending on dataset size
- **Graph Rendering**: ~2-3 seconds for 100 nodes

## 🚀 Deployment

### Local Deployment
```bash
# Run locally
streamlit run streamlit_app.py
```

### Cloud Deployment
The app can be deployed to:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Using Procfile
- **AWS/GCP/Azure**: Container deployment
- **Docker**: Containerized deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CORD-19 Dataset**: Allen Institute for AI
- **SciBERT**: AllenAI for biomedical embeddings
- **GPT-4**: OpenAI for language generation
- **Streamlit**: For the web framework
- **NetworkX**: For graph processing
- **FAISS**: Facebook AI for vector search

## 📞 Contact

- **Author**: Shivram Sriramulu
- **GitHub**: [@ShivramSriramulu](https://github.com/ShivramSriramulu)
- **Repository**: [Graph_ML](https://github.com/ShivramSriramulu/Graph_ML)

## 🔗 Links

- [CORD-19 Dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NetworkX Documentation](https://networkx.org/)
- [SciBERT Paper](https://arxiv.org/abs/1903.10676)

---

⭐ **Star this repository if you find it helpful!**