# 🧬 CORD-19 GraphRAG Streamlit App

An interactive web application for exploring the CORD-19 dataset using GraphRAG (Graph Retrieval-Augmented Generation) technology.

## 🚀 Quick Start

### 1. Prerequisites
Make sure you have run the CORD-19 GraphRAG pipeline first to generate the required data files:
```bash
python run_pipeline.py
```

### 2. Launch the App
```bash
# Option 1: Using the launcher script
./run_streamlit.sh

# Option 2: Manual launch
source cord_env/bin/activate
streamlit run streamlit_app.py
```

### 3. Access the App
Open your browser and go to: `http://localhost:8501`

## 🎯 Features

### 📊 Overview Dashboard
- **Dataset Statistics**: Total papers, graph nodes/edges, embeddings count
- **Publication Trends**: Interactive charts showing papers by year
- **Journal Distribution**: Pie charts of top journals
- **Graph Metrics**: Density, clustering, connectivity analysis

### 🔍 Interactive Query Interface
- **Natural Language Queries**: Ask questions in plain English
- **Semantic Search**: Powered by SciBERT embeddings
- **AI-Generated Summaries**: GPT-4 powered responses with citations
- **Relevant Paper Ranking**: Similarity scores and detailed abstracts

### 🕸️ Graph Visualization
- **Interactive Network**: Explore the knowledge graph visually
- **Node Types**: Papers (red), Authors (green), Entities (blue)
- **Relationship Mapping**: See how papers connect to entities and authors
- **Filtered Views**: Focus on specific papers from search results

### 📈 Advanced Analytics
- **Entity Co-occurrence**: Heatmaps showing related entities
- **Centrality Analysis**: Identify key papers and entities
- **Similarity Analysis**: Compare paper relevance scores
- **Graph Metrics**: Network analysis and statistics

### 📚 Sample Queries
- **Pre-built Questions**: Common COVID-19 research questions
- **One-click Search**: Instant access to example queries
- **Query Suggestions**: Get inspired with relevant questions

## 🛠️ Technical Architecture

### Backend Components
- **Data Loading**: Cached loading of graph, embeddings, and metadata
- **Model Integration**: SciBERT for embeddings, GPT-4 for summarization
- **Search Engine**: FAISS vector search with similarity scoring
- **Graph Processing**: NetworkX for graph operations

### Frontend Features
- **Streamlit UI**: Clean, responsive web interface
- **Interactive Charts**: Plotly-powered visualizations
- **Network Visualization**: Pyvis for interactive graph display
- **Real-time Updates**: Dynamic content based on user interactions

## 📁 File Structure

```
Cord/
├── streamlit_app.py              # Main Streamlit application
├── run_streamlit.sh              # App launcher script
├── requirements_streamlit.txt    # Python dependencies
├── .env                          # Environment variables (API keys)
├── results/                      # Generated data files
│   ├── cord19_graph.gml         # Knowledge graph
│   ├── embeddings.npy           # SciBERT embeddings
│   ├── faiss_index.bin          # Vector search index
│   └── query_results.json       # Query results
└── 2020-04-10/                  # CORD-19 dataset
    └── metadata.csv             # Paper metadata
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Customization Options
- **Max Graph Nodes**: Adjust visualization complexity
- **Search Results**: Control number of returned papers
- **Model Settings**: Modify embedding and summarization parameters

## 💡 Usage Examples

### Basic Query
1. Go to "Query Interface" page
2. Enter: "What drugs are being tested for COVID-19 treatment?"
3. Click "Search"
4. Review results and AI-generated summary

### Graph Exploration
1. Run a query first to get relevant papers
2. Go to "Graph Visualization" page
3. Enable "Show only selected papers from last query"
4. Explore the interactive network

### Analytics Deep Dive
1. Go to "Analytics" page
2. View entity co-occurrence heatmaps
3. Analyze centrality measures
4. Compare paper similarity scores

## 🎨 UI Components

### Navigation
- **Sidebar**: Easy navigation between different features
- **Tabs**: Organized content sections
- **Expandable Cards**: Detailed information on demand

### Visualizations
- **Interactive Charts**: Hover, zoom, and filter capabilities
- **Network Graphs**: Click and drag to explore connections
- **Heatmaps**: Color-coded relationship matrices
- **Progress Bars**: Real-time loading indicators

### Data Display
- **Search Results**: Ranked list with similarity scores
- **Paper Details**: Expandable abstracts and metadata
- **AI Summaries**: Formatted responses with citations
- **Graph Legends**: Clear node type explanations

## 🔍 Sample Queries to Try

1. **Treatment Research**
   - "What drugs are being tested for COVID-19 treatment?"
   - "What are the most promising COVID-19 treatments?"
   - "How effective is remdesivir against COVID-19?"

2. **Symptoms and Effects**
   - "What are the long-term effects of COVID-19?"
   - "How does COVID-19 affect different age groups?"
   - "What are the neurological symptoms of COVID-19?"

3. **Prevention and Control**
   - "How effective are masks in preventing COVID-19?"
   - "What social distancing measures work best?"
   - "How does ventilation affect COVID-19 transmission?"

4. **Vaccines and Immunity**
   - "What vaccines are being developed for COVID-19?"
   - "How long does COVID-19 immunity last?"
   - "What are the side effects of COVID-19 vaccines?"

5. **Diagnostics and Testing**
   - "What diagnostic tests are available for COVID-19?"
   - "How accurate are rapid COVID-19 tests?"
   - "What biomarkers indicate severe COVID-19?"

## 🚨 Troubleshooting

### Common Issues

**App won't start**
- Ensure virtual environment is activated
- Check that all dependencies are installed
- Verify data files exist in `results/` directory

**Search not working**
- Confirm SciBERT model is loaded
- Check FAISS index file exists
- Verify embeddings are properly generated

**Graph not displaying**
- Ensure Pyvis is installed
- Check browser compatibility
- Try reducing max nodes setting

**AI summaries not generating**
- Verify OpenAI API key is set
- Check API key validity
- Ensure sufficient API credits

### Performance Tips

1. **Reduce Graph Size**: Lower max nodes for faster rendering
2. **Cache Results**: Use Streamlit's caching for repeated operations
3. **Batch Processing**: Process multiple queries together
4. **Memory Management**: Restart app if memory usage is high

## 🔮 Future Enhancements

- **Real-time Updates**: Live data synchronization
- **Advanced Filters**: Date ranges, journal types, author filters
- **Export Features**: Download results as PDF/CSV
- **Collaboration**: Share queries and results
- **Mobile Support**: Responsive design for mobile devices
- **API Integration**: RESTful API for external access

## 📊 Performance Metrics

- **Load Time**: ~5-10 seconds for initial data loading
- **Search Speed**: ~1-2 seconds per query
- **Graph Rendering**: ~2-3 seconds for 100 nodes
- **Memory Usage**: ~2-4 GB depending on dataset size

## 🤝 Contributing

Feel free to contribute by:
- Adding new visualization types
- Improving query processing
- Enhancing UI/UX design
- Adding new analytics features
- Optimizing performance

## 📄 License

This project is for research purposes. Please respect the individual licenses of papers in the CORD-19 dataset.

## 🙏 Acknowledgments

- CORD-19 dataset providers
- AllenAI for SciBERT
- OpenAI for GPT-4
- Streamlit team for the framework
- NetworkX and Pyvis for graph visualization
- Plotly for interactive charts
