#!/usr/bin/env python3
"""
CORD-19 GraphRAG Streamlit App
Interactive visualization and query interface for the CORD-19 knowledge graph
"""

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import faiss
import json
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile
import os
from pathlib import Path
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CORD-19 GraphRAG Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .query-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .graph-container {
        height: 600px;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all the preprocessed data"""
    try:
        # Load metadata
        df = pd.read_csv("2020-04-10/metadata.csv")
        df = df.dropna(subset=["abstract"])
        df = df[["cord_uid", "title", "abstract", "authors", "journal", "publish_time"]]
        
        # Load graph
        G = nx.read_gml("results/cord19_graph.gml")
        
        # Load embeddings
        embeddings = np.load("results/embeddings.npy")
        
        # Load FAISS index
        index = faiss.read_index("results/faiss_index.bin")
        
        # Load query results
        with open("results/query_results.json", "r") as f:
            query_results = json.load(f)
        
        return df, G, embeddings, index, query_results
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        # Load spaCy
        nlp = spacy.load("en_core_web_sm")
        
        # Load SciBERT
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        
        # Load OpenAI client
        client = None
        if os.getenv('OPENAI_API_KEY'):
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        return nlp, tokenizer, model, client
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def embed_query(query_text, tokenizer, model):
    """Embed query text using SciBERT"""
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')

def search_papers(query, df, index, tokenizer, model, top_k=5):
    """Search for relevant papers"""
    q_vec = embed_query(query, tokenizer, model)
    D, I = index.search(q_vec, k=top_k)
    
    results = []
    for i, idx in enumerate(I[0]):
        paper = df.iloc[idx]
        results.append({
            'rank': i + 1,
            'cord_uid': paper['cord_uid'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'authors': paper['authors'],
            'journal': paper['journal'],
            'similarity_score': float(D[0][i])
        })
    
    return results

def get_graph_context(paper_ids, G):
    """Get graph context for papers"""
    context = []
    for pid in paper_ids:
        if pid in G.nodes:
            neighbors = list(G.neighbors(pid))
            paper_data = G.nodes[pid]
            title = paper_data.get('title', 'Unknown')
            context.append(f"Paper '{title}' mentions: {neighbors[:10]}")
        else:
            context.append(f"Paper {pid} not found in graph")
    return "\n".join(context)

def generate_summary(query, results, graph_context, client):
    """Generate summary using GPT-4"""
    if not client:
        return "OpenAI API key not available. Please set OPENAI_API_KEY environment variable."
    
    context_text = "\n\n".join([r['abstract'] for r in results])
    
    prompt = f"""
You are a biomedical research assistant specializing in COVID-19 research.
Summarize the following abstracts and their graph context in relation to the query.
Cite specific entities, papers, and authors where relevant.

Query: {query}

Abstracts:
{context_text}

Graph Context:
{graph_context}

Please provide a comprehensive summary with specific citations.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

def create_network_visualization(G, selected_papers=None, max_nodes=100):
    """Create interactive network visualization"""
    # Create a subgraph if too many nodes
    if len(G.nodes) > max_nodes:
        if selected_papers:
            # Include selected papers and their neighbors
            subgraph_nodes = set(selected_papers)
            for paper in selected_papers:
                if paper in G.nodes:
                    neighbors = list(G.neighbors(paper))[:10]  # Limit neighbors
                    subgraph_nodes.update(neighbors)
        else:
            # Random sample
            subgraph_nodes = list(G.nodes)[:max_nodes]
        
        subgraph = G.subgraph(subgraph_nodes)
    else:
        subgraph = G
    
    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Add nodes
    for node, data in subgraph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        
        if node_type == 'paper':
            net.add_node(node, 
                        label=data.get('title', node)[:50] + "...",
                        color="#ff6b6b",
                        size=20,
                        title=f"Type: Paper\nTitle: {data.get('title', 'N/A')}\nJournal: {data.get('journal', 'N/A')}")
        elif node_type == 'author':
            net.add_node(node,
                        label=node[:30] + "...",
                        color="#4ecdc4",
                        size=15,
                        title=f"Type: Author\nName: {node}")
        else:
            net.add_node(node,
                        label=node[:30] + "...",
                        color="#45b7d1",
                        size=10,
                        title=f"Type: {node_type}\nEntity: {node}")
    
    # Add edges
    for edge in subgraph.edges(data=True):
        net.add_edge(edge[0], edge[1], 
                    label=edge[2].get('relation', ''),
                    color="#cccccc")
    
    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.1,
                "springLength": 100,
                "springConstant": 0.05,
                "damping": 0.09
            }
        }
    }
    """)
    
    return net

def create_entity_analysis(G):
    """Create entity type analysis"""
    entity_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        entity_types[node_type] = entity_types.get(node_type, 0) + 1
    
    return entity_types

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 CORD-19 GraphRAG Explorer</h1>', unsafe_allow_html=True)
    st.markdown("Interactive exploration of COVID-19 research papers using GraphRAG (Graph Retrieval-Augmented Generation)")
    
    # Load data
    with st.spinner("Loading data..."):
        df, G, embeddings, index, query_results = load_data()
        nlp, tokenizer, model, client = load_models()
    
    if df is None:
        st.error("Failed to load data. Please ensure the pipeline has been run first.")
        return
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🔍 Query Interface", "🕸️ Graph Visualization", "📈 Analytics", "📚 Sample Queries"]
    )
    
    if page == "📊 Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", f"{len(df):,}")
        
        with col2:
            st.metric("Graph Nodes", f"{len(G.nodes):,}")
        
        with col3:
            st.metric("Graph Edges", f"{len(G.edges):,}")
        
        with col4:
            st.metric("Embeddings", f"{embeddings.shape[0]:,}")
        
        # Dataset statistics
        st.subheader("📈 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Publication year distribution
            df['year'] = pd.to_datetime(df['publish_time'], errors='coerce').dt.year
            year_counts = df['year'].value_counts().sort_index()
            
            fig = px.bar(x=year_counts.index, y=year_counts.values,
                        title="Papers by Publication Year",
                        labels={'x': 'Year', 'y': 'Number of Papers'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Journal distribution
            journal_counts = df['journal'].value_counts().head(10)
            
            fig = px.pie(values=journal_counts.values, names=journal_counts.index,
                        title="Top 10 Journals")
            st.plotly_chart(fig, use_container_width=True)
        
        # Entity analysis
        st.subheader("🧬 Knowledge Graph Analysis")
        entity_types = create_entity_analysis(G)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=list(entity_types.keys()), y=list(entity_types.values()),
                        title="Node Types in Knowledge Graph",
                        labels={'x': 'Node Type', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graph metrics
            st.markdown("**Graph Metrics:**")
            st.write(f"• **Density**: {nx.density(G):.4f}")
            st.write(f"• **Average Clustering**: {nx.average_clustering(G):.4f}")
            st.write(f"• **Number of Components**: {nx.number_connected_components(G)}")
            
            if nx.is_connected(G):
                st.write(f"• **Average Path Length**: {nx.average_shortest_path_length(G):.2f}")
    
    elif page == "🔍 Query Interface":
        st.header("🔍 Interactive Query Interface")
        
        # Query input
        query = st.text_input(
            "Enter your research question:",
            value="What drugs are being tested for COVID-19 treatment?",
            help="Ask questions about COVID-19 research, treatments, symptoms, etc."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and query:
            with st.spinner("Searching..."):
                # Search papers
                results = search_papers(query, df, index, tokenizer, model, top_k)
                
                # Get graph context
                paper_ids = [r['cord_uid'] for r in results]
                graph_context = get_graph_context(paper_ids, G)
                
                # Generate summary
                summary = generate_summary(query, results, graph_context, client)
                
                # Display results
                st.subheader("📋 Search Results")
                
                for result in results:
                    with st.expander(f"**{result['rank']}. {result['title']}** (Score: {result['similarity_score']:.3f})"):
                        st.write(f"**Authors:** {result['authors']}")
                        st.write(f"**Journal:** {result['journal']}")
                        st.write(f"**Abstract:** {result['abstract']}")
                
                # Display summary
                st.subheader("🤖 AI-Generated Summary")
                st.markdown(f'<div class="query-result">{summary}</div>', unsafe_allow_html=True)
                
                # Store results for graph visualization
                st.session_state['last_query_results'] = results
                st.session_state['last_query'] = query
    
    elif page == "🕸️ Graph Visualization":
        st.header("🕸️ Interactive Graph Visualization")
        
        # Graph options
        col1, col2 = st.columns(2)
        
        with col1:
            max_nodes = st.slider("Maximum nodes to display:", min_value=50, max_value=500, value=100)
        
        with col2:
            show_selected = st.checkbox("Show only selected papers from last query", value=False)
        
        # Get selected papers if available
        selected_papers = None
        if show_selected and 'last_query_results' in st.session_state:
            selected_papers = [r['cord_uid'] for r in st.session_state['last_query_results']]
            st.info(f"Showing graph for {len(selected_papers)} selected papers from last query")
        
        # Create visualization
        with st.spinner("Generating graph visualization..."):
            net = create_network_visualization(G, selected_papers, max_nodes)
            
            # Save to temporary file and display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                
                with open(tmp_file.name, "r") as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=600)
                
                # Clean up
                os.unlink(tmp_file.name)
        
        # Graph legend
        st.markdown("""
        **Graph Legend:**
        - 🔴 **Red nodes**: Research papers
        - 🟢 **Green nodes**: Authors
        - 🔵 **Blue nodes**: Entities (diseases, drugs, etc.)
        - **Gray edges**: Relationships between nodes
        """)
    
    elif page == "📈 Analytics":
        st.header("📈 Advanced Analytics")
        
        # Entity co-occurrence analysis
        st.subheader("🔗 Entity Co-occurrence Analysis")
        
        # Get entity pairs
        entity_pairs = []
        for edge in G.edges(data=True):
            node1, node2, data = edge
            node1_type = G.nodes[node1].get('type', 'unknown')
            node2_type = G.nodes[node2].get('type', 'unknown')
            
            if node1_type != 'paper' and node2_type != 'paper':
                entity_pairs.append((node1, node2, data.get('relation', '')))
        
        if entity_pairs:
            # Create co-occurrence matrix
            entities = list(set([pair[0] for pair in entity_pairs] + [pair[1] for pair in entity_pairs]))
            co_occurrence = np.zeros((len(entities), len(entities)))
            
            entity_to_idx = {entity: i for i, entity in enumerate(entities)}
            
            for pair in entity_pairs:
                idx1 = entity_to_idx[pair[0]]
                idx2 = entity_to_idx[pair[1]]
                co_occurrence[idx1][idx2] += 1
                co_occurrence[idx2][idx1] += 1
            
            # Display heatmap
            fig = px.imshow(co_occurrence[:20, :20],  # Show top 20x20
                           labels=dict(x="Entity", y="Entity", color="Co-occurrence"),
                           x=entities[:20],
                           y=entities[:20],
                           title="Entity Co-occurrence Heatmap (Top 20x20)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Paper similarity analysis
        st.subheader("📊 Paper Similarity Analysis")
        
        if 'last_query_results' in st.session_state:
            results = st.session_state['last_query_results']
            
            # Create similarity matrix
            similarity_scores = [r['similarity_score'] for r in results]
            paper_titles = [r['title'][:50] + "..." for r in results]
            
            fig = px.bar(x=paper_titles, y=similarity_scores,
                        title="Paper Similarity Scores",
                        labels={'x': 'Paper', 'y': 'Similarity Score'})
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Graph centrality analysis
        st.subheader("🎯 Centrality Analysis")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Get top central nodes
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Nodes by Degree Centrality:**")
            for node, centrality in top_degree:
                node_type = G.nodes[node].get('type', 'unknown')
                st.write(f"• {node[:50]}... ({node_type}) - {centrality:.3f}")
        
        with col2:
            st.write("**Top 10 Nodes by Betweenness Centrality:**")
            for node, centrality in top_betweenness:
                node_type = G.nodes[node].get('type', 'unknown')
                st.write(f"• {node[:50]}... ({node_type}) - {centrality:.3f}")
    
    elif page == "📚 Sample Queries":
        st.header("📚 Sample Queries")
        
        st.markdown("Try these example queries to explore the CORD-19 dataset:")
        
        sample_queries = [
            "What are the symptoms of COVID-19?",
            "How is SARS-CoV-2 transmitted?",
            "What treatments are available for coronavirus?",
            "What is the mortality rate of COVID-19?",
            "How effective are masks in preventing COVID-19?",
            "What vaccines are being developed for COVID-19?",
            "What are the long-term effects of COVID-19?",
            "How does COVID-19 affect different age groups?",
            "What diagnostic tests are available for COVID-19?",
            "How has COVID-19 impacted healthcare systems?"
        ]
        
        # Display queries in a grid
        cols = st.columns(2)
        for i, query in enumerate(sample_queries):
            with cols[i % 2]:
                if st.button(f"🔍 {query}", key=f"query_{i}"):
                    st.session_state['selected_query'] = query
                    st.rerun()
        
        # If a query is selected, show it in the query interface
        if 'selected_query' in st.session_state:
            st.markdown("---")
            st.markdown("**Selected Query:**")
            st.code(st.session_state['selected_query'])
            st.markdown("Go to the Query Interface page to run this query!")

if __name__ == "__main__":
    main()
