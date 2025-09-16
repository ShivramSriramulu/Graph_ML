#!/usr/bin/env python3
"""
Demo script to showcase CORD-19 GraphRAG Streamlit App features
"""

import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import json
from pathlib import Path

def demo_data_loading():
    """Demonstrate data loading capabilities"""
    print("🔍 Demo: Data Loading")
    print("=" * 40)
    
    try:
        # Load metadata
        df = pd.read_csv("2020-04-10/metadata.csv")
        df = df.dropna(subset=["abstract"])
        print(f"✅ Loaded {len(df):,} papers with abstracts")
        
        # Load graph
        G = nx.read_gml("results/cord19_graph.gml")
        print(f"✅ Loaded graph with {len(G.nodes):,} nodes and {len(G.edges):,} edges")
        
        # Load embeddings
        embeddings = np.load("results/embeddings.npy")
        print(f"✅ Loaded {embeddings.shape[0]:,} embeddings with {embeddings.shape[1]} dimensions")
        
        # Load query results
        with open("results/query_results.json", "r") as f:
            query_results = json.load(f)
        print(f"✅ Loaded query results for: '{query_results['query']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def demo_graph_analysis():
    """Demonstrate graph analysis capabilities"""
    print("\n🕸️ Demo: Graph Analysis")
    print("=" * 40)
    
    try:
        G = nx.read_gml("results/cord19_graph.gml")
        
        # Node type analysis
        entity_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        print("📊 Node Types:")
        for node_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {node_type}: {count:,}")
        
        # Graph metrics
        print(f"\n📈 Graph Metrics:")
        print(f"   Density: {nx.density(G):.4f}")
        print(f"   Average Clustering: {nx.average_clustering(G):.4f}")
        print(f"   Number of Components: {nx.number_connected_components(G)}")
        
        # Centrality analysis
        degree_centrality = nx.degree_centrality(G)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n🎯 Top 5 Most Connected Nodes:")
        for node, centrality in top_degree:
            node_type = G.nodes[node].get('type', 'unknown')
            title = G.nodes[node].get('title', node)[:50]
            print(f"   {title}... ({node_type}) - {centrality:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
        return False

def demo_query_results():
    """Demonstrate query results"""
    print("\n🔍 Demo: Query Results")
    print("=" * 40)
    
    try:
        with open("results/query_results.json", "r") as f:
            query_results = json.load(f)
        
        print(f"Query: '{query_results['query']}'")
        print(f"Found {len(query_results['paper_ids'])} relevant papers")
        print(f"Generated summary: {len(query_results['summary'])} characters")
        
        # Show summary preview
        summary_preview = query_results['summary'][:200] + "..."
        print(f"\n📝 Summary Preview:")
        print(f"   {summary_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading query results: {e}")
        return False

def demo_streamlit_features():
    """Demonstrate Streamlit app features"""
    print("\n🚀 Demo: Streamlit App Features")
    print("=" * 40)
    
    features = [
        "📊 Overview Dashboard - Dataset statistics and visualizations",
        "🔍 Interactive Query Interface - Natural language search with AI summaries",
        "🕸️ Graph Visualization - Interactive network exploration",
        "📈 Advanced Analytics - Entity co-occurrence and centrality analysis",
        "📚 Sample Queries - Pre-built questions for quick exploration"
    ]
    
    print("Available Features:")
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🌐 Access the app at: http://localhost:8501")
    print(f"📖 Read the full documentation: STREAMLIT_README.md")
    
    return True

def main():
    """Run all demos"""
    print("🧬 CORD-19 GraphRAG Streamlit App Demo")
    print("=" * 50)
    
    # Check if data exists
    if not Path("results/cord19_graph.gml").exists():
        print("❌ Graph data not found. Please run the pipeline first:")
        print("   python run_pipeline.py")
        return
    
    # Run demos
    demos = [
        demo_data_loading,
        demo_graph_analysis,
        demo_query_results,
        demo_streamlit_features
    ]
    
    success_count = 0
    for demo in demos:
        if demo():
            success_count += 1
    
    print(f"\n🎉 Demo Complete: {success_count}/{len(demos)} demos successful")
    
    if success_count == len(demos):
        print("\n✅ All systems ready! You can now run the Streamlit app:")
        print("   ./run_streamlit.sh")
        print("   or")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Some demos failed. Check the error messages above.")

if __name__ == "__main__":
    main()
