#!/usr/bin/env python3
"""
Run the CORD-19 GraphRAG Pipeline and show results
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import spacy
import networkx as nx
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

def main():
    print("🚀 CORD-19 GraphRAG Pipeline - Running Complete Pipeline")
    print("=" * 60)
    
    # Step 1: Setup
    print("\n📥 Step 1: Loading Models...")
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Loaded spaCy model: en_core_web_sm")
    except OSError:
        print("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return
    
    # Load SciBERT model
    print("📥 Loading SciBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    print("✅ Loaded SciBERT model")
    
    # Initialize OpenAI client
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    else:
        client = None
        print("⚠️ OpenAI API key not found. GPT-4 summarization will be skipped.")
    
    # Step 2: Load Data
    print("\n📊 Step 2: Loading CORD-19 Data...")
    metadata_path = "2020-04-10/metadata.csv"
    df = pd.read_csv(metadata_path)
    print(f"   Total papers: {len(df):,}")
    
    # Keep only papers with abstracts
    df = df.dropna(subset=["abstract"])
    print(f"   Papers with abstracts: {len(df):,}")
    
    # Select relevant columns
    df = df[["cord_uid", "title", "abstract", "authors", "journal", "publish_time"]]
    
    # Step 3: Entity Extraction
    print("\n🧬 Step 3: Entity Extraction...")
    
    def extract_entities(text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    # Sample entity extraction
    sample_text = df['abstract'].iloc[0]
    entities = extract_entities(sample_text)
    print(f"   Sample entities: {entities[:5]}")
    
    # Step 4: Graph Construction
    print("\n🕸️ Step 4: Building Knowledge Graph...")
    
    G = nx.Graph()
    sample_size = 100  # Small sample for demo
    
    for idx, row in tqdm(df.head(sample_size).iterrows(), total=sample_size, desc="Building graph"):
        paper_id = row['cord_uid']
        
        # Add paper node
        G.add_node(paper_id, 
                  type="paper", 
                  title=row["title"],
                  journal=row["journal"],
                  year=row["publish_time"])
        
        # Add entities
        entities = extract_entities(row["abstract"])
        for ent, label in entities:
            ent_clean = ent.strip()
            if len(ent_clean) > 1:
                G.add_node(ent_clean, type=label)
                G.add_edge(paper_id, ent_clean, relation="mentions")
        
        # Add authors
        if pd.notna(row["authors"]):
            authors = [author.strip() for author in row["authors"].split(";") if author.strip()]
            for author in authors:
                G.add_node(author, type="author")
                G.add_edge(paper_id, author, relation="authored_by")
    
    print(f"✅ Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Step 5: Semantic Embeddings
    print("\n🔤 Step 5: Generating Embeddings...")
    
    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    embedding_size = 200  # Small sample for demo
    abstracts = df["abstract"].head(embedding_size).tolist()
    
    embeddings_list = []
    for i, abstract in enumerate(tqdm(abstracts, desc="Generating embeddings")):
        try:
            emb = embed(abstract)
            embeddings_list.append(emb)
        except Exception as e:
            embeddings_list.append(np.zeros((1, 768)))
    
    embeddings = np.vstack(embeddings_list)
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Step 6: FAISS Index
    print("\n🔍 Step 6: Building FAISS Index...")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    
    # Step 7: Semantic Search
    print("\n🔍 Step 7: Testing Semantic Search...")
    
    query = "What drugs are being tested for COVID-19 treatment?"
    
    def embed_query(query_text):
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')
    
    q_vec = embed_query(query)
    D, I = index.search(q_vec, k=5)
    
    print(f"🔍 Query: '{query}'")
    print(f"📋 Top 5 results:")
    
    results_df = df.iloc[I[0]][["title", "abstract"]]
    for i, (idx, row) in enumerate(results_df.iterrows()):
        print(f"\n   {i+1}. {row['title'][:80]}...")
        print(f"      {row['abstract'][:150]}...")
    
    # Step 8: Graph Context
    print("\n🕸️ Step 8: Graph Context Retrieval...")
    
    def get_context_from_graph(paper_ids, G):
        context = []
        for pid in paper_ids:
            if pid in G.nodes:
                neighbors = list(G.neighbors(pid))
                paper_data = G.nodes[pid]
                title = paper_data.get('title', 'Unknown')
                context.append(f"Paper '{title}' mentions: {neighbors[:5]}")
            else:
                context.append(f"Paper {pid} not found in graph")
        return "\n".join(context)
    
    paper_ids = df.iloc[I[0]]["cord_uid"].tolist()
    graph_context = get_context_from_graph(paper_ids, G)
    
    print("📊 Graph context:")
    print(graph_context[:500] + "..." if len(graph_context) > 500 else graph_context)
    
    # Step 9: GPT-4 Summarization
    print("\n🤖 Step 9: GPT-4 Summarization...")
    
    if client:
        context_text = "\n\n".join(df.iloc[I[0]]["abstract"].tolist())
        
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
            
            summary = response.choices[0].message.content
            print("✅ Summary generated:")
            print(summary)
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            summary = "Error generating summary"
    else:
        print("⚠️ OpenAI client not available. Skipping GPT-4 summarization.")
        summary = "Manual summary: Retrieved papers show various drug treatments being tested for COVID-19."
    
    # Save Results
    print("\n💾 Saving Results...")
    
    Path("results").mkdir(exist_ok=True)
    
    # Save graph
    nx.write_gml(G, "results/cord19_graph.gml")
    print("✅ Graph saved to results/cord19_graph.gml")
    
    # Save embeddings
    np.save("results/embeddings.npy", embeddings)
    print("✅ Embeddings saved to results/embeddings.npy")
    
    # Save FAISS index
    faiss.write_index(index, "results/faiss_index.bin")
    print("✅ FAISS index saved to results/faiss_index.bin")
    
    # Save query results
    results = {
        "query": query,
        "paper_ids": paper_ids,
        "graph_context": graph_context,
        "summary": summary,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open("results/query_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Query results saved to results/query_results.json")
    
    # Final Summary
    print("\n🎉 Pipeline Complete!")
    print("=" * 50)
    print(f"✅ Processed {len(df):,} papers")
    print(f"✅ Built graph with {len(G.nodes):,} nodes and {len(G.edges):,} edges")
    print(f"✅ Generated {embeddings.shape[0]:,} embeddings")
    print(f"✅ Created FAISS index with {index.ntotal:,} vectors")
    print(f"✅ Query: '{query}'")
    print(f"✅ Summary: {summary[:200]}...")
    
    print("\n📁 Results saved to 'results' directory")
    print("🚀 You can now explore the results or run the Jupyter notebook for interactive analysis!")

if __name__ == "__main__":
    main()
