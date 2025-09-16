#!/usr/bin/env python3
"""
CORD-19 GraphRAG Pipeline - Complete Implementation
Following the 8-step process for building a reproducible GraphRAG system
"""

import pandas as pd
import spacy
import networkx as nx
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import os
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Cord19GraphRAG:
    """Complete CORD-19 GraphRAG Pipeline Implementation"""
    
    def __init__(self, openai_api_key=None):
        self.df = None
        self.nlp = None
        self.G = nx.Graph()
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.index = None
        self.client = None
        
        # Initialize OpenAI client if API key provided
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        elif os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI()
    
    def step1_setup(self):
        """STEP 1: Setup & Dataset Loading"""
        print("🚀 STEP 1: Setup & Dataset Loading")
        print("=" * 50)
        
        # Load spaCy model (using web model since sci model not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model: en_core_web_sm")
        except OSError:
            print("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            return False
        
        # Load SciBERT model
        print("📥 Loading SciBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        print("✅ Loaded SciBERT model")
        
        return True
    
    def step2_load_preprocess(self, metadata_path="2020-04-10/metadata.csv"):
        """STEP 2: Load & Preprocess Metadata"""
        print("\n📊 STEP 2: Load & Preprocess Metadata")
        print("=" * 50)
        
        # Load metadata
        print(f"📂 Loading metadata from {metadata_path}...")
        self.df = pd.read_csv(metadata_path)
        print(f"   Total papers: {len(self.df):,}")
        
        # Keep only papers with abstracts
        self.df = self.df.dropna(subset=["abstract"])
        print(f"   Papers with abstracts: {len(self.df):,}")
        
        # Select relevant columns
        self.df = self.df[["cord_uid", "title", "abstract", "authors", "journal", "publish_time"]]
        
        print("📋 Sample data:")
        print(self.df.head())
        
        return True
    
    def step3_entity_extraction(self, sample_size=500):
        """STEP 3: Entity Extraction (SpaCy)"""
        print(f"\n🧬 STEP 3: Entity Extraction (Sample: {sample_size} papers)")
        print("=" * 50)
        
        def extract_entities(text):
            """Extract entities from text using spaCy"""
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        
        # Example on one abstract
        if len(self.df) > 0:
            print("🔍 Example entity extraction:")
            sample_text = self.df['abstract'].iloc[0]
            entities = extract_entities(sample_text)
            print(f"   Text: {sample_text[:200]}...")
            print(f"   Entities found: {entities[:10]}")  # Show first 10 entities
        
        # Store entity extraction function
        self.extract_entities = extract_entities
        
        return True
    
    def step4_graph_construction(self, sample_size=500):
        """STEP 4: Graph Construction (NetworkX)"""
        print(f"\n🕸️ STEP 4: Graph Construction (Sample: {sample_size} papers)")
        print("=" * 50)
        
        # Limit for speed
        sample_df = self.df.head(sample_size)
        
        print(f"📝 Processing {len(sample_df)} papers...")
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Building graph"):
            paper_id = row['cord_uid']
            
            # Add paper node
            self.G.add_node(paper_id, 
                          type="paper", 
                          title=row["title"],
                          journal=row["journal"],
                          year=row["publish_time"])
            
            # Add entities
            entities = self.extract_entities(row["abstract"])
            for ent, label in entities:
                # Clean entity text
                ent_clean = ent.strip()
                if len(ent_clean) > 1:  # Skip single characters
                    self.G.add_node(ent_clean, type=label)
                    self.G.add_edge(paper_id, ent_clean, relation="mentions")
            
            # Add authors
            if pd.notna(row["authors"]):
                authors = [author.strip() for author in row["authors"].split(";") if author.strip()]
                for author in authors:
                    self.G.add_node(author, type="author")
                    self.G.add_edge(paper_id, author, relation="authored_by")
        
        print(f"✅ Graph constructed:")
        print(f"   Nodes: {len(self.G.nodes):,}")
        print(f"   Edges: {len(self.G.edges):,}")
        
        # Show graph statistics
        node_types = {}
        for node, data in self.G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"📊 Node types:")
        for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {node_type}: {count:,}")
        
        return True
    
    def step5_semantic_embeddings(self, sample_size=1000):
        """STEP 5: Semantic Embeddings with SciBERT"""
        print(f"\n🔤 STEP 5: Semantic Embeddings (Sample: {sample_size} abstracts)")
        print("=" * 50)
        
        def embed(text):
            """Generate embeddings using SciBERT"""
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Compute embeddings for abstracts
        abstracts = self.df["abstract"].head(sample_size).tolist()
        print(f"📊 Generating embeddings for {len(abstracts)} abstracts...")
        
        embeddings_list = []
        for i, abstract in enumerate(tqdm(abstracts, desc="Generating embeddings")):
            try:
                emb = embed(abstract)
                embeddings_list.append(emb)
            except Exception as e:
                print(f"   ⚠️ Error processing abstract {i}: {e}")
                # Add zero embedding as fallback
                embeddings_list.append(np.zeros((1, 768)))
        
        self.embeddings = np.vstack(embeddings_list)
        print(f"✅ Generated embeddings: {self.embeddings.shape}")
        
        return True
    
    def step6_faiss_index(self):
        """STEP 6: FAISS Index for Retrieval"""
        print("\n🔍 STEP 6: FAISS Index Construction")
        print("=" * 50)
        
        if self.embeddings is None:
            print("❌ No embeddings found. Run step 5 first.")
            return False
        
        dim = self.embeddings.shape[1]
        print(f"📊 Building FAISS index with dimension {dim}")
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        
        # Test search
        print("🧪 Testing search functionality...")
        query = "What drugs are being tested for COVID-19 treatment?"
        
        def embed_query(query_text):
            """Embed query text"""
            inputs = self.tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')
        
        q_vec = embed_query(query)
        D, I = self.index.search(q_vec, k=5)
        
        print(f"🔍 Query: '{query}'")
        print(f"📋 Top 5 results:")
        
        results_df = self.df.iloc[I[0]][["title", "abstract"]]
        for i, (idx, row) in enumerate(results_df.iterrows()):
            print(f"   {i+1}. {row['title'][:100]}...")
            print(f"      {row['abstract'][:200]}...")
            print()
        
        return True
    
    def step7_graphrag_retrieval(self, paper_ids):
        """STEP 7: GraphRAG Retrieval"""
        print("\n🕸️ STEP 7: GraphRAG Retrieval")
        print("=" * 50)
        
        def get_context_from_graph(paper_ids, G):
            """Get graph context for retrieved papers"""
            context = []
            for pid in paper_ids:
                neighbors = list(G.neighbors(pid))
                paper_data = G.nodes[pid]
                title = paper_data.get('title', 'Unknown')
                
                context.append(f"Paper '{title}' (ID: {pid}) mentions: {neighbors[:10]}")  # Limit to first 10 neighbors
            return "\n".join(context)
        
        graph_context = get_context_from_graph(paper_ids, self.G)
        print("📊 Graph context retrieved:")
        print(graph_context[:500] + "..." if len(graph_context) > 500 else graph_context)
        
        return graph_context
    
    def step8_gpt4_summarization(self, query, paper_ids, graph_context):
        """STEP 8: GPT-4 Summarization with Citations"""
        print("\n🤖 STEP 8: GPT-4 Summarization")
        print("=" * 50)
        
        if not self.client:
            print("❌ OpenAI client not initialized. Please provide API key.")
            return None
        
        # Get abstracts for retrieved papers
        context_text = "\n\n".join(self.df.iloc[paper_ids]["abstract"].tolist())
        
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
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            print("✅ Summary generated:")
            print(summary)
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return None
    
    def run_complete_pipeline(self, sample_size=500, embedding_size=1000, query="What drugs are being tested for COVID-19 treatment?"):
        """Run the complete GraphRAG pipeline"""
        print("🚀 CORD-19 GraphRAG Pipeline - Complete Run")
        print("=" * 60)
        
        # Step 1: Setup
        if not self.step1_setup():
            return False
        
        # Step 2: Load and preprocess
        if not self.step2_load_preprocess():
            return False
        
        # Step 3: Entity extraction
        if not self.step3_entity_extraction(sample_size):
            return False
        
        # Step 4: Graph construction
        if not self.step4_graph_construction(sample_size):
            return False
        
        # Step 5: Semantic embeddings
        if not self.step5_semantic_embeddings(embedding_size):
            return False
        
        # Step 6: FAISS index
        if not self.step6_faiss_index():
            return False
        
        # Step 7 & 8: Retrieval and summarization
        print(f"\n🔍 Running query: '{query}'")
        
        # Get top results
        q_vec = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = self.model(**q_vec)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')
        
        D, I = self.index.search(query_embedding, k=5)
        paper_ids = I[0]
        
        # Get graph context
        graph_context = self.step7_graphrag_retrieval(paper_ids)
        
        # Generate summary
        summary = self.step8_gpt4_summarization(query, paper_ids, graph_context)
        
        # Save results
        self.save_results(query, paper_ids, graph_context, summary)
        
        return True
    
    def save_results(self, query, paper_ids, graph_context, summary):
        """Save pipeline results"""
        print("\n💾 Saving Results")
        print("=" * 30)
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Save graph
        nx.write_gml(self.G, "results/cord19_graph.gml")
        print("✅ Graph saved to results/cord19_graph.gml")
        
        # Save embeddings
        np.save("results/embeddings.npy", self.embeddings)
        print("✅ Embeddings saved to results/embeddings.npy")
        
        # Save FAISS index
        faiss.write_index(self.index, "results/faiss_index.bin")
        print("✅ FAISS index saved to results/faiss_index.bin")
        
        # Save query results
        results = {
            "query": query,
            "paper_ids": paper_ids.tolist(),
            "graph_context": graph_context,
            "summary": summary,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open("results/query_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("✅ Query results saved to results/query_results.json")
        
        # Save metadata
        self.df.to_csv("results/processed_metadata.csv", index=False)
        print("✅ Processed metadata saved to results/processed_metadata.csv")

def main():
    """Main function to run the pipeline"""
    print("🚀 CORD-19 GraphRAG Pipeline")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = Cord19GraphRAG()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline(
        sample_size=500,  # Number of papers for graph construction
        embedding_size=1000,  # Number of abstracts for embeddings
        query="What drugs are being tested for COVID-19 treatment?"
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("📁 Check the 'results' directory for outputs")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

if __name__ == "__main__":
    main()
