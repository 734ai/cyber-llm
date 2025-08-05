#!/usr/bin/env python3
"""
Embedding Generation Script for Cyber-LLM
Generates vector embeddings from processed text data for semantic search and retrieval.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Required packages not installed. Run: pip install scikit-learn sentence-transformers faiss-cpu")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate and manage embeddings for cybersecurity documents."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedding generator with specified model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def load_processed_data(self, data_dir: Path) -> List[Dict]:
        """Load processed JSON files from data directory."""
        json_files = list(data_dir.glob('**/*.json'))
        documents = []
        
        logger.info(f"Loading {len(json_files)} processed documents")
        
        for json_file in json_files:
            if json_file.name == 'conversion_report.json':
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Split content into chunks for better embedding
                chunks = self._chunk_text(data['content'], chunk_size=512)
                
                for i, chunk in enumerate(chunks):
                    doc = {
                        'id': f"{json_file.stem}_{i}",
                        'source_file': str(json_file),
                        'chunk_index': i,
                        'content': chunk,
                        'metadata': data['metadata']
                    }
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
                
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context preservation."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
                
        return chunks
    
    def generate_embeddings(self, documents: List[Dict]) -> Dict[str, Any]:
        """Generate embeddings for document chunks."""
        logger.info("Generating sentence embeddings...")
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate dense embeddings using SentenceTransformer
        dense_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Generate sparse TF-IDF embeddings
        logger.info("Generating TF-IDF embeddings...")
        sparse_embeddings = self.vectorizer.fit_transform(texts)
        
        # Create FAISS index for fast similarity search
        logger.info("Creating FAISS index...")
        dimension = dense_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(dense_embeddings)
        index.add(dense_embeddings.astype('float32'))
        
        return {
            'documents': documents,
            'dense_embeddings': dense_embeddings,
            'sparse_embeddings': sparse_embeddings,
            'faiss_index': index,
            'vectorizer': self.vectorizer,
            'embedding_dim': dimension
        }
    
    def save_embeddings(self, embedding_data: Dict, output_dir: Path):
        """Save embeddings and metadata to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents metadata
        documents_file = output_dir / 'documents.json'
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_data['documents'], f, indent=2, ensure_ascii=False)
        
        # Save dense embeddings
        embeddings_file = output_dir / 'dense_embeddings.npy'
        np.save(embeddings_file, embedding_data['dense_embeddings'])
        
        # Save FAISS index
        index_file = output_dir / 'faiss_index.bin'
        faiss.write_index(embedding_data['faiss_index'], str(index_file))
        
        # Save TF-IDF vectorizer
        import pickle
        vectorizer_file = output_dir / 'tfidf_vectorizer.pkl'
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(embedding_data['vectorizer'], f)
        
        # Save sparse embeddings
        sparse_file = output_dir / 'sparse_embeddings.npz'
        from scipy.sparse import save_npz
        save_npz(sparse_file, embedding_data['sparse_embeddings'])
        
        # Save metadata
        metadata = {
            'total_documents': len(embedding_data['documents']),
            'embedding_dimension': embedding_data['embedding_dim'],
            'model_name': self.model.get_sentence_embedding_dimension(),
            'creation_timestamp': str(Path().cwd())
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Embeddings saved to {output_dir}")
        return metadata
    
    def generate_document_statistics(self, documents: List[Dict]) -> Dict:
        """Generate statistics about the document corpus."""
        stats = {
            'total_chunks': len(documents),
            'document_types': {},
            'avg_chunk_length': 0,
            'source_files': set(),
        }
        
        total_length = 0
        for doc in documents:
            doc_type = doc['metadata']['source_type']
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            stats['source_files'].add(doc['source_file'])
            total_length += len(doc['content'])
        
        stats['avg_chunk_length'] = total_length // len(documents)
        stats['unique_source_files'] = len(stats['source_files'])
        stats['source_files'] = list(stats['source_files'])
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for Cyber-LLM training data')
    parser.add_argument('--input', required=True, help='Input directory with processed JSON files')
    parser.add_argument('--output', required=True, help='Output directory for embeddings')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--chunk-size', type=int, default=512, help='Text chunk size for embeddings')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = EmbeddingGenerator(model_name=args.model)
    
    # Load processed documents
    input_dir = Path(args.input)
    documents = generator.load_processed_data(input_dir)
    
    if not documents:
        logger.error("No documents found to process!")
        return
    
    # Generate document statistics
    stats = generator.generate_document_statistics(documents)
    logger.info(f"Document Statistics: {json.dumps(stats, indent=2)}")
    
    # Generate embeddings
    embedding_data = generator.generate_embeddings(documents)
    
    # Save embeddings
    output_dir = Path(args.output)
    metadata = generator.save_embeddings(embedding_data, output_dir)
    
    logger.info(f"""
    
Embedding Generation Complete!
=============================
Total document chunks: {metadata['total_documents']}
Embedding dimension: {metadata['embedding_dimension']}
Output directory: {output_dir}
    """)

if __name__ == '__main__':
    main()
