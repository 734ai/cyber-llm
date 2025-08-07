"""
Semantic Memory Networks with Knowledge Graphs for Cybersecurity Concepts
Implements concept relationships and knowledge reasoning
"""
import sqlite3
import json
import uuid
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

@dataclass
class SemanticConcept:
    """Individual semantic concept in the knowledge graph"""
    id: str
    name: str
    concept_type: str  # vulnerability, technique, tool, indicator, etc.
    description: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    source: str  # mitre, cve, custom, etc.

@dataclass
class ConceptRelation:
    """Relationship between semantic concepts"""
    id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: str  # uses, mitigates, exploits, indicates, etc.
    strength: float
    properties: Dict[str, Any]
    created_at: datetime
    evidence: List[str]

class SemanticMemoryNetwork:
    """Advanced semantic memory with knowledge graph capabilities"""
    
    def __init__(self, db_path: str = "data/cognitive/semantic_memory.db"):
        """Initialize semantic memory system"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._knowledge_graph = nx.MultiDiGraph()
        self._concept_cache = {}
        self._load_knowledge_graph()
        
    def _init_database(self):
        """Initialize database schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    concept_type TEXT NOT NULL,
                    description TEXT,
                    properties TEXT,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_relations (
                    id TEXT PRIMARY KEY,
                    source_concept_id TEXT,
                    target_concept_id TEXT,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    properties TEXT,
                    evidence TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_concept_id) REFERENCES semantic_concepts(id),
                    FOREIGN KEY (target_concept_id) REFERENCES semantic_concepts(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_queries (
                    id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_type TEXT,
                    concepts_used TEXT,
                    relations_used TEXT,
                    result TEXT,
                    confidence REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_clusters (
                    id TEXT PRIMARY KEY,
                    cluster_name TEXT NOT NULL,
                    concept_ids TEXT,
                    cluster_properties TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_name ON semantic_concepts(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_type ON semantic_concepts(concept_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON concept_relations(relation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_source ON concept_relations(source_concept_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_target ON concept_relations(target_concept_id)")
    
    def add_concept(self, name: str, concept_type: str, description: str = "",
                   properties: Dict[str, Any] = None, confidence: float = 0.5,
                   source: str = "custom") -> str:
        """Add a new semantic concept to the knowledge graph"""
        try:
            concept_id = str(uuid.uuid4())
            
            concept = SemanticConcept(
                id=concept_id,
                name=name,
                concept_type=concept_type,
                description=description,
                properties=properties or {},
                confidence=confidence,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source=source
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO semantic_concepts (
                        id, name, concept_type, description, properties,
                        confidence, source, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept.id, concept.name, concept.concept_type,
                    concept.description, json.dumps(concept.properties),
                    concept.confidence, concept.source,
                    concept.created_at.isoformat(),
                    concept.updated_at.isoformat()
                ))
            
            # Add to knowledge graph
            self._knowledge_graph.add_node(
                concept_id,
                name=name,
                concept_type=concept_type,
                description=description,
                properties=concept.properties,
                confidence=confidence
            )
            
            # Cache the concept
            self._concept_cache[concept_id] = concept
            
            logger.info(f"Added semantic concept: {name} ({concept_type})")
            return concept_id
            
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
            return ""
    
    def add_relation(self, source_concept_id: str, target_concept_id: str,
                    relation_type: str, strength: float = 0.5,
                    properties: Dict[str, Any] = None,
                    evidence: List[str] = None) -> str:
        """Add a relationship between concepts"""
        try:
            relation_id = str(uuid.uuid4())
            
            relation = ConceptRelation(
                id=relation_id,
                source_concept_id=source_concept_id,
                target_concept_id=target_concept_id,
                relation_type=relation_type,
                strength=strength,
                properties=properties or {},
                created_at=datetime.now(),
                evidence=evidence or []
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO concept_relations (
                        id, source_concept_id, target_concept_id, relation_type,
                        strength, properties, evidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.id, relation.source_concept_id,
                    relation.target_concept_id, relation.relation_type,
                    relation.strength, json.dumps(relation.properties),
                    json.dumps(relation.evidence),
                    relation.created_at.isoformat()
                ))
            
            # Add to knowledge graph
            self._knowledge_graph.add_edge(
                source_concept_id,
                target_concept_id,
                relation_id=relation_id,
                relation_type=relation_type,
                strength=strength,
                properties=relation.properties
            )
            
            logger.info(f"Added relation: {relation_type} ({strength:.2f})")
            return relation_id
            
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return ""
    
    def find_concept(self, name: str = "", concept_type: str = "",
                    properties: Dict[str, Any] = None) -> List[SemanticConcept]:
        """Find concepts matching criteria"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conditions = []
                params = []
                
                if name:
                    conditions.append("name LIKE ?")
                    params.append(f"%{name}%")
                
                if concept_type:
                    conditions.append("concept_type = ?")
                    params.append(concept_type)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                cursor = conn.execute(f"""
                    SELECT * FROM semantic_concepts 
                    WHERE {where_clause}
                    ORDER BY confidence DESC, name
                """, params)
                
                concepts = []
                for row in cursor.fetchall():
                    concept = SemanticConcept(
                        id=row[0],
                        name=row[1],
                        concept_type=row[2],
                        description=row[3] or "",
                        properties=json.loads(row[4]) if row[4] else {},
                        confidence=row[5],
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        source=row[6] or "unknown"
                    )
                    
                    # Filter by properties if specified
                    if properties:
                        matches = all(
                            concept.properties.get(k) == v 
                            for k, v in properties.items()
                        )
                        if matches:
                            concepts.append(concept)
                    else:
                        concepts.append(concept)
                
                logger.info(f"Found {len(concepts)} matching concepts")
                return concepts
                
        except Exception as e:
            logger.error(f"Error finding concepts: {e}")
            return []
    
    def reason_about_threat(self, threat_indicators: List[str]) -> Dict[str, Any]:
        """Perform knowledge-based reasoning about a potential threat"""
        try:
            reasoning_result = {
                'indicators': threat_indicators,
                'matched_concepts': [],
                'inferred_relations': [],
                'threat_assessment': {},
                'recommendations': [],
                'confidence': 0.0
            }
            
            # Find concepts matching the indicators
            matched_concepts = []
            for indicator in threat_indicators:
                concepts = self.find_concept(name=indicator)
                matched_concepts.extend(concepts)
            
            reasoning_result['matched_concepts'] = [
                {
                    'id': c.id,
                    'name': c.name,
                    'type': c.concept_type,
                    'confidence': c.confidence
                } for c in matched_concepts
            ]
            
            # Calculate overall threat confidence
            if matched_concepts:
                avg_confidence = sum(c.confidence for c in matched_concepts) / len(matched_concepts)
                reasoning_result['confidence'] = min(avg_confidence, 1.0)
            
            # Generate threat assessment based on concept types
            threat_types = {}
            for concept in matched_concepts:
                if concept.concept_type not in threat_types:
                    threat_types[concept.concept_type] = 0
                threat_types[concept.concept_type] += concept.confidence
            
            if 'vulnerability' in threat_types and 'technique' in threat_types:
                reasoning_result['threat_assessment']['risk_level'] = 'HIGH'
                reasoning_result['threat_assessment']['rationale'] = 'Vulnerability and attack technique combination detected'
            elif 'malware' in threat_types or 'exploit' in threat_types:
                reasoning_result['threat_assessment']['risk_level'] = 'MEDIUM'
                reasoning_result['threat_assessment']['rationale'] = 'Malicious indicators present'
            else:
                reasoning_result['threat_assessment']['risk_level'] = 'LOW'
                reasoning_result['threat_assessment']['rationale'] = 'Limited threat indicators'
            
            logger.info(f"Threat reasoning complete: {reasoning_result['threat_assessment']['risk_level']} risk")
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Error in threat reasoning: {e}")
            return {'error': str(e)}
    
    def _load_knowledge_graph(self):
        """Load knowledge graph from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load concepts
                cursor = conn.execute("SELECT * FROM semantic_concepts")
                for row in cursor.fetchall():
                    concept_id = row[0]
                    self._knowledge_graph.add_node(
                        concept_id,
                        name=row[1],
                        concept_type=row[2],
                        description=row[3] or "",
                        properties=json.loads(row[4]) if row[4] else {},
                        confidence=row[5]
                    )
                
                # Load relations
                cursor = conn.execute("SELECT * FROM concept_relations")
                for row in cursor.fetchall():
                    self._knowledge_graph.add_edge(
                        row[1],  # source_concept_id
                        row[2],  # target_concept_id
                        relation_id=row[0],
                        relation_type=row[3],
                        strength=row[4],
                        properties=json.loads(row[5]) if row[5] else {}
                    )
                
                logger.info(f"Loaded knowledge graph: {self._knowledge_graph.number_of_nodes()} nodes, {self._knowledge_graph.number_of_edges()} edges")
                
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def _store_knowledge_query(self, query_text: str, query_type: str,
                              concepts_used: List[str], relations_used: List[str],
                              result: Dict[str, Any], confidence: float):
        """Store knowledge query for learning"""
        try:
            query_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO knowledge_queries (
                        id, query_text, query_type, concepts_used,
                        relations_used, result, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, query_text, query_type,
                    json.dumps(concepts_used), json.dumps(relations_used),
                    json.dumps(result), confidence
                ))
                
        except Exception as e:
            logger.error(f"Error storing knowledge query: {e}")
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive semantic memory statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM semantic_concepts")
                stats['total_concepts'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM concept_relations")
                stats['total_relations'] = cursor.fetchone()[0]
                
                # Concept type distribution
                cursor = conn.execute("""
                    SELECT concept_type, COUNT(*) 
                    FROM semantic_concepts 
                    GROUP BY concept_type
                """)
                stats['concept_types'] = dict(cursor.fetchall())
                
                # Relation type distribution
                cursor = conn.execute("""
                    SELECT relation_type, COUNT(*) 
                    FROM concept_relations 
                    GROUP BY relation_type
                """)
                stats['relation_types'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting semantic statistics: {e}")
            return {'error': str(e)}

# Export the main classes
__all__ = ['SemanticMemoryNetwork', 'SemanticConcept', 'ConceptRelation']
