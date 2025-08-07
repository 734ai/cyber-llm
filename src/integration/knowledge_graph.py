"""
Knowledge Graph Integration for Cyber-LLM
Real-time threat intelligence and cybersecurity knowledge management

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
import requests
import feedparser
from bs4 import BeautifulSoup

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..memory.persistent_memory import PersistentMemoryManager

class EntityType(Enum):
    """Knowledge graph entity types"""
    VULNERABILITY = "vulnerability"
    THREAT_ACTOR = "threat_actor"
    MALWARE = "malware"
    ATTACK_TECHNIQUE = "attack_technique"
    INDICATOR = "indicator"
    ASSET = "asset"
    ORGANIZATION = "organization"
    CAMPAIGN = "campaign"
    TOOL = "tool"
    MITIGATION = "mitigation"

class RelationType(Enum):
    """Knowledge graph relationship types"""
    EXPLOITS = "exploits"
    MITIGATES = "mitigates"
    TARGETS = "targets"
    USES = "uses"
    ATTRIBUTED_TO = "attributed_to"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    DETECTS = "detects"
    IMPLEMENTS = "implements"
    COMMUNICATES_WITH = "communicates_with"

class ConfidenceLevel(Enum):
    """Confidence levels for knowledge assertions"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class KnowledgeEntity:
    """Knowledge graph entity"""
    entity_id: str
    entity_type: EntityType
    name: str
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    confidence: float = 0.8
    
    # Relationships
    relationships: List['KnowledgeRelationship'] = field(default_factory=list)
    
    # Tags and classification
    tags: Set[str] = field(default_factory=set)
    classification: Optional[str] = None

@dataclass
class KnowledgeRelationship:
    """Knowledge graph relationship"""
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: RelationType
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.8
    source: Optional[str] = None
    
    # Temporal aspects
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

@dataclass
class ThreatIntelligenceData:
    """Threat intelligence data structure"""
    intel_id: str
    title: str
    description: str
    
    # Classification
    threat_type: str
    severity: str
    confidence: ConfidenceLevel
    
    # Temporal information
    discovered_at: datetime
    published_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Indicators
    indicators: List[Dict[str, Any]] = field(default_factory=list)
    
    # Attribution
    threat_actors: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    
    # Source information
    source: str
    source_reliability: str
    
    # References
    references: List[str] = field(default_factory=list)
    
    # Structured data
    mitre_techniques: List[str] = field(default_factory=list)
    affected_products: List[str] = field(default_factory=list)

class CyberKnowledgeGraph:
    """Comprehensive cybersecurity knowledge graph"""
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 memory_manager: PersistentMemoryManager,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.memory_manager = memory_manager
        self.logger = logger or CyberLLMLogger(name="knowledge_graph")
        
        # Graph database connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # In-memory graph for fast operations
        self.graph = nx.MultiDiGraph()
        
        # Entity and relationship tracking
        self.entities = {}
        self.relationships = {}
        
        # Intelligence sources
        self.threat_intel_sources = {}
        self.cve_sources = {}
        self.news_sources = {}
        
        # Update tracking
        self.last_update = {}
        self.update_frequencies = {}
        
        # Initialize knowledge graph
        asyncio.create_task(self._initialize_knowledge_graph())
        
        self.logger.info("Cyber Knowledge Graph initialized")
    
    async def _initialize_knowledge_graph(self):
        """Initialize knowledge graph with base data"""
        
        try:
            # Create database constraints and indexes
            await self._create_database_schema()
            
            # Load base cybersecurity ontology
            await self._load_base_ontology()
            
            # Initialize threat intelligence sources
            await self._initialize_threat_intel_sources()
            
            # Start periodic updates
            asyncio.create_task(self._periodic_updates())
            
            self.logger.info("Knowledge graph initialization completed")
            
        except Exception as e:
            self.logger.error("Knowledge graph initialization failed", error=str(e))
    
    async def add_entity(self, entity: KnowledgeEntity) -> bool:
        """Add entity to knowledge graph"""
        
        try:
            # Store in Neo4j
            with self.driver.session() as session:
                query = f"""
                CREATE (e:{entity.entity_type.value.title()} {{
                    entity_id: $entity_id,
                    name: $name,
                    properties: $properties,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    source: $source,
                    confidence: $confidence,
                    tags: $tags,
                    classification: $classification
                }})
                """
                
                session.run(query, {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "properties": json.dumps(entity.properties),
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat(),
                    "source": entity.source,
                    "confidence": entity.confidence,
                    "tags": list(entity.tags),
                    "classification": entity.classification
                })
            
            # Store in memory
            self.entities[entity.entity_id] = entity
            self.graph.add_node(entity.entity_id, **entity.properties)
            
            self.logger.info("Entity added to knowledge graph",
                           entity_id=entity.entity_id,
                           entity_type=entity.entity_type.value)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to add entity", error=str(e))
            return False
    
    async def add_relationship(self, relationship: KnowledgeRelationship) -> bool:
        """Add relationship to knowledge graph"""
        
        try:
            # Store in Neo4j
            with self.driver.session() as session:
                query = f"""
                MATCH (source {{entity_id: $source_entity}})
                MATCH (target {{entity_id: $target_entity}})
                CREATE (source)-[r:{relationship.relationship_type.value.upper()} {{
                    relationship_id: $relationship_id,
                    properties: $properties,
                    created_at: $created_at,
                    confidence: $confidence,
                    source: $source,
                    valid_from: $valid_from,
                    valid_until: $valid_until
                }}]->(target)
                """
                
                session.run(query, {
                    "source_entity": relationship.source_entity,
                    "target_entity": relationship.target_entity,
                    "relationship_id": relationship.relationship_id,
                    "properties": json.dumps(relationship.properties),
                    "created_at": relationship.created_at.isoformat(),
                    "confidence": relationship.confidence,
                    "source": relationship.source,
                    "valid_from": relationship.valid_from.isoformat() if relationship.valid_from else None,
                    "valid_until": relationship.valid_until.isoformat() if relationship.valid_until else None
                })
            
            # Store in memory
            self.relationships[relationship.relationship_id] = relationship
            self.graph.add_edge(
                relationship.source_entity,
                relationship.target_entity,
                key=relationship.relationship_id,
                relationship_type=relationship.relationship_type.value,
                **relationship.properties
            )
            
            self.logger.info("Relationship added to knowledge graph",
                           relationship_id=relationship.relationship_id,
                           relationship_type=relationship.relationship_type.value)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to add relationship", error=str(e))
            return False
    
    async def query_entities(self, 
                           entity_type: Optional[EntityType] = None,
                           properties: Optional[Dict[str, Any]] = None,
                           tags: Optional[Set[str]] = None) -> List[KnowledgeEntity]:
        """Query entities from knowledge graph"""
        
        try:
            # Build query
            conditions = []
            params = {}
            
            if entity_type:
                label = entity_type.value.title()
            else:
                label = ""
            
            if properties:
                for key, value in properties.items():
                    conditions.append(f"e.properties CONTAINS $prop_{key}")
                    params[f"prop_{key}"] = json.dumps({key: value})
            
            if tags:
                for i, tag in enumerate(tags):
                    conditions.append(f"$tag_{i} IN e.tags")
                    params[f"tag_{i}"] = tag
            
            where_clause = " AND ".join(conditions) if conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            query = f"""
            MATCH (e{':' + label if label else ''})
            {where_clause}
            RETURN e
            """
            
            # Execute query
            with self.driver.session() as session:
                result = session.run(query, params)
                
                entities = []
                for record in result:
                    node = record["e"]
                    entity = KnowledgeEntity(
                        entity_id=node["entity_id"],
                        entity_type=EntityType(node.labels),
                        name=node["name"],
                        properties=json.loads(node.get("properties", "{}")),
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node["updated_at"]),
                        source=node.get("source"),
                        confidence=node.get("confidence", 0.8),
                        tags=set(node.get("tags", [])),
                        classification=node.get("classification")
                    )
                    entities.append(entity)
                
                return entities
            
        except Exception as e:
            self.logger.error("Entity query failed", error=str(e))
            return []
    
    async def find_paths(self, 
                        source_entity: str, 
                        target_entity: str,
                        max_depth: int = 3) -> List[List[str]]:
        """Find paths between entities"""
        
        try:
            # Use NetworkX for efficient path finding
            if self.graph.has_node(source_entity) and self.graph.has_node(target_entity):
                paths = list(nx.all_simple_paths(
                    self.graph, 
                    source_entity, 
                    target_entity, 
                    cutoff=max_depth
                ))
                return paths
            
            return []
            
        except Exception as e:
            self.logger.error("Path finding failed", error=str(e))
            return []
    
    async def get_entity_neighbors(self, entity_id: str, relationship_types: Optional[List[RelationType]] = None) -> List[KnowledgeEntity]:
        """Get neighboring entities"""
        
        try:
            neighbors = []
            
            if entity_id in self.graph:
                for neighbor in self.graph.neighbors(entity_id):
                    if relationship_types:
                        # Check if any edge has the required relationship type
                        edges = self.graph[entity_id][neighbor]
                        for edge_data in edges.values():
                            if edge_data.get('relationship_type') in [rt.value for rt in relationship_types]:
                                if neighbor in self.entities:
                                    neighbors.append(self.entities[neighbor])
                                break
                    else:
                        if neighbor in self.entities:
                            neighbors.append(self.entities[neighbor])
            
            return neighbors
            
        except Exception as e:
            self.logger.error("Failed to get entity neighbors", error=str(e))
            return []

class ThreatIntelligenceAggregator:
    """Aggregates threat intelligence from multiple sources"""
    
    def __init__(self, 
                 knowledge_graph: CyberKnowledgeGraph,
                 logger: Optional[CyberLLMLogger] = None):
        
        self.knowledge_graph = knowledge_graph
        self.logger = logger or CyberLLMLogger(name="threat_intel")
        
        # Intelligence sources
        self.sources = {
            "cve": {
                "url": "https://cve.mitre.org/data/downloads/",
                "update_frequency": timedelta(hours=6)
            },
            "mitre_attack": {
                "url": "https://attack.mitre.org/",
                "update_frequency": timedelta(days=1)
            },
            "threat_feeds": []
        }
        
        # Processing state
        self.last_updates = {}
        self.processing_queue = asyncio.Queue()
        
        # Start processing worker
        asyncio.create_task(self._processing_worker())
        
        self.logger.info("Threat Intelligence Aggregator initialized")
    
    async def aggregate_cve_data(self) -> int:
        """Aggregate CVE data from MITRE"""
        
        try:
            self.logger.info("Starting CVE data aggregation")
            
            # Fetch CVE JSON feed
            cve_url = "https://cve.mitre.org/data/downloads/allitems.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(cve_url) as response:
                    if response.status == 200:
                        cve_data = await response.json()
                    else:
                        raise Exception(f"Failed to fetch CVE data: {response.status}")
            
            processed_count = 0
            
            # Process CVE entries
            for cve_item in cve_data.get("CVE_Items", []):
                cve_id = cve_item["cve"]["CVE_data_meta"]["ID"]
                
                # Create CVE entity
                entity = KnowledgeEntity(
                    entity_id=cve_id,
                    entity_type=EntityType.VULNERABILITY,
                    name=cve_id,
                    properties={
                        "description": cve_item["cve"]["description"]["description_data"][0]["value"],
                        "published_date": cve_item.get("publishedDate"),
                        "modified_date": cve_item.get("lastModifiedDate"),
                        "cvss_score": self._extract_cvss_score(cve_item),
                        "severity": self._determine_severity(cve_item),
                        "affected_products": self._extract_affected_products(cve_item)
                    },
                    source="mitre_cve",
                    confidence=0.95
                )
                
                await self.knowledge_graph.add_entity(entity)
                processed_count += 1
                
                # Add relationships to affected products
                for product in entity.properties.get("affected_products", []):
                    # Create or get product entity
                    product_entity = await self._get_or_create_product_entity(product)
                    
                    # Create vulnerability relationship
                    relationship = KnowledgeRelationship(
                        relationship_id=f"{cve_id}_affects_{product_entity.entity_id}",
                        source_entity=cve_id,
                        target_entity=product_entity.entity_id,
                        relationship_type=RelationType.TARGETS,
                        confidence=0.9,
                        source="mitre_cve"
                    )
                    
                    await self.knowledge_graph.add_relationship(relationship)
            
            self.last_updates["cve"] = datetime.now()
            
            self.logger.info("CVE data aggregation completed",
                           processed_count=processed_count)
            
            return processed_count
            
        except Exception as e:
            self.logger.error("CVE data aggregation failed", error=str(e))
            return 0
    
    async def aggregate_mitre_attack(self) -> int:
        """Aggregate MITRE ATT&CK framework data"""
        
        try:
            self.logger.info("Starting MITRE ATT&CK data aggregation")
            
            # MITRE ATT&CK STIX data
            attack_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(attack_url) as response:
                    if response.status == 200:
                        attack_data = await response.json()
                    else:
                        raise Exception(f"Failed to fetch MITRE ATT&CK data: {response.status}")
            
            processed_count = 0
            
            # Process STIX objects
            for stix_object in attack_data.get("objects", []):
                if stix_object["type"] == "attack-pattern":
                    # Create technique entity
                    technique_id = stix_object.get("external_references", [{}])[0].get("external_id", "")
                    
                    entity = KnowledgeEntity(
                        entity_id=technique_id,
                        entity_type=EntityType.ATTACK_TECHNIQUE,
                        name=stix_object["name"],
                        properties={
                            "description": stix_object.get("description", ""),
                            "kill_chain_phases": [phase["phase_name"] for phase in stix_object.get("kill_chain_phases", [])],
                            "platforms": stix_object.get("x_mitre_platforms", []),
                            "tactics": [ref["external_id"] for ref in stix_object.get("external_references", []) if ref.get("source_name") == "mitre-attack"]
                        },
                        source="mitre_attack",
                        confidence=0.98
                    )
                    
                    await self.knowledge_graph.add_entity(entity)
                    processed_count += 1
            
            self.last_updates["mitre_attack"] = datetime.now()
            
            self.logger.info("MITRE ATT&CK data aggregation completed",
                           processed_count=processed_count)
            
            return processed_count
            
        except Exception as e:
            self.logger.error("MITRE ATT&CK data aggregation failed", error=str(e))
            return 0
    
    async def _processing_worker(self):
        """Background worker for processing intelligence data"""
        
        while True:
            try:
                # Check for scheduled updates
                for source, config in self.sources.items():
                    last_update = self.last_updates.get(source)
                    update_frequency = config.get("update_frequency")
                    
                    if not last_update or (datetime.now() - last_update) > update_frequency:
                        if source == "cve":
                            await self.aggregate_cve_data()
                        elif source == "mitre_attack":
                            await self.aggregate_mitre_attack()
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error("Intelligence processing worker error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Factory functions
def create_knowledge_graph(neo4j_uri: str,
                         neo4j_user: str,
                         neo4j_password: str,
                         memory_manager: PersistentMemoryManager,
                         **kwargs) -> CyberKnowledgeGraph:
    """Create cyber knowledge graph"""
    return CyberKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password, memory_manager, **kwargs)

def create_threat_intelligence_aggregator(knowledge_graph: CyberKnowledgeGraph,
                                        **kwargs) -> ThreatIntelligenceAggregator:
    """Create threat intelligence aggregator"""
    return ThreatIntelligenceAggregator(knowledge_graph, **kwargs)
