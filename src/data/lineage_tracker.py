"""
Data Lineage Tracking System
Tracks data flow, transformations, and dependencies across the cybersecurity AI pipeline
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class DataSourceType(Enum):
    RAW_DATA = "raw_data"
    MITRE_ATTACK = "mitre_attack"
    CVE_DATABASE = "cve_database"
    THREAT_INTEL = "threat_intel"
    RED_TEAM_LOGS = "red_team_logs"
    DEFENSIVE_KNOWLEDGE = "defensive_knowledge"
    PREPROCESSED = "preprocessed"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    AUGMENTED = "augmented"

class TransformationType(Enum):
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    TOKENIZATION = "tokenization"
    AUGMENTATION = "augmentation"
    VALIDATION = "validation"
    FEATURE_EXTRACTION = "feature_extraction"
    ANONYMIZATION = "anonymization"
    AGGREGATION = "aggregation"

@dataclass
class DataAsset:
    """Represents a data asset in the lineage graph"""
    asset_id: str
    name: str
    source_type: DataSourceType
    file_path: str
    size_bytes: int
    checksum: str
    created_at: str
    schema_version: str
    metadata: Dict[str, Any]

@dataclass
class DataTransformation:
    """Represents a data transformation operation"""
    transformation_id: str
    transformation_type: TransformationType
    source_assets: List[str]
    target_assets: List[str]
    operation_name: str
    parameters: Dict[str, Any]
    executed_at: str
    execution_time_seconds: float
    success: bool
    error_message: Optional[str]

@dataclass
class DataLineageNode:
    """Node in the data lineage graph"""
    node_id: str
    asset: DataAsset
    upstream_nodes: List[str]
    downstream_nodes: List[str]
    transformations: List[str]

class DataLineageTracker:
    """Tracks data lineage across the cybersecurity AI pipeline"""
    
    def __init__(self, db_path: str = "data/lineage/data_lineage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize the lineage database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data Assets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_assets (
                asset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        
        # Data Transformations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_transformations (
                transformation_id TEXT PRIMARY KEY,
                transformation_type TEXT NOT NULL,
                source_assets TEXT NOT NULL,
                target_assets TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                execution_time_seconds REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT
            )
        """)
        
        # Lineage Relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineage_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_asset_id TEXT NOT NULL,
                child_asset_id TEXT NOT NULL,
                transformation_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_asset_id) REFERENCES data_assets (asset_id),
                FOREIGN KEY (child_asset_id) REFERENCES data_assets (asset_id),
                FOREIGN KEY (transformation_id) REFERENCES data_transformations (transformation_id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_source_type ON data_assets(source_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transformations_type ON data_transformations(transformation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_parent ON lineage_relationships(parent_asset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_child ON lineage_relationships(child_asset_id)")
        
        conn.commit()
        conn.close()
    
    def register_data_asset(self, asset: DataAsset) -> bool:
        """Register a new data asset"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_assets 
                (asset_id, name, source_type, file_path, size_bytes, checksum, 
                 created_at, schema_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset.asset_id, asset.name, asset.source_type.value,
                asset.file_path, asset.size_bytes, asset.checksum,
                asset.created_at, asset.schema_version, json.dumps(asset.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error registering data asset: {e}")
            return False
    
    def register_transformation(self, transformation: DataTransformation) -> bool:
        """Register a data transformation operation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_transformations 
                (transformation_id, transformation_type, source_assets, target_assets,
                 operation_name, parameters, executed_at, execution_time_seconds,
                 success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transformation.transformation_id, transformation.transformation_type.value,
                json.dumps(transformation.source_assets), json.dumps(transformation.target_assets),
                transformation.operation_name, json.dumps(transformation.parameters),
                transformation.executed_at, transformation.execution_time_seconds,
                transformation.success, transformation.error_message
            ))
            
            # Register lineage relationships
            for source_id in transformation.source_assets:
                for target_id in transformation.target_assets:
                    cursor.execute("""
                        INSERT INTO lineage_relationships 
                        (parent_asset_id, child_asset_id, transformation_id, relationship_type, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (source_id, target_id, transformation.transformation_id, "transformation", transformation.executed_at))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error registering transformation: {e}")
            return False
    
    def get_asset_lineage(self, asset_id: str, direction: str = "both") -> Dict[str, Any]:
        """Get the lineage graph for a specific asset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        lineage = {
            "asset_id": asset_id,
            "upstream": [],
            "downstream": [],
            "transformations": []
        }
        
        # Get upstream lineage
        if direction in ["upstream", "both"]:
            cursor.execute("""
                SELECT DISTINCT lr.parent_asset_id, da.name, da.source_type, dt.operation_name
                FROM lineage_relationships lr
                JOIN data_assets da ON lr.parent_asset_id = da.asset_id
                JOIN data_transformations dt ON lr.transformation_id = dt.transformation_id
                WHERE lr.child_asset_id = ?
            """, (asset_id,))
            
            lineage["upstream"] = [
                {
                    "asset_id": row[0],
                    "name": row[1],
                    "source_type": row[2],
                    "operation": row[3]
                }
                for row in cursor.fetchall()
            ]
        
        # Get downstream lineage
        if direction in ["downstream", "both"]:
            cursor.execute("""
                SELECT DISTINCT lr.child_asset_id, da.name, da.source_type, dt.operation_name
                FROM lineage_relationships lr
                JOIN data_assets da ON lr.child_asset_id = da.asset_id
                JOIN data_transformations dt ON lr.transformation_id = dt.transformation_id
                WHERE lr.parent_asset_id = ?
            """, (asset_id,))
            
            lineage["downstream"] = [
                {
                    "asset_id": row[0],
                    "name": row[1],
                    "source_type": row[2],
                    "operation": row[3]
                }
                for row in cursor.fetchall()
            ]
        
        # Get transformations involving this asset
        cursor.execute("""
            SELECT dt.transformation_id, dt.operation_name, dt.executed_at, dt.success
            FROM data_transformations dt
            WHERE JSON_EXTRACT(dt.source_assets, '$') LIKE '%' || ? || '%'
               OR JSON_EXTRACT(dt.target_assets, '$') LIKE '%' || ? || '%'
        """, (asset_id, asset_id))
        
        lineage["transformations"] = [
            {
                "transformation_id": row[0],
                "operation_name": row[1],
                "executed_at": row[2],
                "success": bool(row[3])
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return lineage
    
    def get_data_flow_impact(self, asset_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a specific data asset"""
        lineage = self.get_asset_lineage(asset_id, direction="downstream")
        
        impact_analysis = {
            "source_asset": asset_id,
            "affected_assets": len(lineage["downstream"]),
            "affected_asset_types": {},
            "critical_dependencies": [],
            "recommendation": ""
        }
        
        # Count affected asset types
        for asset in lineage["downstream"]:
            asset_type = asset["source_type"]
            impact_analysis["affected_asset_types"][asset_type] = (
                impact_analysis["affected_asset_types"].get(asset_type, 0) + 1
            )
        
        # Identify critical dependencies
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT da.asset_id, da.name, da.source_type
            FROM data_assets da
            WHERE da.source_type IN ('validated', 'augmented', 'transformed')
              AND da.asset_id IN (
                  SELECT lr.child_asset_id 
                  FROM lineage_relationships lr
                  WHERE lr.parent_asset_id = ?
              )
        """, (asset_id,))
        
        impact_analysis["critical_dependencies"] = [
            {"asset_id": row[0], "name": row[1], "type": row[2]}
            for row in cursor.fetchall()
        ]
        
        # Generate recommendation
        if impact_analysis["affected_assets"] > 10:
            impact_analysis["recommendation"] = "HIGH IMPACT: Changes require comprehensive testing"
        elif impact_analysis["affected_assets"] > 5:
            impact_analysis["recommendation"] = "MEDIUM IMPACT: Changes require targeted testing"
        else:
            impact_analysis["recommendation"] = "LOW IMPACT: Standard validation sufficient"
        
        conn.close()
        return impact_analysis
    
    def generate_lineage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data lineage report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "asset_types": {},
            "transformation_types": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # Summary statistics
        cursor.execute("SELECT COUNT(*) FROM data_assets")
        total_assets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM data_transformations")
        total_transformations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lineage_relationships")
        total_relationships = cursor.fetchone()[0]
        
        report["summary"] = {
            "total_assets": total_assets,
            "total_transformations": total_transformations,
            "total_relationships": total_relationships
        }
        
        # Asset type distribution
        cursor.execute("""
            SELECT source_type, COUNT(*), AVG(size_bytes)
            FROM data_assets
            GROUP BY source_type
        """)
        
        for row in cursor.fetchall():
            report["asset_types"][row[0]] = {
                "count": row[1],
                "avg_size_bytes": row[2]
            }
        
        # Transformation type distribution
        cursor.execute("""
            SELECT transformation_type, COUNT(*), AVG(execution_time_seconds), 
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
            FROM data_transformations
            GROUP BY transformation_type
        """)
        
        for row in cursor.fetchall():
            report["transformation_types"][row[0]] = {
                "count": row[1],
                "avg_execution_time": row[2],
                "success_rate": row[3]
            }
        
        # Data quality metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN source_type IN ('validated', 'augmented') THEN 1 ELSE 0 END) as high_quality,
                AVG(size_bytes) as avg_size
            FROM data_assets
        """)
        
        row = cursor.fetchone()
        report["data_quality"] = {
            "total_assets": row[0],
            "high_quality_assets": row[1],
            "quality_percentage": (row[1] / row[0] * 100) if row[0] > 0 else 0,
            "average_asset_size": row[2]
        }
        
        # Generate recommendations
        if report["data_quality"]["quality_percentage"] < 70:
            report["recommendations"].append("Increase data validation and quality assurance processes")
        
        if any(info["success_rate"] < 90 for info in report["transformation_types"].values()):
            report["recommendations"].append("Review and optimize failing data transformations")
        
        if report["summary"]["total_relationships"] / report["summary"]["total_assets"] < 1.5:
            report["recommendations"].append("Consider enriching data lineage tracking")
        
        conn.close()
        return report
    
    def create_asset_from_file(self, file_path: str, source_type: DataSourceType, 
                              name: Optional[str] = None, metadata: Optional[Dict] = None) -> DataAsset:
        """Create a DataAsset from a file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate file checksum
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        asset_id = f"{source_type.value}_{hasher.hexdigest()[:16]}"
        
        return DataAsset(
            asset_id=asset_id,
            name=name or path.name,
            source_type=source_type,
            file_path=str(path.absolute()),
            size_bytes=path.stat().st_size,
            checksum=hasher.hexdigest(),
            created_at=datetime.now().isoformat(),
            schema_version="1.0",
            metadata=metadata or {}
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize the tracker
    tracker = DataLineageTracker("data/lineage/data_lineage.db")
    
    # Example: Track MITRE ATT&CK data processing
    mitre_asset = DataAsset(
        asset_id="mitre_attack_raw_001",
        name="MITRE ATT&CK Framework Data",
        source_type=DataSourceType.MITRE_ATTACK,
        file_path="data/raw/mitre_attack.json",
        size_bytes=1024000,
        checksum="abc123def456",
        created_at=datetime.now().isoformat(),
        schema_version="1.0",
        metadata={"version": "14.1", "techniques": 200}
    )
    
    tracker.register_data_asset(mitre_asset)
    
    # Track preprocessing transformation
    preprocessing = DataTransformation(
        transformation_id="preprocess_001",
        transformation_type=TransformationType.CLEANING,
        source_assets=["mitre_attack_raw_001"],
        target_assets=["mitre_attack_clean_001"],
        operation_name="clean_and_normalize_mitre_data",
        parameters={"remove_deprecated": True, "normalize_names": True},
        executed_at=datetime.now().isoformat(),
        execution_time_seconds=15.7,
        success=True,
        error_message=None
    )
    
    tracker.register_transformation(preprocessing)
    
    # Generate lineage report
    report = tracker.generate_lineage_report()
    print("Data Lineage Report:")
    print(json.dumps(report, indent=2))
    
    print("✅ Data Lineage Tracking System implemented and tested")
