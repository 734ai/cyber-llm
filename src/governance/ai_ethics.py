"""
AI Ethics and Responsible AI Framework for Cyber-LLM
Comprehensive ethical AI implementation with bias monitoring, fairness, and transparency

Author: Muzan Sano <sanosensei36@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import sqlite3
from collections import defaultdict

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory
from ..learning.constitutional_ai import ConstitutionalAIManager

class EthicsFramework(Enum):
    """Supported AI ethics frameworks"""
    IEEE_ETHICALLY_ALIGNED = "ieee_ethically_aligned"
    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    RESPONSIBLE_AI_MICROSOFT = "microsoft_responsible_ai"
    PARTNERSHIP_ON_AI = "partnership_on_ai"

class BiasType(Enum):
    """Types of bias to monitor"""
    DEMOGRAPHIC = "demographic"
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"
    HISTORICAL = "historical"
    CONFIRMATION = "confirmation"

class FairnessMetric(Enum):
    """Fairness metrics"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

class TransparencyLevel(Enum):
    """Model transparency levels"""
    BLACK_BOX = "black_box"
    LIMITED_EXPLANATION = "limited_explanation"
    FEATURE_IMPORTANCE = "feature_importance"
    RULE_BASED = "rule_based"
    FULL_TRANSPARENCY = "full_transparency"

@dataclass
class BiasAssessment:
    """Bias assessment result"""
    assessment_id: str
    model_id: str
    assessment_date: datetime
    
    # Bias metrics by type
    bias_scores: Dict[BiasType, float]
    fairness_metrics: Dict[FairnessMetric, float]
    
    # Demographic analysis
    demographic_groups: List[str]
    performance_by_group: Dict[str, Dict[str, float]]
    
    # Assessment details
    assessment_method: str
    confidence_level: float
    recommendations: List[str]
    
    # Overall assessment
    bias_risk_level: str  # low, medium, high, critical
    fairness_compliance: bool
    requires_intervention: bool

@dataclass
class ExplainabilityReport:
    """Model explainability report"""
    report_id: str
    model_id: str
    generated_at: datetime
    
    # Transparency metrics
    transparency_level: TransparencyLevel
    explainability_score: float  # 0-1
    
    # Feature importance
    global_feature_importance: Dict[str, float]
    local_explanations_available: bool
    
    # Explanation methods used
    explanation_methods: List[str]  # SHAP, LIME, attention weights, etc.
    
    # User comprehension
    explanation_quality: Dict[str, float]  # clarity, completeness, actionability
    user_satisfaction_score: Optional[float]

@dataclass
class EthicsViolation:
    """Ethics violation record"""
    violation_id: str
    model_id: str
    violation_type: str
    severity: str  # low, medium, high, critical
    
    description: str
    evidence: Dict[str, Any]
    detected_at: datetime
    
    # Resolution tracking
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    resolution_plan: Optional[str] = None
    resolved_at: Optional[datetime] = None

class AIEthicsManager:
    """Comprehensive AI ethics and responsible AI management"""
    
    def __init__(self, 
                 config_path: str = "configs/ethics_config.yaml",
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="ai_ethics")
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.constitutional_ai = ConstitutionalAIManager()
        self.bias_assessments = {}
        self.explainability_reports = {}
        self.ethics_violations = []
        
        # Database for ethics tracking
        self.db_path = Path("data/ai_ethics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ethics framework
        asyncio.create_task(self._initialize_ethics_system())
        
        self.logger.info("AI Ethics manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ethics configuration"""
        
        default_config = {
            "ethics_frameworks": ["EU_AI_ACT", "NIST_AI_RMF"],
            "bias_thresholds": {
                "demographic_parity": 0.1,
                "equalized_odds": 0.1,
                "equal_opportunity": 0.1
            },
            "fairness_requirements": {
                "minimum_fairness_score": 0.8,
                "demographic_groups": ["gender", "age", "ethnicity", "location"],
                "protected_attributes": ["race", "gender", "religion", "political_affiliation"]
            },
            "transparency_requirements": {
                "minimum_explainability_score": 0.7,
                "explanation_methods": ["SHAP", "LIME", "attention"],
                "local_explanations_required": True
            },
            "monitoring": {
                "continuous_bias_monitoring": True,
                "fairness_drift_detection": True,
                "explanation_quality_tracking": True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        else:
            self.config_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f)
        
        return default_config
    
    async def _initialize_ethics_system(self):
        """Initialize AI ethics system and database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Bias assessments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bias_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    assessment_date TIMESTAMP,
                    bias_scores TEXT,  -- JSON
                    fairness_metrics TEXT,  -- JSON
                    demographic_groups TEXT,  -- JSON
                    performance_by_group TEXT,  -- JSON
                    assessment_method TEXT,
                    confidence_level REAL,
                    recommendations TEXT,  -- JSON
                    bias_risk_level TEXT,
                    fairness_compliance BOOLEAN,
                    requires_intervention BOOLEAN
                )
            """)
            
            # Explainability reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS explainability_reports (
                    report_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    generated_at TIMESTAMP,
                    transparency_level TEXT,
                    explainability_score REAL,
                    global_feature_importance TEXT,  -- JSON
                    local_explanations_available BOOLEAN,
                    explanation_methods TEXT,  -- JSON
                    explanation_quality TEXT,  -- JSON
                    user_satisfaction_score REAL
                )
            """)
            
            # Ethics violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ethics_violations (
                    violation_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    violation_type TEXT,
                    severity TEXT,
                    description TEXT,
                    evidence TEXT,  -- JSON
                    detected_at TIMESTAMP,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT,
                    resolution_plan TEXT,
                    resolved_at TIMESTAMP
                )
            """)
            
            # Fairness monitoring table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fairness_monitoring (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    demographic_group TEXT,
                    threshold_violated BOOLEAN,
                    drift_detected BOOLEAN
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("AI Ethics system database initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize AI ethics system", error=str(e))
            raise CyberLLMError("Ethics system initialization failed", ErrorCategory.SYSTEM)
    
    async def conduct_bias_assessment(self, 
                                    model_id: str,
                                    test_data: pd.DataFrame,
                                    protected_attributes: List[str],
                                    target_column: str) -> BiasAssessment:
        """Conduct comprehensive bias assessment"""
        
        assessment_id = f"bias_assessment_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Calculate bias metrics
            bias_scores = {}
            fairness_metrics = {}
            performance_by_group = {}
            
            # Demographic parity assessment
            for attr in protected_attributes:
                if attr in test_data.columns:
                    dp_score = await self._calculate_demographic_parity(
                        test_data, attr, target_column
                    )
                    bias_scores[BiasType.DEMOGRAPHIC] = dp_score
                    fairness_metrics[FairnessMetric.DEMOGRAPHIC_PARITY] = dp_score
            
            # Equalized odds assessment
            eo_score = await self._calculate_equalized_odds(test_data, protected_attributes, target_column)
            fairness_metrics[FairnessMetric.EQUALIZED_ODDS] = eo_score
            
            # Equal opportunity assessment
            eop_score = await self._calculate_equal_opportunity(test_data, protected_attributes, target_column)
            fairness_metrics[FairnessMetric.EQUAL_OPPORTUNITY] = eop_score
            
            # Performance by demographic group
            for attr in protected_attributes:
                if attr in test_data.columns:
                    group_performance = await self._calculate_group_performance(
                        test_data, attr, target_column
                    )
                    performance_by_group[attr] = group_performance
            
            # Overall bias risk assessment
            bias_risk_level = self._assess_bias_risk_level(bias_scores, fairness_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_bias_recommendations(
                bias_scores, fairness_metrics, performance_by_group
            )
            
            # Create bias assessment
            assessment = BiasAssessment(
                assessment_id=assessment_id,
                model_id=model_id,
                assessment_date=datetime.now(),
                bias_scores=bias_scores,
                fairness_metrics=fairness_metrics,
                demographic_groups=protected_attributes,
                performance_by_group=performance_by_group,
                assessment_method="comprehensive_statistical_analysis",
                confidence_level=0.95,
                recommendations=recommendations,
                bias_risk_level=bias_risk_level,
                fairness_compliance=self._check_fairness_compliance(fairness_metrics),
                requires_intervention=bias_risk_level in ["high", "critical"]
            )
            
            # Store assessment
            await self._store_bias_assessment(assessment)
            self.bias_assessments[assessment_id] = assessment
            
            self.logger.info(f"Bias assessment completed for model: {model_id}",
                           bias_risk=bias_risk_level,
                           fairness_compliant=assessment.fairness_compliance)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Failed to conduct bias assessment for model: {model_id}", error=str(e))
            raise CyberLLMError("Bias assessment failed", ErrorCategory.ANALYSIS)
    
    async def _calculate_demographic_parity(self, 
                                          data: pd.DataFrame,
                                          protected_attr: str,
                                          target_col: str) -> float:
        """Calculate demographic parity score"""
        
        groups = data[protected_attr].unique()
        positive_rates = {}
        
        for group in groups:
            group_data = data[data[protected_attr] == group]
            positive_rate = group_data[target_col].mean()
            positive_rates[group] = positive_rate
        
        # Calculate maximum difference in positive rates
        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates)
        
        # Convert to fairness score (1 - bias_level)
        return 1 - max_diff
    
    async def _calculate_equalized_odds(self, 
                                      data: pd.DataFrame,
                                      protected_attrs: List[str],
                                      target_col: str) -> float:
        """Calculate equalized odds score"""
        
        # Simplified equalized odds calculation
        # In practice, this would require model predictions and true labels
        
        total_score = 0
        valid_attrs = 0
        
        for attr in protected_attrs:
            if attr in data.columns:
                groups = data[attr].unique()
                if len(groups) >= 2:
                    # Calculate TPR and FPR for each group
                    group_scores = []
                    for group in groups:
                        group_data = data[data[attr] == group]
                        # Simplified metric - in practice would use true TPR/FPR
                        score = group_data[target_col].mean()
                        group_scores.append(score)
                    
                    # Equalized odds: minimize difference in TPR and FPR across groups
                    max_diff = max(group_scores) - min(group_scores)
                    attr_score = 1 - max_diff
                    total_score += attr_score
                    valid_attrs += 1
        
        return total_score / valid_attrs if valid_attrs > 0 else 1.0
    
    async def _calculate_equal_opportunity(self, 
                                         data: pd.DataFrame,
                                         protected_attrs: List[str],
                                         target_col: str) -> float:
        """Calculate equal opportunity score"""
        
        # Focus on true positive rates across groups
        total_score = 0
        valid_attrs = 0
        
        for attr in protected_attrs:
            if attr in data.columns:
                groups = data[attr].unique()
                if len(groups) >= 2:
                    tpr_scores = []
                    for group in groups:
                        group_data = data[data[attr] == group]
                        # Simplified - would use actual TPR in practice
                        tpr = group_data[target_col].mean()
                        tpr_scores.append(tpr)
                    
                    max_diff = max(tpr_scores) - min(tpr_scores)
                    attr_score = 1 - max_diff
                    total_score += attr_score
                    valid_attrs += 1
        
        return total_score / valid_attrs if valid_attrs > 0 else 1.0
    
    async def _calculate_group_performance(self, 
                                         data: pd.DataFrame,
                                         protected_attr: str,
                                         target_col: str) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by demographic group"""
        
        group_performance = {}
        groups = data[protected_attr].unique()
        
        for group in groups:
            group_data = data[data[protected_attr] == group]
            
            # Calculate various performance metrics
            performance = {
                "count": len(group_data),
                "positive_rate": group_data[target_col].mean(),
                "negative_rate": 1 - group_data[target_col].mean(),
                "representation": len(group_data) / len(data)
            }
            
            # Add statistical measures
            if len(group_data) > 1:
                performance["std_dev"] = group_data[target_col].std()
                performance["variance"] = group_data[target_col].var()
            
            group_performance[str(group)] = performance
        
        return group_performance
    
    def _assess_bias_risk_level(self, 
                              bias_scores: Dict[BiasType, float],
                              fairness_metrics: Dict[FairnessMetric, float]) -> str:
        """Assess overall bias risk level"""
        
        min_score = 1.0
        
        # Check bias scores
        for score in bias_scores.values():
            min_score = min(min_score, score)
        
        # Check fairness metrics
        for score in fairness_metrics.values():
            min_score = min(min_score, score)
        
        # Determine risk level based on minimum score
        if min_score >= 0.9:
            return "low"
        elif min_score >= 0.8:
            return "medium"
        elif min_score >= 0.6:
            return "high"
        else:
            return "critical"
    
    def _check_fairness_compliance(self, fairness_metrics: Dict[FairnessMetric, float]) -> bool:
        """Check if model meets fairness compliance requirements"""
        
        thresholds = self.config["bias_thresholds"]
        minimum_score = self.config["fairness_requirements"]["minimum_fairness_score"]
        
        for metric, score in fairness_metrics.items():
            threshold = thresholds.get(metric.value, minimum_score)
            if score < threshold:
                return False
        
        return True
    
    async def _generate_bias_recommendations(self, 
                                           bias_scores: Dict[BiasType, float],
                                           fairness_metrics: Dict[FairnessMetric, float],
                                           performance_by_group: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate bias remediation recommendations"""
        
        recommendations = []
        
        # Check demographic parity
        if FairnessMetric.DEMOGRAPHIC_PARITY in fairness_metrics:
            dp_score = fairness_metrics[FairnessMetric.DEMOGRAPHIC_PARITY]
            if dp_score < 0.8:
                recommendations.append("Apply post-processing calibration to achieve demographic parity")
                recommendations.append("Consider re-sampling training data to balance demographic groups")
        
        # Check equalized odds
        if FairnessMetric.EQUALIZED_ODDS in fairness_metrics:
            eo_score = fairness_metrics[FairnessMetric.EQUALIZED_ODDS]
            if eo_score < 0.8:
                recommendations.append("Implement equalized odds post-processing")
                recommendations.append("Review and adjust decision thresholds per demographic group")
        
        # Check representation
        for attr, groups in performance_by_group.items():
            min_representation = min(group["representation"] for group in groups.values())
            if min_representation < 0.1:  # Less than 10% representation
                recommendations.append(f"Increase representation for underrepresented groups in {attr}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue monitoring for bias drift during model operation")
        else:
            recommendations.append("Implement continuous bias monitoring in production")
            recommendations.append("Consider adversarial debiasing techniques during training")
        
        return recommendations
    
    async def generate_explainability_report(self, 
                                           model_id: str,
                                           model: Any,
                                           sample_data: pd.DataFrame) -> ExplainabilityReport:
        """Generate comprehensive explainability report"""
        
        report_id = f"explainability_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Calculate global feature importance (simplified)
            feature_importance = await self._calculate_feature_importance(model, sample_data)
            
            # Determine transparency level
            transparency_level = self._assess_transparency_level(model)
            
            # Calculate explainability score
            explainability_score = await self._calculate_explainability_score(
                model, sample_data, feature_importance
            )
            
            # Assess explanation methods availability
            explanation_methods = self._identify_explanation_methods(model)
            
            # Evaluate explanation quality
            explanation_quality = await self._evaluate_explanation_quality(
                model, sample_data, explanation_methods
            )
            
            # Create explainability report
            report = ExplainabilityReport(
                report_id=report_id,
                model_id=model_id,
                generated_at=datetime.now(),
                transparency_level=transparency_level,
                explainability_score=explainability_score,
                global_feature_importance=feature_importance,
                local_explanations_available=len(explanation_methods) > 0,
                explanation_methods=explanation_methods,
                explanation_quality=explanation_quality,
                user_satisfaction_score=None  # Would be collected from user feedback
            )
            
            # Store report
            await self._store_explainability_report(report)
            self.explainability_reports[report_id] = report
            
            self.logger.info(f"Explainability report generated for model: {model_id}",
                           transparency_level=transparency_level.value,
                           explainability_score=explainability_score)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate explainability report for model: {model_id}", error=str(e))
            raise CyberLLMError("Explainability report generation failed", ErrorCategory.ANALYSIS)
    
    async def _calculate_feature_importance(self, 
                                          model: Any,
                                          sample_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate global feature importance"""
        
        # Simplified feature importance calculation
        # In practice, would use SHAP, permutation importance, etc.
        
        feature_names = sample_data.columns.tolist()
        
        # Generate random importance scores (placeholder)
        # In real implementation, use actual model inspection techniques
        importance_scores = np.random.dirichlet(np.ones(len(feature_names)))
        
        return dict(zip(feature_names, importance_scores.tolist()))
    
    def _assess_transparency_level(self, model: Any) -> TransparencyLevel:
        """Assess model transparency level"""
        
        # Simplified assessment based on model type
        model_type = type(model).__name__.lower()
        
        if "linear" in model_type or "tree" in model_type:
            return TransparencyLevel.FULL_TRANSPARENCY
        elif "ensemble" in model_type or "forest" in model_type:
            return TransparencyLevel.FEATURE_IMPORTANCE
        elif "neural" in model_type or "deep" in model_type:
            return TransparencyLevel.LIMITED_EXPLANATION
        else:
            return TransparencyLevel.BLACK_BOX
    
    async def _calculate_explainability_score(self, 
                                            model: Any,
                                            sample_data: pd.DataFrame,
                                            feature_importance: Dict[str, float]) -> float:
        """Calculate overall explainability score"""
        
        # Factors contributing to explainability
        transparency_score = self._get_transparency_score(model)
        feature_clarity_score = self._assess_feature_clarity(feature_importance)
        interpretability_score = self._assess_model_interpretability(model)
        
        # Weighted average
        weights = [0.4, 0.3, 0.3]
        scores = [transparency_score, feature_clarity_score, interpretability_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _get_transparency_score(self, model: Any) -> float:
        """Get transparency score based on model type"""
        
        transparency_level = self._assess_transparency_level(model)
        
        scores = {
            TransparencyLevel.FULL_TRANSPARENCY: 1.0,
            TransparencyLevel.RULE_BASED: 0.9,
            TransparencyLevel.FEATURE_IMPORTANCE: 0.7,
            TransparencyLevel.LIMITED_EXPLANATION: 0.4,
            TransparencyLevel.BLACK_BOX: 0.1
        }
        
        return scores.get(transparency_level, 0.1)
    
    def _assess_feature_clarity(self, feature_importance: Dict[str, float]) -> float:
        """Assess clarity of feature importance"""
        
        importance_values = list(feature_importance.values())
        
        # High concentration of importance in few features = more interpretable
        gini_coefficient = self._calculate_gini_coefficient(importance_values)
        
        # Convert Gini coefficient to clarity score (higher Gini = more concentrated = clearer)
        return gini_coefficient
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement"""
        
        sorted_values = sorted(values)
        n = len(values)
        cumulative_sum = sum((i + 1) * val for i, val in enumerate(sorted_values))
        
        return (2 * cumulative_sum) / (n * sum(values)) - (n + 1) / n
    
    def _assess_model_interpretability(self, model: Any) -> float:
        """Assess overall model interpretability"""
        
        # Simplified assessment - in practice would analyze model architecture
        model_name = type(model).__name__.lower()
        
        interpretability_scores = {
            "logistic": 0.9,
            "linear": 0.9,
            "tree": 0.8,
            "forest": 0.6,
            "gradient": 0.5,
            "neural": 0.3,
            "deep": 0.2
        }
        
        for model_type, score in interpretability_scores.items():
            if model_type in model_name:
                return score
        
        return 0.1  # Default for unknown models
    
    def _identify_explanation_methods(self, model: Any) -> List[str]:
        """Identify available explanation methods for model"""
        
        methods = []
        model_name = type(model).__name__.lower()
        
        # Universal methods
        methods.extend(["permutation_importance", "partial_dependence"])
        
        # Model-specific methods
        if "linear" in model_name:
            methods.extend(["coefficients", "feature_weights"])
        elif "tree" in model_name:
            methods.extend(["tree_structure", "path_analysis"])
        elif "neural" in model_name:
            methods.extend(["gradient_attribution", "layer_wise_relevance"])
        
        # Advanced methods (if libraries available)
        methods.extend(["shap_values", "lime_explanations"])
        
        return methods
    
    async def _evaluate_explanation_quality(self, 
                                          model: Any,
                                          sample_data: pd.DataFrame,
                                          explanation_methods: List[str]) -> Dict[str, float]:
        """Evaluate quality of explanations"""
        
        quality_metrics = {
            "clarity": 0.0,
            "completeness": 0.0,
            "actionability": 0.0,
            "consistency": 0.0
        }
        
        # Clarity: how easy explanations are to understand
        quality_metrics["clarity"] = 0.8 if "shap_values" in explanation_methods else 0.6
        
        # Completeness: how much of model behavior is explained
        quality_metrics["completeness"] = min(1.0, len(explanation_methods) / 5)
        
        # Actionability: how useful explanations are for decisions
        actionable_methods = ["feature_weights", "shap_values", "lime_explanations"]
        actionable_count = sum(1 for method in explanation_methods if method in actionable_methods)
        quality_metrics["actionability"] = min(1.0, actionable_count / 3)
        
        # Consistency: how stable explanations are
        quality_metrics["consistency"] = 0.7  # Would measure through repeated explanations
        
        return quality_metrics
    
    async def monitor_fairness_drift(self, 
                                   model_id: str,
                                   current_data: pd.DataFrame,
                                   protected_attributes: List[str],
                                   target_column: str) -> Dict[str, Any]:
        """Monitor for fairness drift over time"""
        
        drift_report = {
            "model_id": model_id,
            "monitoring_date": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_metrics": {},
            "affected_groups": [],
            "recommendations": []
        }
        
        try:
            # Get historical fairness metrics
            historical_metrics = await self._get_historical_fairness_metrics(model_id)
            
            if not historical_metrics:
                self.logger.warning(f"No historical fairness data for model: {model_id}")
                return drift_report
            
            # Calculate current fairness metrics
            current_assessment = await self.conduct_bias_assessment(
                model_id, current_data, protected_attributes, target_column
            )
            
            current_metrics = current_assessment.fairness_metrics
            
            # Compare metrics for drift
            for metric, current_value in current_metrics.items():
                if metric.value in historical_metrics:
                    historical_value = historical_metrics[metric.value]
                    drift_magnitude = abs(current_value - historical_value)
                    
                    # Drift threshold (configurable)
                    drift_threshold = 0.05  # 5% change
                    
                    drift_report["drift_metrics"][metric.value] = {
                        "historical_value": historical_value,
                        "current_value": current_value,
                        "drift_magnitude": drift_magnitude,
                        "drift_detected": drift_magnitude > drift_threshold
                    }
                    
                    if drift_magnitude > drift_threshold:
                        drift_report["drift_detected"] = True
            
            # Identify affected demographic groups
            if drift_report["drift_detected"]:
                affected_groups = await self._identify_affected_groups(
                    current_assessment, historical_metrics
                )
                drift_report["affected_groups"] = affected_groups
                
                # Generate recommendations
                recommendations = await self._generate_drift_recommendations(drift_report)
                drift_report["recommendations"] = recommendations
            
            # Store monitoring record
            await self._store_fairness_monitoring_record(drift_report)
            
            return drift_report
            
        except Exception as e:
            self.logger.error(f"Failed to monitor fairness drift for model: {model_id}", error=str(e))
            raise CyberLLMError("Fairness drift monitoring failed", ErrorCategory.ANALYSIS)
    
    async def _get_historical_fairness_metrics(self, model_id: str) -> Dict[str, float]:
        """Get historical fairness metrics for comparison"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fairness_metrics FROM bias_assessments
                WHERE model_id = ?
                ORDER BY assessment_date DESC
                LIMIT 1
            """, (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            
            return {}
            
        except Exception as e:
            self.logger.error("Failed to retrieve historical fairness metrics", error=str(e))
            return {}
    
    async def _identify_affected_groups(self, 
                                       current_assessment: BiasAssessment,
                                       historical_metrics: Dict[str, float]) -> List[str]:
        """Identify demographic groups most affected by drift"""
        
        affected_groups = []
        
        # Compare group performance
        for group, performance in current_assessment.performance_by_group.items():
            # Simplified comparison - in practice would have historical group data
            if performance["positive_rate"] < 0.5:  # Example threshold
                affected_groups.append(group)
        
        return affected_groups
    
    async def _generate_drift_recommendations(self, drift_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing fairness drift"""
        
        recommendations = []
        
        if drift_report["drift_detected"]:
            recommendations.append("Investigate root causes of fairness drift")
            recommendations.append("Consider model retraining with recent data")
            
            if drift_report["affected_groups"]:
                recommendations.append("Focus remediation efforts on affected demographic groups")
                recommendations.append("Implement group-specific bias mitigation techniques")
            
            recommendations.append("Increase frequency of fairness monitoring")
            recommendations.append("Review and update fairness constraints")
        
        return recommendations
    
    def get_ethics_dashboard_data(self) -> Dict[str, Any]:
        """Get data for AI ethics dashboard"""
        
        # Summary statistics
        total_assessments = len(self.bias_assessments)
        compliant_models = sum(
            1 for assessment in self.bias_assessments.values()
            if assessment.fairness_compliance
        )
        
        high_risk_models = sum(
            1 for assessment in self.bias_assessments.values()
            if assessment.bias_risk_level in ["high", "critical"]
        )
        
        # Recent violations
        recent_violations = [
            v for v in self.ethics_violations
            if v.detected_at >= datetime.now() - timedelta(days=7)
        ]
        
        # Transparency metrics
        total_explainability_reports = len(self.explainability_reports)
        high_transparency_models = sum(
            1 for report in self.explainability_reports.values()
            if report.explainability_score >= 0.8
        )
        
        return {
            "bias_assessment": {
                "total_assessments": total_assessments,
                "compliant_models": compliant_models,
                "compliance_rate": compliant_models / total_assessments if total_assessments > 0 else 0,
                "high_risk_models": high_risk_models
            },
            "explainability": {
                "total_reports": total_explainability_reports,
                "high_transparency_models": high_transparency_models,
                "transparency_rate": high_transparency_models / total_explainability_reports if total_explainability_reports > 0 else 0
            },
            "violations": {
                "recent_violations": len(recent_violations),
                "open_violations": sum(1 for v in self.ethics_violations if v.status == "open")
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def _store_bias_assessment(self, assessment: BiasAssessment):
        """Store bias assessment in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO bias_assessments
                (assessment_id, model_id, assessment_date, bias_scores, fairness_metrics,
                 demographic_groups, performance_by_group, assessment_method, confidence_level,
                 recommendations, bias_risk_level, fairness_compliance, requires_intervention)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assessment.assessment_id,
                assessment.model_id,
                assessment.assessment_date.isoformat(),
                json.dumps({k.value: v for k, v in assessment.bias_scores.items()}),
                json.dumps({k.value: v for k, v in assessment.fairness_metrics.items()}),
                json.dumps(assessment.demographic_groups),
                json.dumps(assessment.performance_by_group),
                assessment.assessment_method,
                assessment.confidence_level,
                json.dumps(assessment.recommendations),
                assessment.bias_risk_level,
                assessment.fairness_compliance,
                assessment.requires_intervention
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store bias assessment", error=str(e))
    
    async def _store_explainability_report(self, report: ExplainabilityReport):
        """Store explainability report in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO explainability_reports
                (report_id, model_id, generated_at, transparency_level, explainability_score,
                 global_feature_importance, local_explanations_available, explanation_methods,
                 explanation_quality, user_satisfaction_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.model_id,
                report.generated_at.isoformat(),
                report.transparency_level.value,
                report.explainability_score,
                json.dumps(report.global_feature_importance),
                report.local_explanations_available,
                json.dumps(report.explanation_methods),
                json.dumps(report.explanation_quality),
                report.user_satisfaction_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store explainability report", error=str(e))
    
    async def _store_fairness_monitoring_record(self, drift_report: Dict[str, Any]):
        """Store fairness monitoring record"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_name, metric_data in drift_report["drift_metrics"].items():
                cursor.execute("""
                    INSERT INTO fairness_monitoring
                    (model_id, metric_name, metric_value, threshold_violated, drift_detected)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    drift_report["model_id"],
                    metric_name,
                    metric_data["current_value"],
                    metric_data["drift_detected"],
                    drift_report["drift_detected"]
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store fairness monitoring record", error=str(e))

# Factory function
def create_ai_ethics_manager(**kwargs) -> AIEthicsManager:
    """Create AI ethics manager with configuration"""
    return AIEthicsManager(**kwargs)
