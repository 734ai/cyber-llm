"""
Explainability Agent for Cyber-LLM
Provides rationale and explanation for agent decisions
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplainabilityAgent:
    """
    Agent responsible for providing explainable rationales for decisions
    made by other agents in the Cyber-LLM system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ExplainabilityAgent"""
        self.config = self._load_config(config_path)
        self.explanation_templates = self._load_explanation_templates()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration for the explainability agent"""
        default_config = {
            "explanation_depth": "detailed",  # basic, detailed, comprehensive
            "include_risks": True,
            "include_mitigations": True,
            "include_alternatives": True,
            "format": "json"  # json, markdown, yaml
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
                
        return default_config
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different agent types"""
        return {
            "recon": """
            RECONNAISSANCE DECISION EXPLANATION:
            Action: {action}
            Target: {target}
            
            Justification:
            - {justification}
            
            Risk Assessment:
            - Detection Risk: {detection_risk}
            - Network Impact: {network_impact}
            - Time Investment: {time_investment}
            
            OPSEC Considerations:
            - {opsec_considerations}
            
            Alternative Approaches:
            - {alternatives}
            """,
            
            "c2": """
            C2 CHANNEL DECISION EXPLANATION:
            Channel Type: {channel_type}
            Configuration: {configuration}
            
            Justification:
            - {justification}
            
            Risk Assessment:
            - Stealth Level: {stealth_level}
            - Reliability: {reliability}
            - Bandwidth: {bandwidth}
            
            OPSEC Considerations:
            - {opsec_considerations}
            
            Backup Options:
            - {backup_options}
            """,
            
            "post_exploit": """
            POST-EXPLOITATION DECISION EXPLANATION:
            Action: {action}
            Method: {method}
            
            Justification:
            - {justification}
            
            Risk Assessment:
            - Detection Probability: {detection_probability}
            - System Impact: {system_impact}
            - Evidence Left: {evidence_left}
            
            OPSEC Considerations:
            - {opsec_considerations}
            
            Cleanup Required:
            - {cleanup_required}
            """
        }
    
    def explain_decision(self, agent_type: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for an agent's decision
        
        Args:
            agent_type: Type of agent (recon, c2, post_exploit, etc.)
            decision_data: Data about the decision made
            
        Returns:
            Dictionary containing detailed explanation
        """
        try:
            explanation = {
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type,
                "decision_id": decision_data.get("id", "unknown"),
                "explanation": self._generate_explanation(agent_type, decision_data),
                "risk_assessment": self._assess_risks(agent_type, decision_data),
                "alternatives": self._suggest_alternatives(agent_type, decision_data),
                "confidence_score": self._calculate_confidence(decision_data)
            }
            
            if self.config.get("include_mitigations", True):
                explanation["mitigations"] = self._suggest_mitigations(agent_type, decision_data)
                
            logger.info(f"Generated explanation for {agent_type} decision: {decision_data.get('id', 'unknown')}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                "error": f"Failed to generate explanation: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type
            }
    
    def _generate_explanation(self, agent_type: str, decision_data: Dict[str, Any]) -> str:
        """Generate the core explanation for the decision"""
        if agent_type == "recon":
            return self._explain_recon_decision(decision_data)
        elif agent_type == "c2":
            return self._explain_c2_decision(decision_data)
        elif agent_type == "post_exploit":
            return self._explain_post_exploit_decision(decision_data)
        else:
            return f"Decision made by {agent_type} agent based on available information."
    
    def _explain_recon_decision(self, decision_data: Dict[str, Any]) -> str:
        """Explain reconnaissance decisions"""
        action = decision_data.get("action", "unknown")
        target = decision_data.get("target", "unknown")
        
        explanations = {
            "nmap_scan": f"Initiated Nmap scan against {target} to identify open ports and services. This is a standard reconnaissance technique that provides essential information for attack planning.",
            "shodan_search": f"Performed Shodan search for {target} to gather passive intelligence about exposed services without direct interaction with the target.",
            "dns_enum": f"Conducted DNS enumeration for {target} to map the network infrastructure and identify potential attack vectors."
        }
        
        return explanations.get(action, f"Performed {action} against {target} as part of reconnaissance phase.")
    
    def _explain_c2_decision(self, decision_data: Dict[str, Any]) -> str:
        """Explain C2 channel decisions"""
        channel = decision_data.get("channel_type", "unknown")
        
        explanations = {
            "http": "Selected HTTP channel for C2 communication due to its ability to blend with normal web traffic and bypass many network filters.",
            "https": "Chose HTTPS channel for encrypted C2 communication, providing both stealth and security for command transmission.",
            "dns": "Implemented DNS tunneling for C2 to leverage a protocol that is rarely blocked and often unmonitored."
        }
        
        return explanations.get(channel, f"Established {channel} C2 channel based on network constraints and stealth requirements.")
    
    def _explain_post_exploit_decision(self, decision_data: Dict[str, Any]) -> str:
        """Explain post-exploitation decisions"""
        action = decision_data.get("action", "unknown")
        
        explanations = {
            "credential_dump": "Initiated credential dumping to harvest authentication materials for lateral movement and privilege escalation.",
            "lateral_movement": "Attempting lateral movement to expand access within the target network and reach high-value assets.",
            "persistence": "Establishing persistence mechanisms to maintain access even after system reboots or security updates."
        }
        
        return explanations.get(action, f"Executed {action} to advance the attack chain and achieve mission objectives.")
    
    def _assess_risks(self, agent_type: str, decision_data: Dict[str, Any]) -> Dict[str, str]:
        """Assess risks associated with the decision"""
        risk_factors = {
            "detection_risk": "medium",
            "system_impact": "low",
            "evidence_trail": "minimal",
            "network_noise": "low"
        }
        
        # Adjust risk factors based on agent type and action
        if agent_type == "recon":
            action = decision_data.get("action", "")
            if "aggressive" in action.lower() or "fast" in action.lower():
                risk_factors["detection_risk"] = "high"
                risk_factors["network_noise"] = "high"
                
        elif agent_type == "post_exploit":
            action = decision_data.get("action", "")
            if "dump" in action.lower() or "extract" in action.lower():
                risk_factors["detection_risk"] = "high"
                risk_factors["system_impact"] = "medium"
                risk_factors["evidence_trail"] = "significant"
        
        return risk_factors
    
    def _suggest_alternatives(self, agent_type: str, decision_data: Dict[str, Any]) -> List[str]:
        """Suggest alternative approaches"""
        alternatives = []
        
        if agent_type == "recon":
            alternatives = [
                "Use passive reconnaissance techniques instead of active scanning",
                "Employ slower scan rates to reduce detection probability",
                "Utilize third-party intelligence sources for initial reconnaissance"
            ]
        elif agent_type == "c2":
            alternatives = [
                "Consider domain fronting techniques for additional stealth",
                "Implement multiple fallback C2 channels",
                "Use legitimate cloud services as C2 infrastructure"
            ]
        elif agent_type == "post_exploit":
            alternatives = [
                "Use living-off-the-land techniques instead of custom tools",
                "Implement time delays between actions to avoid pattern detection",
                "Utilize legitimate administrative tools for post-exploitation activities"
            ]
            
        return alternatives
    
    def _suggest_mitigations(self, agent_type: str, decision_data: Dict[str, Any]) -> List[str]:
        """Suggest risk mitigation strategies"""
        mitigations = [
            "Monitor network traffic for anomalous patterns",
            "Implement rate limiting to slow down automated attacks",
            "Deploy behavioral analysis tools to detect suspicious activities",
            "Maintain updated incident response procedures"
        ]
        
        return mitigations
    
    def _calculate_confidence(self, decision_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the decision"""
        # Simple confidence calculation based on available data
        factors = []
        
        if decision_data.get("target"):
            factors.append(0.2)
        if decision_data.get("action"):
            factors.append(0.3)
        if decision_data.get("parameters"):
            factors.append(0.2)
        if decision_data.get("context"):
            factors.append(0.3)
            
        return min(sum(factors), 1.0)
    
    def format_explanation(self, explanation: Dict[str, Any], format_type: str = "json") -> str:
        """Format explanation in the specified format"""
        if format_type == "json":
            return json.dumps(explanation, indent=2)
        elif format_type == "yaml":
            return yaml.dump(explanation, default_flow_style=False)
        elif format_type == "markdown":
            return self._format_as_markdown(explanation)
        else:
            return str(explanation)
    
    def _format_as_markdown(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as markdown"""
        md = f"""
# Decision Explanation Report

**Agent Type**: {explanation.get('agent_type', 'Unknown')}  
**Decision ID**: {explanation.get('decision_id', 'Unknown')}  
**Timestamp**: {explanation.get('timestamp', 'Unknown')}  
**Confidence Score**: {explanation.get('confidence_score', 0.0):.2f}

## Explanation
{explanation.get('explanation', 'No explanation available')}

## Risk Assessment
"""
        
        risks = explanation.get('risk_assessment', {})
        for risk, level in risks.items():
            md += f"- **{risk.replace('_', ' ').title()}**: {level}\n"
        
        if explanation.get('alternatives'):
            md += "\n## Alternative Approaches\n"
            for alt in explanation['alternatives']:
                md += f"- {alt}\n"
        
        if explanation.get('mitigations'):
            md += "\n## Suggested Mitigations\n"
            for mit in explanation['mitigations']:
                md += f"- {mit}\n"
        
        return md

# Example usage and testing
if __name__ == "__main__":
    # Initialize the explainability agent
    explainer = ExplainabilityAgent()
    
    # Example recon decision
    recon_decision = {
        "id": "recon_001",
        "action": "nmap_scan",
        "target": "192.168.1.1-100",
        "parameters": {
            "scan_type": "TCP SYN",
            "ports": "1-1000",
            "timing": "T3"
        },
        "context": "Initial network reconnaissance"
    }
    
    # Generate explanation
    explanation = explainer.explain_decision("recon", recon_decision)
    
    # Format and display
    print("JSON Format:")
    print(explainer.format_explanation(explanation, "json"))
    
    print("\nMarkdown Format:")
    print(explainer.format_explanation(explanation, "markdown"))
