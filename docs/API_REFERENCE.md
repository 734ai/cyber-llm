# Cyber-LLM API Reference

## Overview

The Cyber-LLM API provides comprehensive programmatic access to all platform capabilities including AI agents, compliance frameworks, threat intelligence, and enterprise management features. This RESTful API follows OpenAPI 3.0 specifications and supports both synchronous and asynchronous operations.

## Base URL
```
Production: https://api.cyber-llm.com/v1
Staging: https://staging-api.cyber-llm.com/v1
Development: http://localhost:8080/api/v1
```

## Authentication

### API Key Authentication
```http
GET /api/v1/agents/status
Authorization: Bearer YOUR_API_KEY
```

### OAuth 2.0 (Enterprise)
```http
POST /api/v1/auth/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=YOUR_CLIENT_ID&
client_secret=YOUR_CLIENT_SECRET&
scope=agents:read agents:write compliance:read
```

### JWT Token (Session-based)
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "security_analyst",
  "password": "secure_password",
  "mfa_token": "123456"
}
```

## Core Endpoints

### Agent Management

#### Start Security Assessment
```http
POST /api/v1/agents/assessments
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Monthly Security Assessment",
  "target": {
    "type": "domain",
    "value": "example.com",
    "scope": ["web", "network", "infrastructure"]
  },
  "configuration": {
    "agents": ["recon", "vulnerability", "c2"],
    "stealth_mode": true,
    "compliance_framework": "NIST",
    "max_duration": 3600,
    "notification_webhook": "https://your-webhook.com/notify"
  },
  "metadata": {
    "project_id": "proj_123",
    "requested_by": "john.doe@example.com",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "assessment_id": "asmt_7d4f2a1b8c3e9f5d",
  "status": "queued",
  "created_at": "2024-08-06T10:30:00Z",
  "estimated_completion": "2024-08-06T11:30:00Z",
  "agents_assigned": [
    {
      "agent_id": "recon_001",
      "agent_type": "reconnaissance",
      "status": "initializing"
    }
  ],
  "progress_url": "/api/v1/agents/assessments/asmt_7d4f2a1b8c3e9f5d/progress",
  "results_url": "/api/v1/agents/assessments/asmt_7d4f2a1b8c3e9f5d/results"
}
```

#### Get Assessment Status
```http
GET /api/v1/agents/assessments/{assessment_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "assessment_id": "asmt_7d4f2a1b8c3e9f5d",
  "status": "running",
  "progress": {
    "overall_progress": 45,
    "current_phase": "vulnerability_scanning",
    "phases_completed": ["reconnaissance"],
    "estimated_remaining": 1800
  },
  "agent_status": [
    {
      "agent_id": "recon_001",
      "status": "completed",
      "findings_count": 23,
      "completion_time": "2024-08-06T10:45:00Z"
    },
    {
      "agent_id": "vuln_002", 
      "status": "running",
      "current_target": "192.168.1.100",
      "scanned_hosts": 15,
      "total_hosts": 34
    }
  ],
  "real_time_updates": "wss://api.cyber-llm.com/v1/assessments/asmt_7d4f2a1b8c3e9f5d/stream"
}
```

#### Get Assessment Results
```http
GET /api/v1/agents/assessments/{assessment_id}/results
Authorization: Bearer <token>
```

**Response:**
```json
{
  "assessment_id": "asmt_7d4f2a1b8c3e9f5d",
  "status": "completed",
  "completion_time": "2024-08-06T11:25:00Z",
  "execution_time": 3300,
  "summary": {
    "overall_risk_score": 7.2,
    "critical_findings": 2,
    "high_findings": 8,
    "medium_findings": 15,
    "low_findings": 23,
    "info_findings": 12
  },
  "agent_results": [
    {
      "agent_type": "reconnaissance",
      "findings": [
        {
          "id": "recon_001",
          "type": "subdomain_discovery",
          "severity": "info",
          "title": "Subdomain Enumeration Results",
          "description": "Discovered 15 subdomains for example.com",
          "evidence": {
            "subdomains": ["admin.example.com", "api.example.com", "dev.example.com"],
            "discovery_method": "dns_enumeration",
            "tools_used": ["subfinder", "amass"]
          },
          "recommendations": [
            "Review subdomain exposure policy",
            "Implement subdomain monitoring"
          ]
        }
      ]
    }
  ],
  "compliance_assessment": {
    "framework": "NIST",
    "overall_score": 82,
    "category_scores": {
      "identify": 85,
      "protect": 78,
      "detect": 84,
      "respond": 80,
      "recover": 83
    }
  },
  "reports": [
    {
      "format": "pdf",
      "type": "executive_summary",
      "url": "/api/v1/reports/asmt_7d4f2a1b8c3e9f5d/executive.pdf"
    },
    {
      "format": "json",
      "type": "technical_detailed", 
      "url": "/api/v1/reports/asmt_7d4f2a1b8c3e9f5d/technical.json"
    }
  ]
}
```

### Agent Control

#### List Available Agents
```http
GET /api/v1/agents
Authorization: Bearer <token>
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "recon_001",
      "agent_type": "reconnaissance",
      "version": "2.1.0",
      "status": "available",
      "capabilities": [
        "network_discovery",
        "port_scanning", 
        "service_enumeration",
        "subdomain_discovery",
        "osint_gathering"
      ],
      "resource_usage": {
        "cpu_percent": 12,
        "memory_mb": 256,
        "concurrent_tasks": 2,
        "max_concurrent": 10
      },
      "last_health_check": "2024-08-06T10:29:45Z"
    }
  ],
  "total_agents": 15,
  "available_agents": 12,
  "busy_agents": 3
}
```

#### Execute Single Agent Task
```http
POST /api/v1/agents/{agent_type}/execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "task": {
    "type": "port_scan",
    "target": "192.168.1.100",
    "parameters": {
      "ports": "1-65535",
      "scan_type": "syn",
      "timing": "normal"
    }
  },
  "configuration": {
    "timeout": 300,
    "priority": "medium",
    "stealth_mode": true
  }
}
```

### Compliance Management

#### Get Compliance Status
```http
GET /api/v1/compliance/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overall_compliance_score": 87.5,
  "last_assessment_date": "2024-08-06T09:00:00Z",
  "frameworks": [
    {
      "framework": "SOC2_TYPE_II",
      "status": "compliant",
      "score": 92.3,
      "last_audit": "2024-07-15T00:00:00Z",
      "next_audit": "2025-07-15T00:00:00Z",
      "critical_gaps": 0,
      "recommendations_count": 3
    },
    {
      "framework": "ISO27001", 
      "status": "partial_compliance",
      "score": 78.2,
      "last_audit": "2024-06-30T00:00:00Z",
      "critical_gaps": 2,
      "recommendations_count": 12
    }
  ],
  "trending": {
    "direction": "improving",
    "score_change_30d": +4.2,
    "resolved_issues_30d": 8
  }
}
```

#### Run Compliance Assessment
```http
POST /api/v1/compliance/assessments
Authorization: Bearer <token>
Content-Type: application/json

{
  "framework": "NIST_CYBERSECURITY",
  "scope": {
    "organizational_units": ["engineering", "operations"],
    "systems": ["production", "staging"],
    "assessment_type": "comprehensive"
  },
  "configuration": {
    "automated_evidence_collection": true,
    "include_interviews": false,
    "generate_gap_analysis": true,
    "remediation_planning": true
  },
  "reporting": {
    "executive_summary": true,
    "detailed_findings": true,
    "formats": ["pdf", "json", "csv"]
  }
}
```

### Threat Intelligence

#### Query Threat Intelligence
```http
GET /api/v1/threat-intel/search
Authorization: Bearer <token>
```

**Parameters:**
- `query`: Search query (IOCs, threat actors, etc.)
- `type`: Threat type filter (malware, vulnerability, etc.)
- `severity`: Minimum severity level
- `limit`: Maximum results (default: 100)

**Example:**
```http
GET /api/v1/threat-intel/search?query=APT29&type=threat_actor&limit=50
```

**Response:**
```json
{
  "results": [
    {
      "id": "threat_actor_apt29",
      "type": "threat_actor",
      "name": "APT29 (Cozy Bear)",
      "description": "Russian state-sponsored threat group",
      "severity": "critical",
      "first_seen": "2008-01-01T00:00:00Z",
      "last_activity": "2024-07-28T00:00:00Z",
      "attributes": {
        "country": "Russia",
        "motivation": "espionage",
        "sophistication": "high",
        "targeting": ["government", "technology", "healthcare"]
      },
      "indicators": [
        {
          "type": "domain",
          "value": "cozy-bear-c2.com",
          "confidence": 95,
          "last_seen": "2024-07-25T00:00:00Z"
        }
      ],
      "techniques": [
        "T1566.001",  # Spearphishing Attachment
        "T1055",      # Process Injection
        "T1083"       # File and Directory Discovery
      ]
    }
  ],
  "total_results": 127,
  "page": 1,
  "per_page": 50
}
```

#### Get CVE Information
```http
GET /api/v1/threat-intel/cve/{cve_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "cve_id": "CVE-2024-12345",
  "description": "Remote code execution vulnerability in Example Software",
  "published_date": "2024-08-01T00:00:00Z",
  "modified_date": "2024-08-05T00:00:00Z",
  "severity": {
    "cvss_v3": {
      "base_score": 9.8,
      "base_severity": "Critical",
      "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    }
  },
  "affected_products": [
    {
      "vendor": "Example Corp",
      "product": "Example Software",
      "versions": ["< 2.1.5", ">= 3.0.0, < 3.2.1"]
    }
  ],
  "references": [
    "https://example.com/security-advisory-2024-001",
    "https://nvd.nist.gov/vuln/detail/CVE-2024-12345"
  ],
  "exploit_availability": {
    "public_exploits": true,
    "exploit_sources": ["exploit-db", "github"],
    "weaponization_level": "high"
  }
}
```

### Knowledge Graph

#### Query Knowledge Graph
```http
POST /api/v1/knowledge-graph/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query_type": "cypher",
  "query": "MATCH (ta:ThreatActor)-[:USES]->(m:Malware) WHERE ta.name CONTAINS 'APT' RETURN ta, m LIMIT 10",
  "parameters": {}
}
```

#### Find Related Entities
```http
GET /api/v1/knowledge-graph/entities/{entity_id}/related
Authorization: Bearer <token>
```

**Parameters:**
- `relationship_types`: Filter by relationship types
- `max_depth`: Maximum relationship depth (default: 2)
- `limit`: Maximum results

### Reporting

#### Generate Report
```http
POST /api/v1/reports/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "report_type": "security_assessment",
  "data_sources": [
    {
      "type": "assessment",
      "assessment_id": "asmt_7d4f2a1b8c3e9f5d"
    }
  ],
  "template": "executive_summary",
  "format": "pdf",
  "configuration": {
    "include_charts": true,
    "include_recommendations": true,
    "classification": "confidential",
    "branding": {
      "logo_url": "https://company.com/logo.png",
      "company_name": "Acme Corp"
    }
  }
}
```

#### List Reports
```http
GET /api/v1/reports
Authorization: Bearer <token>
```

### System Management

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-06T10:30:00Z",
  "services": {
    "api_gateway": "healthy",
    "orchestrator": "healthy", 
    "agents": {
      "recon": "healthy",
      "c2": "healthy",
      "safety": "healthy"
    },
    "databases": {
      "postgresql": "healthy",
      "neo4j": "healthy",
      "redis": "healthy"
    }
  },
  "version": "2.1.0",
  "uptime_seconds": 86400
}
```

#### System Metrics
```http
GET /api/v1/metrics
Authorization: Bearer <token>
```

**Response:**
```json
{
  "performance": {
    "requests_per_second": 125.3,
    "average_response_time_ms": 245,
    "error_rate_percent": 0.02
  },
  "resource_usage": {
    "cpu_percent": 34.5,
    "memory_percent": 68.2,
    "disk_usage_percent": 45.8
  },
  "agents": {
    "total_assessments_24h": 47,
    "active_assessments": 12,
    "average_assessment_duration_minutes": 42
  }
}
```

## WebSocket API

### Real-time Assessment Updates
```javascript
const ws = new WebSocket('wss://api.cyber-llm.com/v1/assessments/asmt_123/stream');

ws.onopen = function(event) {
    console.log('Connected to assessment stream');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'progress_update':
            updateProgressBar(data.progress);
            break;
            
        case 'finding_discovered':
            displayNewFinding(data.finding);
            break;
            
        case 'agent_status_change':
            updateAgentStatus(data.agent_id, data.status);
            break;
            
        case 'assessment_completed':
            showCompletionNotification(data.results_url);
            break;
    }
};
```

### System Alerts
```javascript
const alertsWs = new WebSocket('wss://api.cyber-llm.com/v1/alerts/stream');

alertsWs.onmessage = function(event) {
    const alert = JSON.parse(event.data);
    
    if (alert.severity === 'critical') {
        showCriticalAlert(alert);
    }
};
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_TARGET",
    "message": "The specified target is not valid",
    "details": {
      "target": "invalid-domain..com",
      "validation_errors": [
        "Invalid domain format",
        "Consecutive dots not allowed"
      ]
    },
    "request_id": "req_7d4f2a1b8c3e9f5d",
    "timestamp": "2024-08-06T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid or expired authentication credentials |
| `AUTHORIZATION_DENIED` | 403 | Insufficient permissions for requested operation |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `INVALID_REQUEST` | 400 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests in time window |
| `AGENT_UNAVAILABLE` | 503 | Requested agent is not available |
| `ASSESSMENT_FAILED` | 500 | Assessment execution failed |
| `COMPLIANCE_ERROR` | 500 | Compliance assessment error |

## Rate Limiting

### Default Limits
- **Standard API**: 1000 requests/hour per API key
- **Assessment API**: 100 assessments/day per API key
- **WebSocket**: 10 concurrent connections per API key

### Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1628097600
X-RateLimit-Retry-After: 3600
```

## SDK Examples

### Python
```python
import asyncio
from cyber_llm_sdk import CyberLLMClient

async def run_assessment():
    client = CyberLLMClient(
        api_key="your-api-key",
        base_url="https://api.cyber-llm.com/v1"
    )
    
    # Start assessment
    assessment = await client.start_assessment({
        "target": {"type": "domain", "value": "example.com"},
        "configuration": {
            "agents": ["recon", "vulnerability"],
            "stealth_mode": True
        }
    })
    
    print(f"Assessment started: {assessment.id}")
    
    # Monitor progress
    async for update in client.stream_assessment_progress(assessment.id):
        print(f"Progress: {update.progress}%")
        
        if update.status == "completed":
            break
    
    # Get results
    results = await client.get_assessment_results(assessment.id)
    print(f"Found {len(results.findings)} findings")
    
    return results

# Run
results = asyncio.run(run_assessment())
```

### JavaScript/Node.js
```javascript
const CyberLLM = require('cyber-llm-sdk');

const client = new CyberLLM.Client({
    apiKey: 'your-api-key',
    baseURL: 'https://api.cyber-llm.com/v1'
});

async function runAssessment() {
    try {
        // Start assessment
        const assessment = await client.startAssessment({
            target: { type: 'domain', value: 'example.com' },
            configuration: {
                agents: ['recon', 'vulnerability'],
                stealthMode: true
            }
        });
        
        console.log(`Assessment started: ${assessment.id}`);
        
        // Wait for completion
        const results = await client.waitForAssessment(assessment.id);
        
        console.log(`Assessment completed with ${results.findings.length} findings`);
        return results;
        
    } catch (error) {
        console.error('Assessment failed:', error);
    }
}

runAssessment();
```

## Changelog

### v2.1.0 (2024-08-06)
- Added advanced cognitive architecture endpoints
- Enhanced multi-agent collaboration APIs
- Improved knowledge graph integration
- New universal tool integration framework

### v2.0.0 (2024-07-01)
- Complete API redesign with OpenAPI 3.0
- Added enterprise compliance endpoints
- Enhanced real-time streaming capabilities
- Improved error handling and validation

---

For more information and updates, visit [api-docs.cyber-llm.com](https://api-docs.cyber-llm.com)
