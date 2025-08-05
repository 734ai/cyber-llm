# Agent Prompt & Task Definitions (Advanced)

tasks:
  - id: setup_environment
    description: "Initialize Git repo, branches, and CI config files."
    actions:
      - create_branch: [main, dev, agent-ci]
      - generate_file: .github/workflows/ci.yaml
      - configure_dvc_remote: configs/dvc_remote.yaml

  - id: ingest_and_validate_data
    description: "Convert, embed, validate, and version datasets."
    steps:
      - run: scripts/convert_pdf_to_txt.py --input data/raw --output data/processed
      - run: scripts/generate_embeddings.py
      - test: pytest tests/data_validation
      - dvc: dvc run -n embed --deps data/processed --outs src/data/embeddings scripts/generate_embeddings.py

  - id: train_adapters
    description: "Fine-tune LoRA adapters with adversarial loops and track metrics."
    steps:
      - run: python src/training/adversarial_train.py --config configs/finetune.yaml
      - log: mlflow tracking
      - test: pytest tests/model_validation

  - id: generate_agent_code
    description: "Scaffold agent modules with error handling, logging, and human checkpoints."
    templates:
      - src/agents/recon_agent.py.hbs
      - src/agents/c2_agent.py.hbs
      - src/agents/post_exploit_agent.py.hbs
      - src/agents/explainability_agent.py.hbs
      - src/agents/safety_agent.py.hbs
    post_process:
      - insert: "# HUMAN_APPROVAL_REQUIRED" at key decision points

  - id: orchestrate_pipeline
    description: "Define and execute orchestrator workflows with dry-run support."
    config: src/orchestration/workflows/red_team.yaml
    run: scripts/run_agents.sh --pipeline red-team --dry-run

  - id: ci_cd_and_deploy
    description: "Run CI, build images, deploy to staging, and promote on approval."
    workflows:
      - .github/workflows/ci.yaml
      - charts/cyber-llm
      - k8s/prod/deployment.yaml
    notifications:
      - slack: "#red-team-alerts"

  - id: evaluate_and_report
    description: "Execute benchmarks, generate explainability reports, and open issues if failures."
    steps:
      - run: python src/evaluation/evaluate.py --suite full
      - run: python src/evaluation/explainability_report.py
      - issue: GitHub issue creation on failure

# Agent Workflow Templates

## ReconAgent
```yaml
tool: ReconAgent
description: |
  Perform stealth reconnaissance on target.
prompt: |
  You are ReconAgent. Input: target IP or domain.
  1. Generate optimized Nmap commands for stealth and speed.
  2. Create Shodan and Censys query strings.
  3. Recommend passive DNS and OSINT steps.
output_format: JSON
fields:
  nmap: list[str]
  shodan: str
  passive_dns: str
  notes: str
```

## C2Agent
```yaml
tool: C2Agent
description: |
  Configure C2 channel within network constraints.
prompt: |
  You are C2Agent with Empire and Cobalt Strike.
  Input: payload type, network environment.
  1. Select C2 profile (HTTP/DNS/Jitter).
  2. Configure beacon parameters with OPSEC.
  3. Output API calls and commands.
output_format: JSON
```

## PostExploitAgent
```yaml
tool: PostExploitAgent
description: |
  Harvest credentials and move laterally.
prompt: |
  You are PostExploitAgent. With initial shell:
  1. Dump creds (Invoke-Mimikatz/SharpHound).
  2. Query BloodHound for lateral paths.
  3. Suggest persistence strategies.
```

## ExplainabilityAgent
```yaml
tool: ExplainabilityAgent
description: |
  Provide rationale for each step.
prompt: |
  You are ExplainabilityAgent.
  Given agent action JSON, output pillars:
    - Justification
    - Detected risks
    - Mitigations
```

## SafetyAgent
```yaml
tool: SafetyAgent
description: |
  Validate OPSEC compliance.
prompt: |
  You are SafetyAgent.
  Input: planned commands.
  1. Check for high-detection flags.
  2. Recommend safer alternatives.
```

## Orchestrator Workflow
```yaml
workflow_steps:
  - ReconAgent
  - SafetyAgent
  - C2Agent
  - ExplainabilityAgent
  - PostExploitAgent
  - EvaluateAgent
```
