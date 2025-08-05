#!/usr/bin/env python3
"""
Cyber-LLM Command Line Interface
Provides command-line access to Cyber-LLM agents and capabilities.
"""

import click
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agents.recon_agent import ReconAgent, ReconTarget
    from agents.safety_agent import SafetyAgent
    from orchestration.orchestrator import CyberOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version='0.4.0')
@click.option('--config', default='configs/model_config.json', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Cyber-LLM: Advanced Cybersecurity AI Assistant."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = Path(config)
    ctx.obj['verbose'] = verbose

@cli.group()
@click.pass_context
def agent(ctx):
    """Run individual agents."""
    pass

@agent.command()
@click.option('--target', required=True, help='Target IP, domain, or network')
@click.option('--type', 'target_type', default='auto', 
              type=click.Choice(['auto', 'ip', 'domain', 'network', 'organization']))
@click.option('--opsec', default='medium',
              type=click.Choice(['low', 'medium', 'high', 'maximum']))
@click.option('--output', '-o', help='Output file for results')
@click.option('--dry-run', is_flag=True, help='Show plan without execution')
@click.pass_context
def recon(ctx, target, target_type, opsec, output, dry_run):
    """Run reconnaissance operations."""
    try:
        # Initialize ReconAgent
        agent = ReconAgent()
        
        # Auto-detect target type if needed
        if target_type == 'auto':
            if target.count('.') == 3 and all(p.isdigit() for p in target.split('.')):
                target_type = 'ip'
            elif '.' in target:
                target_type = 'domain'
            else:
                target_type = 'organization'
        
        # Create target info
        target_info = ReconTarget(
            target=target,
            target_type=target_type,
            constraints={'dry_run': dry_run},
            opsec_level=opsec
        )
        
        # Execute reconnaissance
        result = agent.execute_reconnaissance(target_info)
        
        # Display results
        click.echo(f"Reconnaissance Results for {target}")
        click.echo("=" * 50)
        click.echo(f"Target Type: {target_type}")
        click.echo(f"OPSEC Level: {opsec}")
        click.echo(f"Status: {result['execution_status']}")
        
        if dry_run:
            click.echo("\n[DRY RUN MODE - No actual commands executed]")
        
        # Show planned commands
        plan = result['plan']
        
        click.echo("\nPlanned Commands:")
        for category, commands in plan['commands'].items():
            if commands:
                click.echo(f"\n{category.upper()}:")
                for cmd in commands:
                    click.echo(f"  {cmd}")
        
        # Show OPSEC notes
        if plan['opsec_notes']:
            click.echo("\nOPSEC Considerations:")
            for note in plan['opsec_notes']:
                click.echo(f"  • {note}")
        
        # Show risk assessment
        click.echo(f"\nRisk Assessment: {plan['risk_assessment']}")
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"\nResults saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"Error during reconnaissance: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@agent.command()
@click.option('--commands-file', required=True, help='JSON file containing commands to validate')
@click.option('--opsec', default='medium',
              type=click.Choice(['low', 'medium', 'high', 'maximum']))
@click.option('--output', '-o', help='Output file for assessment results')
@click.pass_context
def safety(ctx, commands_file, opsec, output):
    """Validate commands for safety and OPSEC compliance."""
    try:
        # Load commands from file
        commands_path = Path(commands_file)
        if not commands_path.exists():
            click.echo(f"Commands file not found: {commands_file}", err=True)
            return
        
        with open(commands_path, 'r') as f:
            commands = json.load(f)
        
        # Initialize SafetyAgent
        agent = SafetyAgent()
        
        # Validate commands
        assessment = agent.validate_commands(commands, opsec_level=opsec)
        
        # Display results
        click.echo("Safety Assessment Results")
        click.echo("=" * 30)
        click.echo(f"Overall Risk: {assessment.overall_risk.value.upper()}")
        click.echo(f"Approved: {'✓' if assessment.approved else '✗'}")
        
        # Show individual checks
        click.echo("\nDetailed Checks:")
        for check in assessment.checks:
            status = "✓" if check.risk_level.value == 'low' else "⚠" if check.risk_level.value == 'medium' else "✗"
            click.echo(f"  {status} {check.check_name}: {check.risk_level.value.upper()}")
            
            if check.violations:
                for violation in check.violations:
                    click.echo(f"    • {violation}")
        
        # Show recommendations
        if any(check.recommendations for check in assessment.checks):
            click.echo("\nRecommendations:")
            for check in assessment.checks:
                for rec in check.recommendations:
                    click.echo(f"  • {rec}")
        
        # Show safe alternatives if not approved
        if not assessment.approved and assessment.safe_alternatives:
            click.echo("\nSafe Alternatives:")
            for alt in assessment.safe_alternatives:
                click.echo(f"  • {alt}")
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            assessment_dict = {
                'overall_risk': assessment.overall_risk.value,
                'approved': assessment.approved,
                'summary': assessment.summary,
                'checks': [
                    {
                        'name': check.check_name,
                        'risk_level': check.risk_level.value,
                        'description': check.description,
                        'violations': check.violations,
                        'recommendations': check.recommendations
                    }
                    for check in assessment.checks
                ],
                'safe_alternatives': assessment.safe_alternatives
            }
            
            with open(output_path, 'w') as f:
                json.dump(assessment_dict, f, indent=2)
            click.echo(f"\nAssessment saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"Error during safety assessment: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--scenario', required=True, help='Scenario name or file path')
@click.option('--target', help='Target for the scenario')
@click.option('--opsec', default='medium',
              type=click.Choice(['low', 'medium', 'high', 'maximum']))
@click.option('--dry-run', is_flag=True, help='Simulation mode only')
@click.option('--output', '-o', help='Output directory for results')
@click.pass_context
def orchestrate(ctx, scenario, target, opsec, dry_run, output):
    """Run orchestrated multi-agent scenarios."""
    try:
        click.echo(f"Orchestrating scenario: {scenario}")
        click.echo(f"Target: {target}")
        click.echo(f"OPSEC Level: {opsec}")
        
        if dry_run:
            click.echo("\n[SIMULATION MODE]")
        
        # Initialize orchestrator
        # orchestrator = CyberOrchestrator()
        
        # For now, show what would be orchestrated
        click.echo("\nPlanned Orchestration Flow:")
        click.echo("1. ReconAgent - Initial target analysis")
        click.echo("2. SafetyAgent - OPSEC compliance validation")
        click.echo("3. ReconAgent - Execute approved reconnaissance")
        click.echo("4. ExplainabilityAgent - Generate rationale")
        click.echo("5. Generate final report")
        
        click.echo("\n[ORCHESTRATION FEATURE IN DEVELOPMENT]")
        click.echo("This feature will be available in the next release.")
        
    except Exception as e:
        click.echo(f"Error during orchestration: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--input-dir', required=True, help='Input directory with raw data')
@click.option('--output-dir', required=True, help='Output directory for processed data')
@click.option('--stage', default='all', 
              type=click.Choice(['convert', 'embed', 'preprocess', 'all']))
@click.pass_context
def data(ctx, input_dir, output_dir, stage):
    """Data processing pipeline."""
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            click.echo(f"Input directory not found: {input_dir}", err=True)
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        if stage in ['convert', 'all']:
            click.echo("Converting PDF files to text...")
            # Run PDF conversion
            import subprocess
            result = subprocess.run([
                'python', 'scripts/convert_pdf_to_txt.py',
                '--input', str(input_path),
                '--output', str(output_path / 'converted')
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                click.echo(f"PDF conversion failed: {result.stderr}", err=True)
            else:
                click.echo("✓ PDF conversion completed")
        
        if stage in ['embed', 'all']:
            click.echo("Generating embeddings...")
            # Run embedding generation
            import subprocess
            result = subprocess.run([
                'python', 'scripts/generate_embeddings.py',
                '--input', str(output_path / 'converted'),
                '--output', str(output_path / 'embeddings')
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                click.echo(f"Embedding generation failed: {result.stderr}", err=True)
            else:
                click.echo("✓ Embedding generation completed")
        
        if stage in ['preprocess', 'all']:
            click.echo("Preprocessing training data...")
            # Run preprocessing
            import subprocess
            result = subprocess.run([
                'python', 'src/training/preprocess.py',
                '--input', str(output_path / 'converted'),
                '--output', str(output_path / 'processed')
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                click.echo(f"Preprocessing failed: {result.stderr}", err=True)
            else:
                click.echo("✓ Preprocessing completed")
        
        click.echo(f"\nData processing completed. Results in: {output_path}")
        
    except Exception as e:
        click.echo(f"Error during data processing: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--module', required=True, 
              type=click.Choice(['recon', 'c2', 'postexploit', 'explainability', 'safety', 'all']))
@click.option('--config', help='Training configuration file')
@click.option('--data-dir', default='data/processed', help='Processed data directory')
@click.option('--output-dir', default='models/adapters', help='Output directory for trained adapters')
@click.pass_context
def train(ctx, module, config, data_dir, output_dir):
    """Train LoRA adapters."""
    try:
        click.echo(f"Training {module} adapter...")
        click.echo(f"Data directory: {data_dir}")
        click.echo(f"Output directory: {output_dir}")
        
        # This would call the actual training script
        click.echo("\n[TRAINING FEATURE IN DEVELOPMENT]")
        click.echo("Training pipeline will be available in the next release.")
        click.echo("Configure your training in configs/model_config.py")
        
    except Exception as e:
        click.echo(f"Error during training: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
def status():
    """Show system status and health check."""
    click.echo("Cyber-LLM System Status")
    click.echo("=" * 25)
    
    # Check components
    components = {
        'ReconAgent': True,
        'SafetyAgent': True,
        'ExplainabilityAgent': False,  # Not implemented yet
        'C2Agent': False,  # Not implemented yet
        'PostExploitAgent': False,  # Not implemented yet
        'Orchestrator': False,  # Not implemented yet
        'Training Pipeline': False,  # Not implemented yet
    }
    
    for component, status in components.items():
        status_icon = "✓" if status else "✗"
        status_text = "Available" if status else "In Development"
        click.echo(f"  {status_icon} {component}: {status_text}")
    
    # Check directories
    click.echo("\nDirectory Structure:")
    important_dirs = [
        'src/agents',
        'src/training',
        'src/evaluation',
        'configs',
        'scripts',
        'data'
    ]
    
    for dir_path in important_dirs:
        path = Path(dir_path)
        status_icon = "✓" if path.exists() else "✗"
        click.echo(f"  {status_icon} {dir_path}")

if __name__ == '__main__':
    cli()
