"""
Workflow Analytics and Reporting System for Advanced Orchestration
Provides comprehensive analytics, performance metrics, and detailed reporting
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..utils.logging_system import CyberLLMLogger
from .advanced_workflows import WorkflowContext, WorkflowStatus

@dataclass
class WorkflowMetrics:
    """Comprehensive workflow execution metrics"""
    workflow_id: str
    template_name: str
    execution_time: float
    total_stages: int
    completed_stages: int
    failed_stages: int
    success_rate: float
    average_stage_time: float
    resource_utilization: Dict[str, float]
    agent_performance: Dict[str, Dict[str, float]]
    external_tool_performance: Dict[str, Dict[str, float]]
    adaptation_events: int
    rollback_events: int
    error_count: int
    warning_count: int

@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    total_workflows: int
    success_rate: float
    average_execution_time: float
    top_performing_templates: List[Dict[str, Any]]
    bottleneck_analysis: Dict[str, Any]
    trend_analysis: Dict[str, List[float]]
    recommendations: List[str]

class WorkflowAnalytics:
    """Advanced workflow analytics and reporting system"""
    
    def __init__(self, 
                 data_directory: str = "analytics_data",
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="workflow_analytics")
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True)
        
        # Analytics storage
        self.workflow_history: List[Dict[str, Any]] = []
        self.performance_metrics: List[WorkflowMetrics] = []
        self.agent_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.template_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Load existing data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical analytics data"""
        history_file = self.data_dir / "workflow_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.workflow_history = data.get("workflows", [])
                    
                self.logger.info(f"Loaded {len(self.workflow_history)} historical workflow records")
                
            except Exception as e:
                self.logger.error(f"Failed to load historical data", error=str(e))
    
    def record_workflow_execution(self, 
                                workflow_id: str,
                                template_name: str,
                                execution_result: Dict[str, Any],
                                context: WorkflowContext):
        """Record workflow execution for analytics"""
        
        # Extract metrics from execution result
        metrics = self._extract_workflow_metrics(
            workflow_id, template_name, execution_result, context
        )
        
        # Store workflow record
        workflow_record = {
            "workflow_id": workflow_id,
            "template_name": template_name,
            "executed_at": datetime.now().isoformat(),
            "duration": execution_result.get("duration", 0),
            "success": execution_result.get("overall_success", False),
            "stages_executed": execution_result.get("stages_executed", 0),
            "results": execution_result.get("results", {}),
            "metrics": asdict(metrics),
            "context_variables": context.variables,
            "adaptation_events": len(context.execution_history),
            "rollback_events": len(context.rollback_points)
        }
        
        self.workflow_history.append(workflow_record)
        self.performance_metrics.append(metrics)
        
        # Update statistics
        self._update_template_statistics(template_name, workflow_record)
        self._update_agent_statistics(execution_result.get("results", {}))
        
        # Save to disk
        self._save_analytics_data()
        
        self.logger.info(f"Recorded workflow execution: {workflow_id}")
    
    def _extract_workflow_metrics(self, 
                                workflow_id: str,
                                template_name: str,
                                execution_result: Dict[str, Any],
                                context: WorkflowContext) -> WorkflowMetrics:
        """Extract comprehensive metrics from workflow execution"""
        
        results = execution_result.get("results", {})
        
        # Calculate stage metrics
        total_stages = len(results)
        completed_stages = sum(1 for r in results.values() if r.get("success", False))
        failed_stages = total_stages - completed_stages
        success_rate = completed_stages / total_stages if total_stages > 0 else 0
        
        # Calculate timing metrics
        stage_times = [r.get("duration", 0) for r in results.values()]
        average_stage_time = np.mean(stage_times) if stage_times else 0
        
        # Extract agent performance
        agent_performance = {}
        for stage_name, stage_result in results.items():
            tasks = stage_result.get("tasks", {})
            for task_name, task_result in tasks.items():
                if "agent" in task_result.get("task_config", {}):
                    agent_name = task_result["task_config"]["agent"]
                    if agent_name not in agent_performance:
                        agent_performance[agent_name] = {
                            "success_rate": 0,
                            "avg_duration": 0,
                            "task_count": 0
                        }
                    
                    agent_performance[agent_name]["task_count"] += 1
                    if task_result.get("success", False):
                        agent_performance[agent_name]["success_rate"] += 1
                    
                    task_duration = self._parse_duration(
                        task_result.get("started_at", ""),
                        task_result.get("completed_at", "")
                    )
                    agent_performance[agent_name]["avg_duration"] += task_duration
        
        # Calculate final agent metrics
        for agent_name, metrics in agent_performance.items():
            if metrics["task_count"] > 0:
                metrics["success_rate"] = metrics["success_rate"] / metrics["task_count"]
                metrics["avg_duration"] = metrics["avg_duration"] / metrics["task_count"]
        
        # Extract external tool performance
        external_tool_performance = {}
        for stage_name, stage_result in results.items():
            tasks = stage_result.get("tasks", {})
            for task_name, task_result in tasks.items():
                if "external_tool" in task_result.get("task_config", {}):
                    tool_name = task_result["task_config"]["external_tool"]
                    if tool_name not in external_tool_performance:
                        external_tool_performance[tool_name] = {
                            "success_rate": 0,
                            "avg_duration": 0,
                            "usage_count": 0
                        }
                    
                    external_tool_performance[tool_name]["usage_count"] += 1
                    if task_result.get("success", False):
                        external_tool_performance[tool_name]["success_rate"] += 1
                    
                    task_duration = self._parse_duration(
                        task_result.get("started_at", ""),
                        task_result.get("completed_at", "")
                    )
                    external_tool_performance[tool_name]["avg_duration"] += task_duration
        
        # Calculate final tool metrics
        for tool_name, metrics in external_tool_performance.items():
            if metrics["usage_count"] > 0:
                metrics["success_rate"] = metrics["success_rate"] / metrics["usage_count"]
                metrics["avg_duration"] = metrics["avg_duration"] / metrics["usage_count"]
        
        # Resource utilization (simulated for now)
        resource_utilization = {
            "cpu_avg": np.random.uniform(20, 80),
            "memory_avg": np.random.uniform(30, 70),
            "network_usage": np.random.uniform(10, 50)
        }
        
        return WorkflowMetrics(
            workflow_id=workflow_id,
            template_name=template_name,
            execution_time=execution_result.get("duration", 0),
            total_stages=total_stages,
            completed_stages=completed_stages,
            failed_stages=failed_stages,
            success_rate=success_rate,
            average_stage_time=average_stage_time,
            resource_utilization=resource_utilization,
            agent_performance=agent_performance,
            external_tool_performance=external_tool_performance,
            adaptation_events=len(context.execution_history),
            rollback_events=len(context.rollback_points),
            error_count=len([e for r in results.values() for e in r.get("errors", [])]),
            warning_count=0  # Would be extracted from logs
        )
    
    def _parse_duration(self, start_time: str, end_time: str) -> float:
        """Parse duration between timestamp strings"""
        try:
            if start_time and end_time:
                start = datetime.fromisoformat(start_time)
                end = datetime.fromisoformat(end_time)
                return (end - start).total_seconds()
        except:
            pass
        return 0.0
    
    def _update_template_statistics(self, template_name: str, workflow_record: Dict[str, Any]):
        """Update template performance statistics"""
        
        if template_name not in self.template_statistics:
            self.template_statistics[template_name] = {
                "execution_count": 0,
                "success_count": 0,
                "total_duration": 0,
                "avg_stages_completed": 0,
                "adaptation_events": 0,
                "first_seen": workflow_record["executed_at"],
                "last_seen": workflow_record["executed_at"]
            }
        
        stats = self.template_statistics[template_name]
        stats["execution_count"] += 1
        stats["total_duration"] += workflow_record["duration"]
        stats["avg_stages_completed"] += workflow_record["stages_executed"]
        stats["adaptation_events"] += workflow_record["adaptation_events"]
        stats["last_seen"] = workflow_record["executed_at"]
        
        if workflow_record["success"]:
            stats["success_count"] += 1
    
    def _update_agent_statistics(self, stage_results: Dict[str, Any]):
        """Update agent performance statistics"""
        
        for stage_name, stage_result in stage_results.items():
            tasks = stage_result.get("tasks", {})
            for task_name, task_result in tasks.items():
                if "agent" in task_result.get("task_config", {}):
                    agent_name = task_result["task_config"]["agent"]
                    
                    if agent_name not in self.agent_statistics:
                        self.agent_statistics[agent_name] = {
                            "total_tasks": 0,
                            "successful_tasks": 0,
                            "total_duration": 0,
                            "error_count": 0
                        }
                    
                    stats = self.agent_statistics[agent_name]
                    stats["total_tasks"] += 1
                    
                    if task_result.get("success", False):
                        stats["successful_tasks"] += 1
                    
                    task_duration = self._parse_duration(
                        task_result.get("started_at", ""),
                        task_result.get("completed_at", "")
                    )
                    stats["total_duration"] += task_duration
                    stats["error_count"] += len(task_result.get("errors", []))
    
    def generate_performance_report(self, 
                                  days_back: int = 30,
                                  template_filter: Optional[str] = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Filter workflows by date range and template
        filtered_workflows = []
        for workflow in self.workflow_history:
            executed_at = datetime.fromisoformat(workflow["executed_at"])
            if start_date <= executed_at <= end_date:
                if not template_filter or workflow["template_name"] == template_filter:
                    filtered_workflows.append(workflow)
        
        if not filtered_workflows:
            self.logger.warning("No workflows found for report generation")
            return PerformanceReport(
                report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                time_period=(start_date, end_date),
                total_workflows=0,
                success_rate=0.0,
                average_execution_time=0.0,
                top_performing_templates=[],
                bottleneck_analysis={},
                trend_analysis={},
                recommendations=[]
            )
        
        # Calculate overall metrics
        total_workflows = len(filtered_workflows)
        successful_workflows = sum(1 for w in filtered_workflows if w["success"])
        success_rate = successful_workflows / total_workflows
        average_execution_time = np.mean([w["duration"] for w in filtered_workflows])
        
        # Top performing templates
        template_performance = defaultdict(lambda: {"count": 0, "success": 0, "avg_time": 0})
        for workflow in filtered_workflows:
            template = workflow["template_name"]
            template_performance[template]["count"] += 1
            template_performance[template]["avg_time"] += workflow["duration"]
            if workflow["success"]:
                template_performance[template]["success"] += 1
        
        top_templates = []
        for template, stats in template_performance.items():
            success_rate_template = stats["success"] / stats["count"]
            avg_time = stats["avg_time"] / stats["count"]
            
            top_templates.append({
                "template": template,
                "execution_count": stats["count"],
                "success_rate": success_rate_template,
                "average_time": avg_time,
                "score": success_rate_template * (1 / (avg_time + 1))  # Combined score
            })
        
        top_templates.sort(key=lambda x: x["score"], reverse=True)
        
        # Bottleneck analysis
        bottleneck_analysis = self._analyze_bottlenecks(filtered_workflows)
        
        # Trend analysis
        trend_analysis = self._analyze_trends(filtered_workflows, days_back)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            filtered_workflows, bottleneck_analysis, trend_analysis
        )
        
        report = PerformanceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            time_period=(start_date, end_date),
            total_workflows=total_workflows,
            success_rate=success_rate,
            average_execution_time=average_execution_time,
            top_performing_templates=top_templates[:5],
            bottleneck_analysis=bottleneck_analysis,
            trend_analysis=trend_analysis,
            recommendations=recommendations
        )
        
        self.logger.info(f"Generated performance report: {report.report_id}")
        return report
    
    def _analyze_bottlenecks(self, workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        
        stage_times = defaultdict(list)
        agent_times = defaultdict(list)
        tool_times = defaultdict(list)
        
        for workflow in workflows:
            results = workflow.get("results", {})
            for stage_name, stage_result in results.items():
                stage_duration = stage_result.get("duration", 0)
                stage_times[stage_name].append(stage_duration)
                
                tasks = stage_result.get("tasks", {})
                for task_name, task_result in tasks.items():
                    task_duration = self._parse_duration(
                        task_result.get("started_at", ""),
                        task_result.get("completed_at", "")
                    )
                    
                    if "agent" in task_result.get("task_config", {}):
                        agent_name = task_result["task_config"]["agent"]
                        agent_times[agent_name].append(task_duration)
                    
                    if "external_tool" in task_result.get("task_config", {}):
                        tool_name = task_result["task_config"]["external_tool"]
                        tool_times[tool_name].append(task_duration)
        
        # Find bottlenecks
        slowest_stages = sorted(
            [(stage, np.mean(times)) for stage, times in stage_times.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        slowest_agents = sorted(
            [(agent, np.mean(times)) for agent, times in agent_times.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        slowest_tools = sorted(
            [(tool, np.mean(times)) for tool, times in tool_times.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        return {
            "slowest_stages": [{"name": name, "avg_time": time} for name, time in slowest_stages],
            "slowest_agents": [{"name": name, "avg_time": time} for name, time in slowest_agents],
            "slowest_tools": [{"name": name, "avg_time": time} for name, time in slowest_tools]
        }
    
    def _analyze_trends(self, workflows: List[Dict[str, Any]], days_back: int) -> Dict[str, List[float]]:
        """Analyze performance trends over time"""
        
        # Group workflows by day
        daily_metrics = defaultdict(lambda: {"count": 0, "success": 0, "avg_time": 0})
        
        for workflow in workflows:
            executed_date = datetime.fromisoformat(workflow["executed_at"]).date()
            daily_metrics[executed_date]["count"] += 1
            daily_metrics[executed_date]["avg_time"] += workflow["duration"]
            if workflow["success"]:
                daily_metrics[executed_date]["success"] += 1
        
        # Calculate daily averages
        dates = sorted(daily_metrics.keys())
        daily_success_rates = []
        daily_avg_times = []
        daily_counts = []
        
        for date in dates:
            metrics = daily_metrics[date]
            success_rate = metrics["success"] / metrics["count"] if metrics["count"] > 0 else 0
            avg_time = metrics["avg_time"] / metrics["count"] if metrics["count"] > 0 else 0
            
            daily_success_rates.append(success_rate)
            daily_avg_times.append(avg_time)
            daily_counts.append(metrics["count"])
        
        return {
            "dates": [date.isoformat() for date in dates],
            "success_rates": daily_success_rates,
            "average_times": daily_avg_times,
            "workflow_counts": daily_counts
        }
    
    def _generate_recommendations(self, 
                                workflows: List[Dict[str, Any]],
                                bottlenecks: Dict[str, Any],
                                trends: Dict[str, List[float]]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Success rate recommendations
        overall_success_rate = np.mean([w["success"] for w in workflows])
        if overall_success_rate < 0.8:
            recommendations.append(
                f"Overall success rate is {overall_success_rate:.1%}. Consider reviewing and improving "
                "error handling and retry mechanisms."
            )
        
        # Performance recommendations
        avg_execution_time = np.mean([w["duration"] for w in workflows])
        if avg_execution_time > 1800:  # 30 minutes
            recommendations.append(
                f"Average execution time is {avg_execution_time/60:.1f} minutes. Consider optimizing "
                "slow stages and enabling more parallel execution."
            )
        
        # Bottleneck recommendations
        if bottlenecks["slowest_stages"]:
            slowest_stage = bottlenecks["slowest_stages"][0]
            recommendations.append(
                f"'{slowest_stage['name']}' stage is the slowest ({slowest_stage['avg_time']:.1f}s). "
                "Consider breaking it into smaller parallel tasks."
            )
        
        # Trend-based recommendations
        if len(trends["success_rates"]) >= 7:
            recent_trend = np.polyfit(range(7), trends["success_rates"][-7:], 1)[0]
            if recent_trend < -0.01:  # Declining success rate
                recommendations.append(
                    "Success rate has been declining recently. Investigate recent changes "
                    "and consider additional testing."
                )
        
        # Agent performance recommendations
        agent_success_rates = {}
        for workflow in workflows:
            results = workflow.get("results", {})
            for stage_result in results.values():
                tasks = stage_result.get("tasks", {})
                for task_result in tasks.values():
                    if "agent" in task_result.get("task_config", {}):
                        agent_name = task_result["task_config"]["agent"]
                        if agent_name not in agent_success_rates:
                            agent_success_rates[agent_name] = []
                        agent_success_rates[agent_name].append(task_result.get("success", False))
        
        for agent_name, successes in agent_success_rates.items():
            success_rate = np.mean(successes)
            if success_rate < 0.7:
                recommendations.append(
                    f"{agent_name} has a low success rate ({success_rate:.1%}). "
                    "Consider reviewing its implementation and training data."
                )
        
        if not recommendations:
            recommendations.append("System performance looks good! Continue monitoring for any changes.")
        
        return recommendations
    
    def create_performance_dashboard(self, 
                                   report: PerformanceReport,
                                   output_file: str = "performance_dashboard.html"):
        """Create interactive performance dashboard"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Success Rate Trend', 'Execution Time Trend',
                'Template Performance', 'Stage Performance',
                'Agent Performance', 'Resource Utilization'
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # Success rate trend
        if report.trend_analysis.get("dates"):
            fig.add_trace(
                go.Scatter(
                    x=report.trend_analysis["dates"],
                    y=report.trend_analysis["success_rates"],
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
        
        # Execution time trend
        if report.trend_analysis.get("dates"):
            fig.add_trace(
                go.Scatter(
                    x=report.trend_analysis["dates"],
                    y=report.trend_analysis["average_times"],
                    mode='lines+markers',
                    name='Avg Execution Time',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
        
        # Template performance
        if report.top_performing_templates:
            templates = [t["template"] for t in report.top_performing_templates]
            success_rates = [t["success_rate"] for t in report.top_performing_templates]
            
            fig.add_trace(
                go.Bar(
                    x=templates,
                    y=success_rates,
                    name='Template Success Rate',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Bottleneck analysis
        if report.bottleneck_analysis.get("slowest_stages"):
            stages = [s["name"] for s in report.bottleneck_analysis["slowest_stages"]]
            times = [s["avg_time"] for s in report.bottleneck_analysis["slowest_stages"]]
            
            fig.add_trace(
                go.Bar(
                    x=stages,
                    y=times,
                    name='Stage Avg Time',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # Agent performance (if available)
        if hasattr(self, 'agent_statistics') and self.agent_statistics:
            agents = list(self.agent_statistics.keys())
            success_rates = [
                self.agent_statistics[agent]["successful_tasks"] / 
                max(self.agent_statistics[agent]["total_tasks"], 1)
                for agent in agents
            ]
            
            fig.add_trace(
                go.Bar(
                    x=agents,
                    y=success_rates,
                    name='Agent Success Rate',
                    marker_color='purple'
                ),
                row=3, col=1
            )
        
        # Resource utilization pie chart (simulated)
        resource_types = ['CPU', 'Memory', 'Network', 'Storage']
        resource_usage = [25, 35, 20, 20]
        
        fig.add_trace(
            go.Pie(
                labels=resource_types,
                values=resource_usage,
                name="Resource Usage"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Workflow Performance Dashboard - {report.report_id}",
            height=900,
            showlegend=True
        )
        
        # Save dashboard
        dashboard_path = self.data_dir / output_file
        fig.write_html(str(dashboard_path))
        
        self.logger.info(f"Created performance dashboard: {dashboard_path}")
        return str(dashboard_path)
    
    def _save_analytics_data(self):
        """Save analytics data to disk"""
        try:
            # Save workflow history
            history_data = {
                "workflows": self.workflow_history,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_dir / "workflow_history.json", 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            # Save performance metrics
            metrics_data = [asdict(metric) for metric in self.performance_metrics]
            with open(self.data_dir / "performance_metrics.json", 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            # Save statistics
            with open(self.data_dir / "template_statistics.json", 'w') as f:
                json.dump(dict(self.template_statistics), f, indent=2, default=str)
            
            with open(self.data_dir / "agent_statistics.json", 'w') as f:
                json.dump(dict(self.agent_statistics), f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error("Failed to save analytics data", error=str(e))
    
    def export_report(self, 
                     report: PerformanceReport,
                     format: str = "json",
                     filename: Optional[str] = None) -> str:
        """Export performance report in various formats"""
        
        if not filename:
            filename = f"{report.report_id}.{format}"
        
        output_path = self.data_dir / filename
        
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            elif format.lower() == "csv":
                # Convert to DataFrame for CSV export
                df_data = []
                for template in report.top_performing_templates:
                    df_data.append({
                        "template": template["template"],
                        "execution_count": template["execution_count"],
                        "success_rate": template["success_rate"],
                        "average_time": template["average_time"]
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
            elif format.lower() == "markdown":
                markdown_content = self._generate_markdown_report(report)
                with open(output_path, 'w') as f:
                    f.write(markdown_content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported report to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export report", error=str(e))
            raise
    
    def _generate_markdown_report(self, report: PerformanceReport) -> str:
        """Generate markdown formatted report"""
        
        md_lines = [
            f"# Workflow Performance Report",
            f"**Report ID:** {report.report_id}",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Time Period:** {report.time_period[0].strftime('%Y-%m-%d')} to {report.time_period[1].strftime('%Y-%m-%d')}",
            "",
            "## Executive Summary",
            f"- **Total Workflows:** {report.total_workflows}",
            f"- **Overall Success Rate:** {report.success_rate:.1%}",
            f"- **Average Execution Time:** {report.average_execution_time:.1f} seconds",
            "",
            "## Top Performing Templates",
        ]
        
        for i, template in enumerate(report.top_performing_templates, 1):
            md_lines.extend([
                f"### {i}. {template['template']}",
                f"- **Executions:** {template['execution_count']}",
                f"- **Success Rate:** {template['success_rate']:.1%}",
                f"- **Average Time:** {template['average_time']:.1f}s",
                ""
            ])
        
        md_lines.extend([
            "## Performance Bottlenecks",
            "### Slowest Stages"
        ])
        
        for stage in report.bottleneck_analysis.get("slowest_stages", []):
            md_lines.append(f"- **{stage['name']}:** {stage['avg_time']:.1f}s")
        
        md_lines.extend([
            "",
            "### Slowest Agents"
        ])
        
        for agent in report.bottleneck_analysis.get("slowest_agents", []):
            md_lines.append(f"- **{agent['name']}:** {agent['avg_time']:.1f}s")
        
        md_lines.extend([
            "",
            "## Recommendations"
        ])
        
        for i, recommendation in enumerate(report.recommendations, 1):
            md_lines.append(f"{i}. {recommendation}")
        
        return "\n".join(md_lines)

# Example usage
if __name__ == "__main__":
    # Initialize analytics system
    analytics = WorkflowAnalytics()
    
    # Generate performance report
    report = analytics.generate_performance_report(days_back=7)
    
    # Create dashboard
    dashboard_path = analytics.create_performance_dashboard(report)
    print(f"Dashboard created: {dashboard_path}")
    
    # Export report
    json_report = analytics.export_report(report, "json")
    markdown_report = analytics.export_report(report, "markdown")
    
    print(f"Reports exported:")
    print(f"- JSON: {json_report}")
    print(f"- Markdown: {markdown_report}")
