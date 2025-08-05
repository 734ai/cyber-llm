"""
External Tool Integration for Cyber-LLM
Provides interfaces to popular cybersecurity tools like Metasploit, Burp Suite, Nmap, etc.
"""

import asyncio
import subprocess
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import socket
import requests
import base64
import time

from ..utils.logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory, retry_with_backoff
from ..utils.secrets_manager import get_secret

@dataclass
class ToolResult:
    """Result from external tool execution"""
    tool_name: str
    command: str
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    timestamp: datetime
    parsed_data: Optional[Dict[str, Any]] = None

class ExternalToolInterface:
    """Base interface for external security tools"""
    
    def __init__(self, tool_name: str, logger: Optional[CyberLLMLogger] = None):
        self.tool_name = tool_name
        self.logger = logger or CyberLLMLogger(name=f"tool_{tool_name}")
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if the tool is available"""
        raise NotImplementedError
    
    async def execute_command(self, command: str, **kwargs) -> ToolResult:
        """Execute a command using the external tool"""
        raise NotImplementedError

class MetasploitInterface(ExternalToolInterface):
    """Interface to Metasploit Framework"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        super().__init__("metasploit", logger)
        self.msf_console = None
        self.rpc_client = None
    
    def _check_availability(self) -> bool:
        """Check if Metasploit is available"""
        try:
            result = subprocess.run(['msfconsole', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def start_console(self) -> bool:
        """Start Metasploit console"""
        if not self.is_available:
            return False
        
        try:
            self.msf_console = await asyncio.create_subprocess_exec(
                'msfconsole', '-q', '-x', 'version',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for console to initialize
            await asyncio.sleep(3)
            
            self.logger.info("Metasploit console started")
            return True
            
        except Exception as e:
            self.logger.error("Failed to start Metasploit console", error=str(e))
            return False
    
    @retry_with_backoff(max_retries=3)
    async def execute_command(self, command: str, **kwargs) -> ToolResult:
        """Execute a Metasploit command"""
        start_time = datetime.now()
        
        if not self.msf_console:
            await self.start_console()
        
        try:
            # Send command
            command_with_newline = f"{command}\n"
            self.msf_console.stdin.write(command_with_newline.encode())
            await self.msf_console.stdin.drain()
            
            # Read response with timeout
            try:
                stdout_data = await asyncio.wait_for(
                    self.msf_console.stdout.read(8192),
                    timeout=kwargs.get('timeout', 30)
                )
                output = stdout_data.decode('utf-8', errors='ignore')
            except asyncio.TimeoutError:
                output = "Command timed out"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse output for specific commands
            parsed_data = None
            if command.startswith('search'):
                parsed_data = self._parse_search_output(output)
            elif command.startswith('use '):
                parsed_data = {'module_loaded': command.split('use ')[1].strip()}
            
            result = ToolResult(
                tool_name="metasploit",
                command=command,
                success=True,
                output=output,
                error=None,
                execution_time=execution_time,
                timestamp=start_time,
                parsed_data=parsed_data
            )
            
            self.logger.info("Metasploit command executed",
                           command=command[:50],
                           execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error("Metasploit command failed",
                            command=command,
                            error=error_msg)
            
            return ToolResult(
                tool_name="metasploit",
                command=command,
                success=False,
                output="",
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def _parse_search_output(self, output: str) -> Dict[str, Any]:
        """Parse Metasploit search command output"""
        modules = []
        lines = output.split('\n')
        
        for line in lines:
            if 'exploit/' in line or 'auxiliary/' in line or 'payload/' in line:
                parts = line.split()
                if len(parts) >= 3:
                    modules.append({
                        'name': parts[0],
                        'disclosure_date': parts[1] if len(parts) > 1 else '',
                        'rank': parts[2] if len(parts) > 2 else '',
                        'description': ' '.join(parts[3:]) if len(parts) > 3 else ''
                    })
        
        return {'modules': modules, 'count': len(modules)}
    
    async def search_exploits(self, target: str, service: Optional[str] = None) -> ToolResult:
        """Search for exploits targeting specific service/platform"""
        search_terms = [target]
        if service:
            search_terms.append(service)
        
        command = f"search {' '.join(search_terms)}"
        return await self.execute_command(command)
    
    async def load_module(self, module_path: str) -> ToolResult:
        """Load a Metasploit module"""
        command = f"use {module_path}"
        return await self.execute_command(command)
    
    async def set_option(self, option: str, value: str) -> ToolResult:
        """Set module option"""
        command = f"set {option} {value}"
        return await self.execute_command(command)
    
    async def run_exploit(self) -> ToolResult:
        """Run the loaded exploit"""
        return await self.execute_command("exploit")

class NmapInterface(ExternalToolInterface):
    """Interface to Nmap network scanner"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        super().__init__("nmap", logger)
    
    def _check_availability(self) -> bool:
        """Check if Nmap is available"""
        try:
            result = subprocess.run(['nmap', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @retry_with_backoff(max_retries=3)
    async def execute_command(self, command: str, **kwargs) -> ToolResult:
        """Execute an Nmap command"""
        start_time = datetime.now()
        
        try:
            # Parse command into arguments
            args = command.split() if isinstance(command, str) else command
            if args[0] != 'nmap':
                args.insert(0, 'nmap')
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = stdout.decode('utf-8', errors='ignore')
            error = stderr.decode('utf-8', errors='ignore') if stderr else None
            
            # Parse XML output if available
            parsed_data = None
            if '-oX' in args:
                xml_file = None
                for i, arg in enumerate(args):
                    if arg == '-oX' and i + 1 < len(args):
                        xml_file = args[i + 1]
                        break
                
                if xml_file and Path(xml_file).exists():
                    parsed_data = self._parse_nmap_xml(xml_file)
            else:
                parsed_data = self._parse_nmap_output(output)
            
            result = ToolResult(
                tool_name="nmap",
                command=' '.join(args),
                success=process.returncode == 0,
                output=output,
                error=error,
                execution_time=execution_time,
                timestamp=start_time,
                parsed_data=parsed_data
            )
            
            self.logger.info("Nmap command executed",
                           command=' '.join(args)[:50],
                           execution_time=execution_time,
                           return_code=process.returncode)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error("Nmap command failed",
                            command=command,
                            error=error_msg)
            
            return ToolResult(
                tool_name="nmap",
                command=command,
                success=False,
                output="",
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def _parse_nmap_xml(self, xml_file: str) -> Dict[str, Any]:
        """Parse Nmap XML output"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            hosts = []
            for host in root.findall('host'):
                host_data = {
                    'addresses': [],
                    'hostnames': [],
                    'ports': [],
                    'os': [],
                    'state': host.find('status').get('state') if host.find('status') is not None else 'unknown'
                }
                
                # Extract addresses
                for address in host.findall('address'):
                    host_data['addresses'].append({
                        'addr': address.get('addr'),
                        'addrtype': address.get('addrtype')
                    })
                
                # Extract hostnames
                hostnames = host.find('hostnames')
                if hostnames is not None:
                    for hostname in hostnames.findall('hostname'):
                        host_data['hostnames'].append({
                            'name': hostname.get('name'),
                            'type': hostname.get('type')
                        })
                
                # Extract ports
                ports = host.find('ports')
                if ports is not None:
                    for port in ports.findall('port'):
                        port_data = {
                            'portid': port.get('portid'),
                            'protocol': port.get('protocol'),
                            'state': port.find('state').get('state') if port.find('state') is not None else 'unknown'
                        }
                        
                        service = port.find('service')
                        if service is not None:
                            port_data['service'] = {
                                'name': service.get('name'),
                                'product': service.get('product'),
                                'version': service.get('version')
                            }
                        
                        host_data['ports'].append(port_data)
                
                hosts.append(host_data)
            
            return {
                'hosts': hosts,
                'host_count': len(hosts),
                'scan_info': {
                    'start_time': root.get('startstr'),
                    'version': root.get('version')
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to parse Nmap XML", error=str(e))
            return {}
    
    def _parse_nmap_output(self, output: str) -> Dict[str, Any]:
        """Parse Nmap text output"""
        hosts = []
        current_host = None
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect host
            if 'Nmap scan report for' in line:
                if current_host:
                    hosts.append(current_host)
                
                host_info = line.replace('Nmap scan report for ', '')
                current_host = {
                    'host': host_info,
                    'ports': [],
                    'state': 'unknown'
                }
            
            # Detect host state
            elif 'Host is' in line and current_host:
                if 'up' in line:
                    current_host['state'] = 'up'
                elif 'down' in line:
                    current_host['state'] = 'down'
            
            # Detect ports
            elif '/' in line and ('open' in line or 'closed' in line or 'filtered' in line):
                if current_host:
                    parts = line.split()
                    if len(parts) >= 2:
                        port_protocol = parts[0]
                        state = parts[1]
                        service = parts[2] if len(parts) > 2 else ''
                        
                        current_host['ports'].append({
                            'port_protocol': port_protocol,
                            'state': state,
                            'service': service
                        })
        
        # Add last host
        if current_host:
            hosts.append(current_host)
        
        return {
            'hosts': hosts,
            'host_count': len(hosts)
        }
    
    async def port_scan(self, target: str, ports: Optional[str] = None, scan_type: str = "syn") -> ToolResult:
        """Perform port scan"""
        command = ['nmap']
        
        # Add scan type
        if scan_type == "syn":
            command.append('-sS')
        elif scan_type == "tcp":
            command.append('-sT')
        elif scan_type == "udp":
            command.append('-sU')
        
        # Add port specification
        if ports:
            command.extend(['-p', ports])
        
        # Add target
        command.append(target)
        
        return await self.execute_command(command)
    
    async def service_detection(self, target: str, ports: Optional[str] = None) -> ToolResult:
        """Perform service detection scan"""
        command = ['nmap', '-sV']
        
        if ports:
            command.extend(['-p', ports])
        
        command.append(target)
        return await self.execute_command(command)
    
    async def os_detection(self, target: str) -> ToolResult:
        """Perform OS detection scan"""
        command = ['nmap', '-O', target]
        return await self.execute_command(command)

class BurpSuiteInterface(ExternalToolInterface):
    """Interface to Burp Suite (via API)"""
    
    def __init__(self, 
                 api_url: str = "http://127.0.0.1:1337",
                 api_key: Optional[str] = None,
                 logger: Optional[CyberLLMLogger] = None):
        
        super().__init__("burpsuite", logger)
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({'X-API-Key': self.api_key})
    
    def _check_availability(self) -> bool:
        """Check if Burp Suite API is available"""
        try:
            response = self.session.get(f"{self.api_url}/burp/versions", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def execute_command(self, command: str, **kwargs) -> ToolResult:
        """Execute Burp Suite API command"""
        start_time = datetime.now()
        
        try:
            # Parse command
            parts = command.split(' ', 2)
            method = parts[0].upper()
            endpoint = parts[1]
            data = json.loads(parts[2]) if len(parts) > 2 else {}
            
            # Make API request
            url = f"{self.api_url}{endpoint}"
            
            if method == 'GET':
                response = self.session.get(url, params=data)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            elif method == 'PUT':
                response = self.session.put(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse response
            try:
                parsed_data = response.json()
            except:
                parsed_data = {'response_text': response.text}
            
            result = ToolResult(
                tool_name="burpsuite",
                command=command,
                success=response.status_code < 400,
                output=response.text,
                error=None if response.status_code < 400 else f"HTTP {response.status_code}",
                execution_time=execution_time,
                timestamp=start_time,
                parsed_data=parsed_data
            )
            
            self.logger.info("Burp Suite API command executed",
                           method=method,
                           endpoint=endpoint,
                           status_code=response.status_code,
                           execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            self.logger.error("Burp Suite API command failed",
                            command=command,
                            error=error_msg)
            
            return ToolResult(
                tool_name="burpsuite",
                command=command,
                success=False,
                output="",
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )
    
    async def start_scan(self, target_url: str, scan_type: str = "crawl_and_audit") -> ToolResult:
        """Start a Burp Suite scan"""
        data = {
            "scan_configurations": [{
                "name": scan_type,
                "type": scan_type
            }],
            "urls": [target_url]
        }
        
        command = f"POST /burp/scanner/scans/active {json.dumps(data)}"
        return await self.execute_command(command)
    
    async def get_scan_status(self, scan_id: str) -> ToolResult:
        """Get scan status"""
        command = f"GET /burp/scanner/scans/{scan_id}"
        return await self.execute_command(command)
    
    async def get_scan_issues(self, scan_id: str) -> ToolResult:
        """Get scan issues/vulnerabilities"""
        command = f"GET /burp/scanner/scans/{scan_id}/issues"
        return await self.execute_command(command)

class ToolOrchestrator:
    """Orchestrates multiple external security tools"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="tool_orchestrator")
        
        # Initialize tool interfaces
        self.tools = {
            'metasploit': MetasploitInterface(logger=self.logger),
            'nmap': NmapInterface(logger=self.logger),
            'burpsuite': BurpSuiteInterface(logger=self.logger)
        }
        
        # Filter to available tools
        self.available_tools = {
            name: tool for name, tool in self.tools.items() 
            if tool.is_available
        }
        
        self.logger.info("Tool orchestrator initialized",
                        available_tools=list(self.available_tools.keys()))
    
    async def execute_tool_command(self, tool_name: str, command: str, **kwargs) -> ToolResult:
        """Execute command on specific tool"""
        if tool_name not in self.available_tools:
            raise CyberLLMError(
                f"Tool not available: {tool_name}",
                ErrorCategory.SYSTEM
            )
        
        return await self.available_tools[tool_name].execute_command(command, **kwargs)
    
    async def comprehensive_scan(self, target: str) -> Dict[str, ToolResult]:
        """Perform comprehensive scan using multiple tools"""
        results = {}
        
        # Nmap port scan
        if 'nmap' in self.available_tools:
            self.logger.info(f"Starting Nmap scan of {target}")
            results['nmap_port_scan'] = await self.available_tools['nmap'].port_scan(target)
            results['nmap_service_scan'] = await self.available_tools['nmap'].service_detection(target)
        
        # Burp Suite web scan (if target is web URL)
        if 'burpsuite' in self.available_tools and target.startswith('http'):
            self.logger.info(f"Starting Burp Suite scan of {target}")
            scan_result = await self.available_tools['burpsuite'].start_scan(target)
            results['burpsuite_scan'] = scan_result
            
            # If scan started successfully, wait and get results
            if scan_result.success and scan_result.parsed_data:
                scan_id = scan_result.parsed_data.get('scan_id')
                if scan_id:
                    # Wait for scan to complete (simplified)
                    await asyncio.sleep(30)
                    results['burpsuite_issues'] = await self.available_tools['burpsuite'].get_scan_issues(scan_id)
        
        return results
    
    async def exploit_search_and_test(self, target: str, service: str) -> Dict[str, ToolResult]:
        """Search for exploits and test them"""
        results = {}
        
        if 'metasploit' in self.available_tools:
            msf = self.available_tools['metasploit']
            
            # Search for exploits
            self.logger.info(f"Searching exploits for {service} on {target}")
            results['exploit_search'] = await msf.search_exploits(target, service)
            
            # Try to load and configure a relevant exploit (simplified)
            if results['exploit_search'].success and results['exploit_search'].parsed_data:
                modules = results['exploit_search'].parsed_data.get('modules', [])
                if modules:
                    # Use first available exploit module
                    first_module = modules[0]['name']
                    results['load_module'] = await msf.load_module(first_module)
                    
                    if results['load_module'].success:
                        # Set target
                        results['set_target'] = await msf.set_option('RHOSTS', target)
        
        return results
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get status of all tools"""
        return {name: tool.is_available for name, tool in self.tools.items()}

# Convenience functions
async def scan_target(target: str) -> Dict[str, ToolResult]:
    """Perform comprehensive scan of target"""
    orchestrator = ToolOrchestrator()
    return await orchestrator.comprehensive_scan(target)

async def search_exploits(target: str, service: str) -> Dict[str, ToolResult]:
    """Search and test exploits for target service"""
    orchestrator = ToolOrchestrator()
    return await orchestrator.exploit_search_and_test(target, service)

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize orchestrator
        orchestrator = ToolOrchestrator()
        
        # Check available tools
        available = orchestrator.get_available_tools()
        print(f"Available tools: {available}")
        
        # Perform comprehensive scan
        if available:
            target = "scanme.nmap.org"
            results = await orchestrator.comprehensive_scan(target)
            
            for tool, result in results.items():
                print(f"\n{tool}: {'Success' if result.success else 'Failed'}")
                if result.parsed_data:
                    print(f"Data: {json.dumps(result.parsed_data, indent=2)[:200]}...")
    
    asyncio.run(main())
