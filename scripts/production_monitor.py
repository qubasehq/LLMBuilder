#!/usr/bin/env python3
"""
Production monitoring and alerting script for LLMBuilder.
Monitors system health, performance metrics, and sends alerts.
"""

import os
import sys
import time
import json
import psutil
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class ProductionMonitor:
    """Production monitoring and alerting system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize production monitor.
        
        Args:
            config_path: Path to monitoring configuration file
        """
        self.config_path = config_path or "monitoring/monitor_config.json"
        self.config = self.load_config()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'logs/monitor.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.alerts_sent = {}
        self.last_check = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "check_interval": 60,
            "log_level": "INFO",
            "log_file": "logs/monitor.log",
            "thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "gpu_memory_percent": 90,
                "gpu_utilization": 95,
                "process_memory_mb": 8192,
                "pipeline_timeout_minutes": 120
            },
            "alerts": {
                "enabled": False,
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "",
                    "to_emails": []
                },
                "cooldown_minutes": 30
            },
            "processes": [
                "python",
                "llmbuilder"
            ],
            "directories": [
                "data",
                "exports",
                "logs"
            ]
        }\
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage.
        
        Returns:
            System resource metrics
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk_usage = {}
            for directory in self.config['directories']:
                if os.path.exists(directory):
                    disk = psutil.disk_usage(directory)
                    disk_usage[directory] = {
                        'total_gb': disk.total / (1024**3),
                        'used_gb': disk.used / (1024**3),
                        'free_gb': disk.free / (1024**3),
                        'percent': (disk.used / disk.total) * 100
                    }
            
            # GPU usage (if available)
            gpu_info = self.check_gpu_usage()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'disk_usage': disk_usage,
                'gpu_info': gpu_info
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {}
    
    def check_gpu_usage(self) -> Dict[str, Any]:
        """Check GPU usage using nvidia-smi.
        
        Returns:
            GPU usage information
        """
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {'available': False, 'error': 'nvidia-smi not available'}
            
            gpus = []
            for line in result.stdout.strip().split('\\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpu_info = {
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'memory_used_mb': int(parts[3]),
                            'memory_free_mb': int(parts[4]),
                            'utilization_percent': int(parts[5]),
                            'temperature_c': int(parts[6]),
                            'memory_percent': (int(parts[3]) / int(parts[2])) * 100
                        }
                        gpus.append(gpu_info)
            
            return {'available': True, 'gpus': gpus}
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            return {'available': False, 'error': str(e)}
    
    def check_processes(self) -> Dict[str, Any]:
        """Check LLMBuilder processes.
        
        Returns:
            Process information
        """
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    cmdline = ' '.join(proc_info['cmdline'] or [])
                    
                    # Check if this is a relevant process
                    is_relevant = any(
                        keyword in proc_info['name'].lower() or keyword in cmdline.lower()
                        for keyword in self.config['processes']
                    )
                    
                    if is_relevant:
                        memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
                        processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                            'memory_mb': memory_mb,
                            'cpu_percent': proc_info['cpu_percent']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'timestamp': datetime.now().isoformat(),
                'process_count': len(processes),
                'processes': processes,
                'total_memory_mb': sum(p['memory_mb'] for p in processes)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking processes: {e}")
            return {}
    
    def check_pipeline_health(self) -> Dict[str, Any]:
        """Check pipeline health and status.
        
        Returns:
            Pipeline health information
        """
        try:
            health_info = {
                'timestamp': datetime.now().isoformat(),
                'status': 'unknown',
                'last_activity': None,
                'errors': []
            }
            
            # Check log files for recent activity
            log_files = ['logs/ingestion.log', 'logs/training.log', 'logs/export.log']
            recent_activity = False
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        stat = os.stat(log_file)
                        last_modified = datetime.fromtimestamp(stat.st_mtime)
                        
                        if datetime.now() - last_modified < timedelta(hours=1):
                            recent_activity = True
                            health_info['last_activity'] = last_modified.isoformat()
                            
                        # Check for errors in recent log entries
                        with open(log_file, 'r') as f:
                            lines = f.readlines()[-100:]  # Last 100 lines
                            for line in lines:
                                if 'ERROR' in line.upper() or 'CRITICAL' in line.upper():
                                    health_info['errors'].append({
                                        'file': log_file,
                                        'message': line.strip()
                                    })
                                    
                    except Exception as e:
                        health_info['errors'].append({
                            'file': log_file,
                            'message': f"Could not read log file: {e}"
                        })
            
            # Determine overall status
            if health_info['errors']:
                health_info['status'] = 'error'
            elif recent_activity:
                health_info['status'] = 'active'
            else:
                health_info['status'] = 'idle'
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"Error checking pipeline health: {e}")
            return {'status': 'error', 'errors': [str(e)]}
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any metrics exceed thresholds.
        
        Args:
            metrics: System metrics
            
        Returns:
            List of threshold violations
        """
        violations = []
        thresholds = self.config['thresholds']
        
        # CPU threshold
        if metrics.get('cpu_percent', 0) > thresholds['cpu_percent']:
            violations.append({
                'type': 'cpu',
                'value': metrics['cpu_percent'],
                'threshold': thresholds['cpu_percent'],
                'message': f"CPU usage ({metrics['cpu_percent']:.1f}%) exceeds threshold ({thresholds['cpu_percent']}%)"
            })
        
        # Memory threshold
        if metrics.get('memory_percent', 0) > thresholds['memory_percent']:
            violations.append({
                'type': 'memory',
                'value': metrics['memory_percent'],
                'threshold': thresholds['memory_percent'],
                'message': f"Memory usage ({metrics['memory_percent']:.1f}%) exceeds threshold ({thresholds['memory_percent']}%)"
            })
        
        # Disk thresholds
        for directory, disk_info in metrics.get('disk_usage', {}).items():
            if disk_info['percent'] > thresholds['disk_percent']:
                violations.append({
                    'type': 'disk',
                    'directory': directory,
                    'value': disk_info['percent'],
                    'threshold': thresholds['disk_percent'],
                    'message': f"Disk usage in {directory} ({disk_info['percent']:.1f}%) exceeds threshold ({thresholds['disk_percent']}%)"
                })
        
        # GPU thresholds
        gpu_info = metrics.get('gpu_info', {})
        if gpu_info.get('available'):
            for gpu in gpu_info.get('gpus', []):
                if gpu['memory_percent'] > thresholds['gpu_memory_percent']:
                    violations.append({
                        'type': 'gpu_memory',
                        'gpu_index': gpu['index'],
                        'value': gpu['memory_percent'],
                        'threshold': thresholds['gpu_memory_percent'],
                        'message': f"GPU {gpu['index']} memory ({gpu['memory_percent']:.1f}%) exceeds threshold ({thresholds['gpu_memory_percent']}%)"
                    })
                
                if gpu['utilization_percent'] > thresholds['gpu_utilization']:
                    violations.append({
                        'type': 'gpu_utilization',
                        'gpu_index': gpu['index'],
                        'value': gpu['utilization_percent'],
                        'threshold': thresholds['gpu_utilization'],
                        'message': f"GPU {gpu['index']} utilization ({gpu['utilization_percent']:.1f}%) exceeds threshold ({thresholds['gpu_utilization']}%)"
                    })
        
        return violations
    
    def send_alert(self, violations: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Send alert notifications.
        
        Args:
            violations: List of threshold violations
            metrics: System metrics
        """
        if not self.config['alerts']['enabled']:
            return
        
        # Check cooldown period
        now = datetime.now()
        alert_key = 'general'
        
        if alert_key in self.alerts_sent:
            last_sent = self.alerts_sent[alert_key]
            cooldown = timedelta(minutes=self.config['alerts']['cooldown_minutes'])
            if now - last_sent < cooldown:
                return
        
        try:
            # Prepare email content
            subject = f"LLMBuilder Alert: {len(violations)} threshold violations"
            
            body = f"""
LLMBuilder Production Alert

Timestamp: {now.isoformat()}

Threshold Violations:
"""
            
            for violation in violations:
                body += f"- {violation['message']}\\n"
            
            body += f"""

System Summary:
- CPU: {metrics.get('cpu_percent', 'N/A'):.1f}%
- Memory: {metrics.get('memory_percent', 'N/A'):.1f}%
- GPU Available: {metrics.get('gpu_info', {}).get('available', False)}
"""
            
            if metrics.get('gpu_info', {}).get('available'):
                for gpu in metrics['gpu_info'].get('gpus', []):
                    body += f"- GPU {gpu['index']}: {gpu['utilization_percent']}% util, {gpu['memory_percent']:.1f}% memory\\n"
            
            # Send email
            self.send_email(subject, body)
            self.alerts_sent[alert_key] = now
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def send_email(self, subject: str, body: str):
        """Send email notification.
        
        Args:
            subject: Email subject
            body: Email body
        """
        email_config = self.config['alerts']['email']
        
        if not email_config['username'] or not email_config['to_emails']:
            self.logger.warning("Email configuration incomplete, skipping email alert")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = email_config['from_email'] or email_config['username']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            text = msg.as_string()
            server.sendmail(email_config['username'], email_config['to_emails'], text)
            server.quit()
            
            self.logger.info(f"Alert email sent to {email_config['to_emails']}")
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
    
    def generate_report(self, metrics: Dict[str, Any], process_info: Dict[str, Any], 
                       health_info: Dict[str, Any]) -> str:
        """Generate monitoring report.
        
        Args:
            metrics: System metrics
            process_info: Process information
            health_info: Pipeline health information
            
        Returns:
            Report string
        """
        report = f"""
LLMBuilder Production Monitoring Report
Generated: {datetime.now().isoformat()}

=== SYSTEM RESOURCES ===
CPU Usage: {metrics.get('cpu_percent', 'N/A'):.1f}%
Memory Usage: {metrics.get('memory_percent', 'N/A'):.1f}% ({metrics.get('memory_used_gb', 'N/A'):.1f}GB / {metrics.get('memory_total_gb', 'N/A'):.1f}GB)

Disk Usage:"""
        
        for directory, disk_info in metrics.get('disk_usage', {}).items():
            report += f"\\n  {directory}: {disk_info['percent']:.1f}% ({disk_info['used_gb']:.1f}GB / {disk_info['total_gb']:.1f}GB)"
        
        gpu_info = metrics.get('gpu_info', {})
        if gpu_info.get('available'):
            report += "\\n\\nGPU Information:"
            for gpu in gpu_info.get('gpus', []):
                report += f"\\n  GPU {gpu['index']} ({gpu['name']}): {gpu['utilization_percent']}% util, {gpu['memory_percent']:.1f}% memory ({gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB)"
        else:
            report += "\\n\\nGPU: Not available"
        
        report += f"""

=== PROCESSES ===
Active LLMBuilder Processes: {process_info.get('process_count', 0)}
Total Memory Usage: {process_info.get('total_memory_mb', 0):.1f}MB

Process Details:"""
        
        for proc in process_info.get('processes', [])[:5]:  # Show top 5 processes
            report += f"\\n  PID {proc['pid']}: {proc['name']} - {proc['memory_mb']:.1f}MB, {proc['cpu_percent']:.1f}% CPU"
        
        report += f"""

=== PIPELINE HEALTH ===
Status: {health_info.get('status', 'unknown').upper()}
Last Activity: {health_info.get('last_activity', 'N/A')}
Errors: {len(health_info.get('errors', []))}"""
        
        if health_info.get('errors'):
            report += "\\n\\nRecent Errors:"
            for error in health_info['errors'][-3:]:  # Show last 3 errors
                report += f"\\n  {error['file']}: {error['message'][:100]}..."
        
        return report
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        self.logger.info("Starting monitoring cycle")
        
        try:
            # Collect metrics
            metrics = self.check_system_resources()
            process_info = self.check_processes()
            health_info = self.check_pipeline_health()
            
            # Check thresholds
            violations = self.check_thresholds(metrics)
            
            # Log current status
            self.logger.info(f"System: CPU {metrics.get('cpu_percent', 0):.1f}%, Memory {metrics.get('memory_percent', 0):.1f}%, Processes: {process_info.get('process_count', 0)}")
            
            if violations:
                self.logger.warning(f"Found {len(violations)} threshold violations")
                for violation in violations:
                    self.logger.warning(violation['message'])
                
                # Send alerts
                self.send_alert(violations, metrics)
            
            # Generate and save report
            report = self.generate_report(metrics, process_info, health_info)
            
            # Save report to file
            report_dir = Path("monitoring/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Monitoring cycle complete. Report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def run_continuous(self):
        """Run continuous monitoring."""
        self.logger.info("Starting continuous monitoring")
        
        try:
            while True:
                self.run_monitoring_cycle()
                time.sleep(self.config['check_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLMBuilder production monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single monitoring cycle
  python scripts/production_monitor.py --once
  
  # Run continuous monitoring
  python scripts/production_monitor.py --continuous
  
  # Use custom config
  python scripts/production_monitor.py --config monitoring/custom_config.json
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to monitoring configuration file')
    parser.add_argument('--once', action='store_true', help='Run single monitoring cycle')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--report', action='store_true', help='Generate report only')
    
    args = parser.parse_args()
    
    try:
        monitor = ProductionMonitor(config_path=args.config)
        
        if args.once or args.report:
            monitor.run_monitoring_cycle()
        elif args.continuous:
            monitor.run_continuous()
        else:
            # Default: run once
            monitor.run_monitoring_cycle()
            
    except KeyboardInterrupt:
        print("\\nMonitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\nMonitoring failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()