"""
System metrics collection for LLMBuilder monitoring.

This module provides comprehensive system resource monitoring including
CPU, memory, disk, GPU, and network metrics with alerting capabilities.
"""

import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_temperature: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None


class MetricsCollector:
    """
    System metrics collector with alerting and export capabilities.
    
    Collects CPU, memory, disk, GPU, and network metrics at configurable
    intervals with support for alerts and data export.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metrics collector.
        
        Args:
            config: Configuration dictionary with collection settings
        """
        self.config = config
        self.duration = config.get('duration', 60)
        self.interval = config.get('interval', 1.0)
        self.include_gpu = config.get('include_gpu', True)
        self.include_network = config.get('include_network', False)
        self.alerts = config.get('alerts', {})
        
        self.snapshots: List[MetricSnapshot] = []
        self.collecting = False
        self.start_time = None
        
        # GPU availability check
        self.gpu_available = self._check_gpu_availability()
        
        logger.info("Metrics collector initialized")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics for the specified duration.
        
        Returns:
            Dictionary containing collected metrics and statistics
        """
        self.collecting = True
        self.start_time = time.time()
        self.snapshots = []
        
        logger.info(f"Starting metrics collection for {self.duration}s at {self.interval}s intervals")
        
        try:
            end_time = self.start_time + self.duration
            
            while time.time() < end_time and self.collecting:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                time.sleep(self.interval)
            
            # Calculate statistics
            return self._calculate_statistics()
            
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
            raise
        finally:
            self.collecting = False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics snapshot.
        
        Returns:
            Dictionary containing current metrics
        """
        try:
            snapshot = self._take_snapshot()
            
            return {
                'timestamp': snapshot.timestamp,
                'cpu': {
                    'current': snapshot.cpu_percent,
                    'avg': snapshot.cpu_percent,
                    'max': snapshot.cpu_percent
                },
                'memory': {
                    'current': snapshot.memory_percent,
                    'used_gb': snapshot.memory_used_gb,
                    'total_gb': snapshot.memory_total_gb,
                    'avg': snapshot.memory_percent,
                    'max': snapshot.memory_percent
                },
                'disk': {
                    'current': snapshot.disk_percent,
                    'used_gb': snapshot.disk_used_gb,
                    'total_gb': snapshot.disk_total_gb,
                    'avg': snapshot.disk_percent,
                    'max': snapshot.disk_percent
                }
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def save_metrics(self, metrics_data: Dict[str, Any], output_path: Path):
        """
        Save metrics data to file.
        
        Args:
            metrics_data: Metrics data to save
            output_path: Path to save the metrics file
        """
        try:
            export_data = {
                'collection_info': {
                    'start_time': self.start_time,
                    'duration': self.duration,
                    'interval': self.interval,
                    'total_snapshots': len(self.snapshots)
                },
                'config': self.config,
                'metrics': metrics_data,
                'raw_snapshots': [asdict(snapshot) for snapshot in self.snapshots]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise
    
    def stop_collection(self):
        """Stop ongoing metrics collection."""
        self.collecting = False
        logger.info("Metrics collection stopped")
    
    def _take_snapshot(self) -> MetricSnapshot:
        """Take a single metrics snapshot."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # GPU metrics (if available and enabled)
            gpu_utilization = None
            gpu_memory_used = None
            gpu_memory_total = None
            gpu_temperature = None
            
            if self.include_gpu and self.gpu_available:
                gpu_info = self._get_gpu_metrics()
                if gpu_info:
                    gpu_utilization = gpu_info.get('utilization', 0)
                    gpu_memory_used = gpu_info.get('memory_used', 0)
                    gpu_memory_total = gpu_info.get('memory_total', 0)
                    gpu_temperature = gpu_info.get('temperature', 0)
            
            # Network metrics (if enabled)
            network_bytes_sent = None
            network_bytes_recv = None
            
            if self.include_network:
                network_io = psutil.net_io_counters()
                network_bytes_sent = network_io.bytes_sent
                network_bytes_recv = network_io.bytes_recv
            
            return MetricSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_temperature=gpu_temperature,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
            
        except Exception as e:
            logger.error(f"Error taking metrics snapshot: {e}")
            # Return empty snapshot on error
            return MetricSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0
            )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            logger.debug("GPUtil not available - GPU monitoring disabled")
            return False
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")
            return False
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Get GPU metrics if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return None
            
            # Use first GPU for now
            gpu = gpus[0]
            
            return {
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            }
            
        except Exception as e:
            logger.debug(f"Error getting GPU metrics: {e}")
            return None
    
    def _check_alerts(self, snapshot: MetricSnapshot):
        """Check if any alert thresholds are exceeded."""
        alerts = []
        
        # CPU alert
        cpu_threshold = self.alerts.get('cpu')
        if cpu_threshold and snapshot.cpu_percent > cpu_threshold:
            alerts.append(f"CPU usage {snapshot.cpu_percent:.1f}% exceeds threshold {cpu_threshold}%")
        
        # Memory alert
        memory_threshold = self.alerts.get('memory')
        if memory_threshold and snapshot.memory_percent > memory_threshold:
            alerts.append(f"Memory usage {snapshot.memory_percent:.1f}% exceeds threshold {memory_threshold}%")
        
        # Disk alert
        disk_threshold = self.alerts.get('disk')
        if disk_threshold and snapshot.disk_percent > disk_threshold:
            alerts.append(f"Disk usage {snapshot.disk_percent:.1f}% exceeds threshold {disk_threshold}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from collected snapshots."""
        if not self.snapshots:
            return {}
        
        try:
            # CPU statistics
            cpu_values = [s.cpu_percent for s in self.snapshots]
            cpu_stats = {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            }
            
            # Memory statistics
            memory_values = [s.memory_percent for s in self.snapshots]
            memory_stats = {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current': memory_values[-1] if memory_values else 0,
                'total_gb': self.snapshots[-1].memory_total_gb
            }
            
            # Disk statistics
            disk_values = [s.disk_percent for s in self.snapshots]
            disk_stats = {
                'avg': sum(disk_values) / len(disk_values),
                'max': max(disk_values),
                'min': min(disk_values),
                'current': disk_values[-1] if disk_values else 0,
                'total_gb': self.snapshots[-1].disk_total_gb
            }
            
            result = {
                'collection_duration': len(self.snapshots) * self.interval,
                'total_snapshots': len(self.snapshots),
                'cpu': cpu_stats,
                'memory': memory_stats,
                'disk': disk_stats
            }
            
            # GPU statistics (if available)
            gpu_values = [s.gpu_utilization for s in self.snapshots if s.gpu_utilization is not None]
            if gpu_values:
                result['gpu'] = {
                    'avg': sum(gpu_values) / len(gpu_values),
                    'max': max(gpu_values),
                    'min': min(gpu_values),
                    'current': gpu_values[-1]
                }
            
            # Network statistics (if available)
            network_sent_values = [s.network_bytes_sent for s in self.snapshots if s.network_bytes_sent is not None]
            network_recv_values = [s.network_bytes_recv for s in self.snapshots if s.network_bytes_recv is not None]
            
            if network_sent_values and network_recv_values:
                # Calculate throughput (bytes per second)
                if len(network_sent_values) > 1:
                    sent_throughput = (network_sent_values[-1] - network_sent_values[0]) / (len(network_sent_values) * self.interval)
                    recv_throughput = (network_recv_values[-1] - network_recv_values[0]) / (len(network_recv_values) * self.interval)
                    
                    result['network'] = {
                        'sent_throughput_mbps': sent_throughput / (1024**2),
                        'recv_throughput_mbps': recv_throughput / (1024**2),
                        'total_sent_gb': network_sent_values[-1] / (1024**3),
                        'total_recv_gb': network_recv_values[-1] / (1024**3)
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}