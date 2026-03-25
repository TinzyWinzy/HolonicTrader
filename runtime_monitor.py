
# RUNTIME MONITORING SYSTEM
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque

class RuntimeMonitor:
    def __init__(self, health_config_file="system_health_config.json"):
        self.health_config = self.load_config(health_config_file)
        self.error_counts = defaultdict(int)
        self.error_timestamps = defaultdict(deque)
        self.running = False
        self.monitor_thread = None
        
    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "error_monitoring": {"critical_error_threshold_per_hour": 3},
                "runtime_monitoring": {"check_interval_seconds": 30}
            }
    
    def record_error(self, error_type="GENERAL_ERROR"):
        """Record an error occurrence"""
        now = datetime.now()
        
        # Add timestamp to deque
        self.error_timestamps[error_type].append(now)
        self.error_counts[error_type] += 1
        
        # Clean old errors (older than 1 hour)
        cutoff_time = now - timedelta(hours=1)
        while (self.error_timestamps[error_type] and 
               self.error_timestamps[error_type][0] < cutoff_time):
            self.error_timestamps[error_type].popleft()
        
        # Check if we've exceeded threshold
        threshold = self.health_config.get("error_monitoring", {}).get("critical_error_threshold_per_hour", 3)
        
        if len(self.error_timestamps[error_type]) >= threshold:
            self.trigger_critical_alert(error_type, len(self.error_timestamps[error_type]))
    
    def trigger_critical_alert(self, error_type, count):
        """Trigger critical system alert"""
        alert_msg = f"CRITICAL: {error_type} errors exceeded threshold ({count} in last hour)"
        print(f"ALERT: {alert_msg}")
        
        # Check if auto-shutdown is enabled
        if self.health_config.get("error_monitoring", {}).get("auto_shutdown_on_critical", False):
            print("Initiating graceful shutdown due to critical errors...")
            self.initiate_shutdown()
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown"""
        # This would integrate with your main trading loop
        print("RUNTIME MONITOR: Shutdown signal sent")
        
        # Set flag for main loop to check
        with open(".shutdown_flag", "w") as f:
            f.write(datetime.now().isoformat())
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Runtime monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Runtime monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        check_interval = self.health_config.get("runtime_monitoring", {}).get("check_interval_seconds", 30)
        
        while self.running:
            try:
                # Check system health
                self.perform_health_check()
                time.sleep(check_interval)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def perform_health_check(self):
        """Perform system health check"""
        # Check for shutdown flag
        if os.path.exists(".shutdown_flag"):
            print("Shutdown flag detected - stopping monitor")
            self.running = False
            return
        
        # Add more health checks here (memory, CPU, disk space, etc.)
        
    def get_status(self):
        """Get current monitoring status"""
        return {
            "running": self.running,
            "error_counts": dict(self.error_counts),
            "monitoring_duration": "active" if self.running else "stopped"
        }

# Global instance
runtime_monitor = RuntimeMonitor()
