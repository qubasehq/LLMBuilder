"""
Example alarm tool template.
This demonstrates how to create an alarm/notification tool.
"""

import time
from datetime import datetime, timedelta
from typing import Optional


def schedule_alarm(
    message: str,
    delay_seconds: int,
    repeat: bool = False,
    repeat_interval: Optional[int] = None
) -> str:
    """Schedule an alarm with a message.
    
    Args:
        message: The alarm message to display
        delay_seconds: Delay in seconds before the alarm triggers
        repeat: Whether to repeat the alarm
        repeat_interval: Interval in seconds for repeating (required if repeat=True)
        
    Returns:
        Confirmation message with alarm details
    """
    if repeat and repeat_interval is None:
        raise ValueError("repeat_interval is required when repeat=True")
    
    trigger_time = datetime.now() + timedelta(seconds=delay_seconds)
    
    # In a real implementation, this would integrate with the system's
    # notification system or task scheduler
    alarm_id = f"alarm_{int(time.time())}"
    
    result = f"Alarm '{alarm_id}' scheduled for {trigger_time.strftime('%Y-%m-%d %H:%M:%S')}"
    result += f"\nMessage: {message}"
    
    if repeat:
        result += f"\nRepeating every {repeat_interval} seconds"
    
    # Simulate scheduling (in real implementation, this would use a proper scheduler)
    print(f"[ALARM SCHEDULED] {result}")
    
    return result


def cancel_alarm(alarm_id: str) -> str:
    """Cancel a scheduled alarm.
    
    Args:
        alarm_id: ID of the alarm to cancel
        
    Returns:
        Confirmation message
    """
    # In a real implementation, this would cancel the actual alarm
    print(f"[ALARM CANCELLED] {alarm_id}")
    return f"Alarm '{alarm_id}' has been cancelled"


def list_alarms() -> str:
    """List all active alarms.
    
    Returns:
        List of active alarms
    """
    # In a real implementation, this would query the actual alarm system
    return "No active alarms (this is a demo implementation)"


def snooze_alarm(alarm_id: str, snooze_minutes: int = 5) -> str:
    """Snooze an active alarm.
    
    Args:
        alarm_id: ID of the alarm to snooze
        snooze_minutes: Minutes to snooze the alarm
        
    Returns:
        Confirmation message
    """
    new_time = datetime.now() + timedelta(minutes=snooze_minutes)
    result = f"Alarm '{alarm_id}' snoozed until {new_time.strftime('%H:%M:%S')}"
    
    print(f"[ALARM SNOOZED] {result}")
    return result


# Tool metadata for registration
TOOL_METADATA = {
    "name": "alarm_tool",
    "description": "Tool for scheduling and managing alarms and notifications",
    "category": "alarm",
    "version": "1.0.0",
    "author": "LLMBuilder Team",
    "functions": [
        "schedule_alarm",
        "cancel_alarm", 
        "list_alarms",
        "snooze_alarm"
    ]
}