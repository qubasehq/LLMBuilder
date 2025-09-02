"""
Example messaging tool template.
This demonstrates how to create a messaging/communication tool.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any


def send_message(
    recipient: str,
    message: str,
    channel: str = "default",
    priority: str = "normal"
) -> str:
    """Send a message to a recipient.
    
    Args:
        recipient: The message recipient (email, username, etc.)
        message: The message content
        channel: Communication channel (email, slack, sms, etc.)
        priority: Message priority (low, normal, high, urgent)
        
    Returns:
        Confirmation with message ID
    """
    if priority not in ["low", "normal", "high", "urgent"]:
        raise ValueError("Priority must be one of: low, normal, high, urgent")
    
    message_id = f"msg_{int(datetime.now().timestamp())}"
    timestamp = datetime.now().isoformat()
    
    # In a real implementation, this would integrate with actual messaging services
    message_data = {
        "id": message_id,
        "recipient": recipient,
        "message": message,
        "channel": channel,
        "priority": priority,
        "timestamp": timestamp,
        "status": "sent"
    }
    
    print(f"[MESSAGE SENT] {json.dumps(message_data, indent=2)}")
    
    return f"Message sent successfully. ID: {message_id}"


def send_broadcast(
    recipients: List[str],
    message: str,
    channel: str = "default"
) -> str:
    """Send a broadcast message to multiple recipients.
    
    Args:
        recipients: List of message recipients
        message: The message content
        channel: Communication channel
        
    Returns:
        Confirmation with broadcast details
    """
    broadcast_id = f"broadcast_{int(datetime.now().timestamp())}"
    
    results = []
    for recipient in recipients:
        msg_id = send_message(recipient, message, channel)
        results.append(f"{recipient}: {msg_id}")
    
    return f"Broadcast {broadcast_id} sent to {len(recipients)} recipients:\n" + "\n".join(results)


def schedule_message(
    recipient: str,
    message: str,
    send_at: str,
    channel: str = "default"
) -> str:
    """Schedule a message to be sent later.
    
    Args:
        recipient: The message recipient
        message: The message content
        send_at: When to send the message (ISO format: YYYY-MM-DD HH:MM:SS)
        channel: Communication channel
        
    Returns:
        Confirmation with scheduled message ID
    """
    try:
        scheduled_time = datetime.fromisoformat(send_at)
    except ValueError:
        raise ValueError("send_at must be in ISO format: YYYY-MM-DD HH:MM:SS")
    
    if scheduled_time <= datetime.now():
        raise ValueError("Scheduled time must be in the future")
    
    message_id = f"scheduled_{int(datetime.now().timestamp())}"
    
    # In a real implementation, this would use a task scheduler
    schedule_data = {
        "id": message_id,
        "recipient": recipient,
        "message": message,
        "channel": channel,
        "scheduled_for": send_at,
        "status": "scheduled"
    }
    
    print(f"[MESSAGE SCHEDULED] {json.dumps(schedule_data, indent=2)}")
    
    return f"Message scheduled successfully. ID: {message_id}, Send at: {send_at}"


def get_message_status(message_id: str) -> str:
    """Get the status of a sent message.
    
    Args:
        message_id: ID of the message to check
        
    Returns:
        Message status information
    """
    # In a real implementation, this would query the actual messaging system
    status_data = {
        "id": message_id,
        "status": "delivered",
        "sent_at": datetime.now().isoformat(),
        "delivered_at": datetime.now().isoformat()
    }
    
    return json.dumps(status_data, indent=2)


def create_message_template(
    name: str,
    template: str,
    variables: Optional[List[str]] = None
) -> str:
    """Create a reusable message template.
    
    Args:
        name: Template name
        template: Message template with {variable} placeholders
        variables: List of variable names used in the template
        
    Returns:
        Confirmation of template creation
    """
    if variables is None:
        # Extract variables from template
        import re
        variables = re.findall(r'\{(\w+)\}', template)
    
    template_data = {
        "name": name,
        "template": template,
        "variables": variables,
        "created_at": datetime.now().isoformat()
    }
    
    print(f"[TEMPLATE CREATED] {json.dumps(template_data, indent=2)}")
    
    return f"Message template '{name}' created with variables: {', '.join(variables)}"


def send_from_template(
    template_name: str,
    recipient: str,
    variables: Dict[str, str],
    channel: str = "default"
) -> str:
    """Send a message using a template.
    
    Args:
        template_name: Name of the template to use
        recipient: Message recipient
        variables: Dictionary of variable values for the template
        channel: Communication channel
        
    Returns:
        Confirmation of message sent
    """
    # In a real implementation, this would load the actual template
    # For demo purposes, we'll create a simple template
    template = "Hello {name}, your {item} is ready for pickup at {location}."
    
    try:
        message = template.format(**variables)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")
    
    return send_message(recipient, message, channel)


# Tool metadata for registration
TOOL_METADATA = {
    "name": "messaging_tool",
    "description": "Tool for sending messages and managing communications",
    "category": "messaging",
    "version": "1.0.0",
    "author": "LLMBuilder Team",
    "functions": [
        "send_message",
        "send_broadcast",
        "schedule_message",
        "get_message_status",
        "create_message_template",
        "send_from_template"
    ]
}