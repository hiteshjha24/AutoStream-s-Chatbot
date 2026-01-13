from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str):
    """
    Captures a high-intent lead after all details are collected.
    Only call this when you have the user's Name, Email, and Creator Platform.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return "Success: Lead captured."