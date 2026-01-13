import operator
from typing import Annotated, List, TypedDict, Optional

from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The state of the agent graph.
    """
    # using add reducer to add the message in history
    messages: Annotated[List[BaseMessage], operator.add]

    #lead information slot; "None" untill collected from user 
    user_name: Optional[str]
    user_email: Optional[str]
    user_platform: Optional[str]

    intent: Optional[str]  #finds the intent of the user
    lead_captured: bool
