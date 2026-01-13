import os
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from src.state import AgentState
from src.rag import setup_rag_retriever
from src.tools import mock_lead_capture
from dotenv import load_dotenv
load_dotenv()

#intializing LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
convert_system_message_to_human=True
)

retriever = setup_rag_retriever()

class IntentClassification(BaseModel):
    """Classify the user's intent."""
    intent: Literal["greeting", "product_inquiry", "high_intent"] = Field(
        ..., description="The user's primary intent."
    )

class LeadInfoExtraction(BaseModel):
    """Extract lead details if present."""
    name: str | None = Field(None, description="User's name")
    email: str | None = Field(None, description="User's email address")
    platform: str | None = Field(None, description="Content creation platform (YouTube, Instagram, etc)")

def classify_input_node(state: AgentState):
    """
    Analyzes the conversation history to determine the user's intent.
    """
    messages = state['messages']
    classifier = llm.with_structured_output(IntentClassification)

    system_prompt = """You are an AI assistant for AutoStream, a video editing SaaS.
    Classify the user's intent into one of these categories:
    1. greeting: Casual hellos or introductions.
    2. product_inquiry: Questions about pricing, features, plans, or policies.
    3. high_intent: User expresses interest in buying, signing up, trying the Pro plan, or gives personal info.
    
    Current Conversation:
    """

    #adding system prompt with history for the model to understand context

    response = classifier.invoke([SystemMessage(content=system_prompt)] + messages[-3:])
    
    return {"intent": response.intent}

def greeting_node(state: AgentState):
    """Responds to casual greetings."""
    return {"messages": [AIMessage(content="Hi there! I'm the AutoStream assistant. How can I help you with your video editing needs today?")]}

def rag_node(state: AgentState):
    """
    Retrieves information and answers product questions.
    """
    last_message = state['messages'][-1].content

    #retreiving relevant context from RAG
    docs = retriever.invoke(last_message)
    context = "\n\n".join([d.page_content for d in docs])

    #generating answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for AutoStream. Use the following context to answer the user's question. If the answer isn't in the context, say you don't know. Keep it concise.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": last_message})
    
    return {"messages": [response]}

def lead_capture_node(state: AgentState):
    """
    Handles the lead capture process (Slot Filling).
    1. Extracts info from current message.
    2. Checks what is missing.
    3. Asks for missing info OR calls tool.
    """
    messages = state['messages']
    last_message = messages[-1].content

    #extract nly if user said something new
    if isinstance(messages[-1], HumanMessage):
        extractor = llm.with_structured_output(LeadInfoExtraction)
        extraction_prompt = "Extract user name, email, and platform from the text if present. If not, return null."
        extracted_data = extractor.invoke([SystemMessage(content=extraction_prompt), HumanMessage(content=last_message)])
        
        # Update state only if new info is found
        updates = {}
        if extracted_data.name and not state.get('user_name'):
            updates['user_name'] = extracted_data.name
        if extracted_data.email and not state.get('user_email'):
            updates['user_email'] = extracted_data.email
        if extracted_data.platform and not state.get('user_platform'):
            updates['user_platform'] = extracted_data.platform

        if updates:
            pass 
    else:
        updates = {}
    
    # merging current state with potential updates to check real-time status

    current_name = updates.get('user_name') or state.get('user_name')
    current_email = updates.get('user_email') or state.get('user_email')
    current_platform = updates.get('user_platform') or state.get('user_platform')

    # Case 1st: All data collected -> Execute Tool
    if current_name and current_email and current_platform:
        # Calling the tool
        tool_output = mock_lead_capture.invoke({
            "name": current_name, 
            "email": current_email, 
            "platform": current_platform
        })
        
        # Creating final response
        response_text = f"Thanks {current_name}! I've registered your interest for the Pro plan on {current_platform}. Our team will contact you at {current_email} shortly."
        
        #setting a flag 'lead_captured' to True for cleaning up state later
        updates['lead_captured'] = True
        updates['messages'] = [AIMessage(content=response_text)]
        return updates
    

    # Case 2nd: Missing Data -> Ask for it
    missing_fields = []
    if not current_name: missing_fields.append("name")
    if not current_email: missing_fields.append("email")
    if not current_platform: missing_fields.append("platform (e.g., YouTube, Instagram)")
    
    #asking for the first missing field to keep conversation natural
    next_needed = missing_fields[0]
    
    if next_needed == "name":
        question = "Great! To get you set up, could I please have your full name?"
    elif next_needed == "email":
        question = f"Thanks {current_name}. What is the best email address to reach you?"
    else:
        question = "One last thing: which content platform do you primarily use (e.g., YouTube, Instagram)?"
        
    updates['messages'] = [AIMessage(content=question)]
    
    updates['intent'] = 'high_intent' 
    
    return updates

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Adding Nodes
    workflow.add_node("classifier", classify_input_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("lead_manager", lead_capture_node)
    
    #the Entry Point
    workflow.set_entry_point("classifier")
    
    # Defining Conditional Edges
    def route_intent(state: AgentState):
        intent = state.get("intent")
        if intent == "greeting":
            return "greeting"
        elif intent == "high_intent":
            return "high_intent"
        elif intent == "product_inquiry":
            return "product_inquiry"
        else:
            return "greeting" # Fallback

    workflow.add_conditional_edges(
        "classifier",
        route_intent,
        {
            # Key (Intent) -> Value (Node Name)
            "greeting": "greeting",
            "product_inquiry": "rag",         
            "high_intent": "lead_manager"    
        }
    )
    
    # All response nodes go to END
    workflow.add_edge("greeting", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("lead_manager", END)
    
    return workflow