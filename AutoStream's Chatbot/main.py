import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Loading environment variables
load_dotenv()

from src.graph import build_graph

def main():
    memory = MemorySaver()
    graph = build_graph()
    
    app = graph.compile(checkpointer=memory)
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("--- AutoStream AI Agent (Type 'q' to quit) ---")
    print("Agent: Hi! I'm the AutoStream assistant. How can I help you today?")
    
    # 3. Chat Loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Exiting...")
                break
            
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            # Printing the final response from the agent
            for event in events:
                if "messages" in event:
                    # Get the last message
                    last_msg = event["messages"][-1]
                    
                    if hasattr(last_msg, 'type') and last_msg.type == "ai":
                        print(f"Agent: {last_msg.content}")
                        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()