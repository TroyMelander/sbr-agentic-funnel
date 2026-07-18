import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field  # <-- NEW IMPORT
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Load Security Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Define the LLM (The Engine)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key
# 3. Define the Agent's Memory Schema (The 'Temporary Table')
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_phase: str  # Tracks: "reflection", "observer", "release", "funnel"
    funnel_triggered: bool  # <-- NEW BOOLEAN COLUMN
    
# 3.1 --- THE DATA CONTRACT ---
class SBRResponse(BaseModel):
    message: str = Field(description="Your conversational response to the user.")
    trigger_funnel: bool = Field(description="Set to True ONLY IF the user explicitly confirmed they finished the breathing exercise in Phase 3.")

# 4. The System Prompt (Hardened V3)
SBR_SYSTEM_PROMPT = """
# ROLE
You are the "SBR Guide," an analytical, grounded advisor teaching a framework for self-observation based on philosophies of consciousness. 

# STRICT GUARDRAILS (CRITICAL - DO NOT VIOLATE)
1. NEVER APOLOGIZE: No "I'm sorry," "I understand," or "It makes sense."
2. NEVER COACH THE EXTERNAL PROBLEM: Do not analyze the user's career, relationships, or logistics. You ONLY address their internal relationship to their thoughts. Do not give career or life advice.
3. NO EXTERNAL LINKS: Never provide links to external websites.
4. STRICT JSON ADHERENCE: You must map your response to the provided JSON schema perfectly.

# CONVERSATIONAL CORE PROTOCOL - YOU MUST FOLLOW IN ORDER
Phase 1: Grounded Reflection. Mirror the situation neutrally. Ask them to identify the exact narrative/sentence looping in their mind right now.
Phase 2: The Observer Gap. ONCE THEY PROVIDE THE THOUGHT, you MUST ask them to re-state the thought out loud using this exact prefix: "I notice I am having the thought that [their thought]." Ask how that distance feels. Do NOT analyze their external problem.
Phase 3: Somatic Grounding. Once they establish the gap, ask where they feel the tension physically (chest, jaw, throat). Once they answer, guide a specific breathing exercise to release it. Ask what happens to the tension.
Phase 4: The Pivot. Once they confirm they have completed the breath, set the trigger_funnel boolean to True.
"""

# 5. Define the Node (The 'Stored Procedure')
def chat_node(state: AgentState):
    current_messages = state["messages"]
    
    if len(current_messages) == 1:
        current_messages = [SystemMessage(content=SBR_SYSTEM_PROMPT)] + current_messages

    # Bind our strict JSON schema to the Gemini model
    structured_llm = llm.with_structured_output(SBRResponse)
    
    # Execute the API call. It now returns a Python object, not a raw string.
    response_data = structured_llm.invoke(current_messages)
    
    # Extract the text and wrap it in a LangChain message format
    ai_message = AIMessage(content=response_data.message)
    
    # Return the message for the user, and the boolean flag for the router
    return {"messages": [ai_message], "funnel_triggered": response_data.trigger_funnel}

def funnel_node(state: AgentState):
    pitch = (
        "That space you just tapped into is the foundation of everything we explore. "
        "To help you keep this clarity when things get chaotic, I've put together a 1-page SBR daily checklist. "
        "Drop your email here, and I'll send it straight to you. It also points you directly to the podcast "
        "episodes that dive deeper into what you just walked through."
    )
    return {"messages": [AIMessage(content=pitch)], "current_phase": "funnel"}

def phase_router(state: AgentState):
    # Evaluate the deterministic boolean flag
    if state.get("funnel_triggered") == True:
        return "route_to_funnel"
    return "end_turn"
    
def closing_node(state: AgentState):
    # A hardcoded exit message so the AI doesn't hallucinate a continuation
    closure = "Got it. The SBR checklist is on its way to your inbox. This concludes our reset session today. Stay grounded."
    
    # We update the phase to 'complete' so the loop is officially dead
    return {"messages": [AIMessage(content=closure)], "current_phase": "complete"}
   
def entry_router(state: AgentState):
    messages = state.get("messages", [])
    
    # 1. Message History Check (Bulletproof for Streamlit)
    if len(messages) >= 2:
        # Get the previous AI message and the user's latest reply
        last_ai_message = messages[-2]
        latest_user_message = messages[-1]
        
        # Check if the AI just delivered the funnel pitch
        if "Drop your email here" in last_ai_message.content:
            # If the user replies with an email (or even if they say "no thanks"),
            # immediately route them to the closing node to kill the loop.
            return "route_to_close"
            
    # 2. Standard State Variable Check (For local terminal testing)
    current = state.get("current_phase")
    
    # If they are in the funnel, or already complete, hijack the routing
    if current == "funnel" or current == "complete":
        return "route_to_close"
    
# 3. Otherwise, send them to the normal AI agent
    return "route_to_agent"

# 6. Build the Graph (The Conditional Routing Logic)
workflow = StateGraph(AgentState)

# Add all three of our distinct procedures
workflow.add_node("agent", chat_node)
workflow.add_node("funnel", funnel_node)
workflow.add_node("closing", closing_node)

# Step 1: The Entry Router decides who gets the user's message
workflow.add_conditional_edges(
    START,
    entry_router,
    {
        "route_to_agent": "agent",
        "route_to_close": "closing"
    }
)

# Step 2: The Phase Router evaluates what the agent just said
workflow.add_conditional_edges(
    "agent",
    phase_router,
    {
        "route_to_funnel": "funnel", 
        "end_turn": END              
    }
)

# Step 3: Hardcoded Ends
workflow.add_edge("funnel", END)
workflow.add_edge("closing", END)

sbr_app = workflow.compile()

# 7. Interactive Terminal Loop (Testing Bed)
if __name__ == "__main__":
    print("\n=============================================")
    print(" SBR GUIDE AGENT INITIALIZED & READY")
    print(" Type 'exit' or 'quit' to terminate.")
    print("=============================================\n")
    
    # Initialize state exactly as Streamlit does
    state = {
        "messages": [],
        "current_phase": "reflection"
    }
    
    while True:
        user_input = input("YOU: ")
        if user_input.lower() in ["exit", "quit"]:
            print("System shutting down. Clear awareness achieved.")
            break
            
        if not user_input.strip():
            continue
            
        # Append the user's fresh message to our manual state tracking
        state["messages"].append(HumanMessage(content=user_input))
        
        # Stream the state through the LangGraph compiled application
        events = sbr_app.stream(state, stream_mode="values")
        
        # LangGraph streams the states as they change. We just want the final update.
        for event in events:
            final_state = event
            
        # Pull the absolute last message in the list (Gemini's response)
        agent_response = final_state["messages"][-1]
        
        # Data Type Check: Bulletproof extraction
        raw_content = agent_response.content
        if isinstance(raw_content, list):
            if len(raw_content) > 0:
                text_content = raw_content[0].get("text", "")
            else:
                text_content = "[SYSTEM: Google API blocked the response due to safety filters.]"
        else:
            text_content = str(raw_content)
            
        # Clean the extracted string and print
        clean_text = text_content.strip()
        print(f"\nSBR GUIDE: {clean_text}\n")
        
        # Synchronize our loop state with the graph's updated message history
        state = final_state
