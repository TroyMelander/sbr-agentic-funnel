import streamlit as st
from langchain_core.messages import HumanMessage
from agent import sbr_app, AgentState # Importing your compiled LangGraph app

# 1. Page Configuration
st.set_page_config(page_title="SBR Guide", page_icon="🔋", layout="centered")
st.title("The Social Battery Reset")
st.markdown("### Observe the Roommate. Release the Vessel.")

# 2. Initialize Session Memory (Streamlit re-runs the script on every click, so we must store state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Render Chat History to the Screen
for message in st.session_state.messages:
    # Check if the message was sent by the AI or the Human
    role = "assistant" if message.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# 4. The User Input Field
user_input = st.chat_input("What is looping in your mind right now?")

if user_input:
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Store the user's message in Streamlit's memory
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Format the state for LangGraph
    graph_state: AgentState = {"messages": st.session_state.messages, "current_phase": "reflection"}
    
    # 5. Stream the Data through your LangGraph Agent
    with st.spinner("The Guide is reflecting..."):
        events = sbr_app.stream(graph_state, stream_mode="values")
        
        # Capture the final state
        for event in events:
            final_state = event
            
        # Extract the AI's response
        agent_response = final_state["messages"][-1]
        raw_content = agent_response.content
        
        # Data Type Check (The bulletproof fix we built earlier)
        if isinstance(raw_content, list):
            if len(raw_content) > 0:
                text_content = raw_content[0].get("text", "")
            else:
                text_content = "Google API blocked the response due to safety filters."
        else:
            text_content = str(raw_content)
            
        clean_text = text_content.replace("[FUNNEL_TRIGGER]", "").strip()

    # 6. Display the AI's Response
    with st.chat_message("assistant"):
        st.markdown(clean_text)
        
    # Store the AI's response in memory
    st.session_state.messages.append(agent_response)