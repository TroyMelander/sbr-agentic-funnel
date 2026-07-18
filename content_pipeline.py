import os
from datetime import datetime
from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# 1. Load Security Key & Engine
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

# 2. Define the Pipeline State (The Data Record)
class PipelineState(TypedDict):
    raw_transcript: str
    core_theme: str
    roommate_narrative: str
    vessel_reaction: str
    tiktok_script: str  
    newsletter_body: str  
    social_thread: str  # <-- NEW: Holding tank for Twitter/LinkedIn

# 3. The Extraction Contract
class ExtractedTheme(BaseModel):
    core_theme: str = Field(description="A 1-sentence summary of the main philosophical friction discussed.")
    roommate_narrative: str = Field(description="The specific intrusive thought, false expectation, or ego-driven narrative looping in the mind.")
    vessel_reaction: str = Field(description="The physical or emotional symptom the person is feeling (e.g., burnout, racing heart, anger).")

# 4. The Prompts
EXTRACTION_PROMPT = """
You are the lead content strategist for 'The Social Battery Reset'.
Your job is to read a raw podcast transcript and extract the core psychological conflict using the SBR philosophy.

Identify:
1. The 'Roommate Narrative': The specific ego-driven, invasive thought, or false expectation causing mental noise.
2. The 'Vessel Reaction': The physical or emotional tension/symptom the person is experiencing.
3. The Core Theme: A concise summary of the fundamental friction discussed.

Do not summarize the logistics or tangents of the transcript. Isolate the internal narrative and the somatic symptom.
"""

HOOK_PROMPT = """
You are the lead social media copywriter for the podcast 'The Social Battery Reset'.
Write a punchy, 60-second vertical video script based strictly on the provided SBR data.

Follow this exact blueprint:
1. The Hook (Validate the Symptom): Start immediately with the physical/emotional friction (The Vessel Reaction). Make it visceral.
2. The Pivot (Introduce the Concept): Introduce the specific intrusive thought (The Roommate Narrative). Explain that this is just a looping script, not reality. Use the phrase "seat of awareness" at least once.
3. The Promise (The ROI): Give a quick somatic action to reset, and tell them to listen to the new episode to debug this mental system.

Keep it analytical, grounded, and observant. Do not use emojis. Output only the spoken script.
"""

NEWSLETTER_PROMPT = """
You are the lead content strategist for 'The Social Battery Reset' newsletter.
Write a deep-dive, structured email based on the provided SBR data.

Follow this blueprint:
1. The Subject Line: Catchy, curious, and related to the psychological friction.
2. The Symptom: Start by validating the physical/emotional symptom (The Vessel Reaction).
3. The Anatomy of the Noise: Break down the specific intrusive thought (The Roommate Narrative). Explain why the brain clings to this unreliable narrative.
4. The Somatic Reset: Provide a clear, actionable breathing or grounding technique to break the fusion with the thought.
5. The Sign-Off: Warm, grounded, and encouraging.

Maintain a compassionate, observational tone. Speak to the human experience of finding distance from our thoughts. Avoid technical or computing metaphors. Do not use emojis. Output only the final email text.
"""

THREAD_PROMPT = """
You are the lead social media strategist for 'The Social Battery Reset'.
Write a high-impact, scannable Twitter/LinkedIn thread (3-5 short posts) based on the SBR data.

Blueprint:
1. Hook Post: Call out the Symptom (Vessel Reaction) and the specific thought (Roommate Narrative).
2. Philosophy Post: Explain that the physical tension is an alarm system, and the thought is just a default script, not reality.
3. Action Post: Provide a quick somatic grounding technique to break the loop.
4. CTA Post: Tell them to listen to the new episode to debug the system.

Keep it punchy, tech-adjacent, and analytical. Separate posts with "---". Output only the thread text.
"""

# 5. The Nodes (Workers)
def extraction_node(state: PipelineState):
    raw_text = state["raw_transcript"]
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"Analyze this transcript and extract the data:\n\n{raw_text}")
    ]
    extractor = llm.with_structured_output(ExtractedTheme)
    result = extractor.invoke(messages)
    return {
        "core_theme": result.core_theme,
        "roommate_narrative": result.roommate_narrative,
        "vessel_reaction": result.vessel_reaction
    }

def clean_llm_output(response_content):
    if isinstance(response_content, list) and len(response_content) > 0:
        return response_content[0].get("text", "")
    return str(response_content)

def hook_node(state: PipelineState):
    context = f"CORE THEME: {state.get('core_theme')}\nROOMMATE: {state.get('roommate_narrative')}\nVESSEL: {state.get('vessel_reaction')}"
    messages = [SystemMessage(content=HOOK_PROMPT), HumanMessage(content=context)]
    return {"tiktok_script": clean_llm_output(llm.invoke(messages).content)}

def newsletter_node(state: PipelineState):
    context = f"CORE THEME: {state.get('core_theme')}\nROOMMATE: {state.get('roommate_narrative')}\nVESSEL: {state.get('vessel_reaction')}"
    messages = [SystemMessage(content=NEWSLETTER_PROMPT), HumanMessage(content=context)]
    return {"newsletter_body": clean_llm_output(llm.invoke(messages).content)}

def thread_node(state: PipelineState):
    context = f"CORE THEME: {state.get('core_theme')}\nROOMMATE: {state.get('roommate_narrative')}\nVESSEL: {state.get('vessel_reaction')}"
    messages = [SystemMessage(content=THREAD_PROMPT), HumanMessage(content=context)]
    return {"social_thread": clean_llm_output(llm.invoke(messages).content)}

# 6. Build the Assembly Line
if __name__ == "__main__":
    builder = StateGraph(PipelineState)
    
    # Add nodes
    builder.add_node("extract", extraction_node)
    builder.add_node("hook", hook_node)
    builder.add_node("newsletter", newsletter_node)
    builder.add_node("thread", thread_node) # <-- NEW
    
    # Route data
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "hook")
    builder.add_edge("extract", "newsletter")
    builder.add_edge("extract", "thread")   # <-- NEW
    builder.add_edge("hook", END)
    builder.add_edge("newsletter", END)
    builder.add_edge("thread", END)         # <-- NEW
    
    pipeline = builder.compile()
    
    # 7. Run the Job
    dummy_transcript = (
        "Hey everyone, welcome back to the Social Battery Reset. It's Troy. Look, today I want to talk about "
        "getting ghosted. Last week I had plans with a friend, we were supposed to go surfing, and they just never showed up. "
        "No text, nothing. And immediately, my mind starts racing. The voice in my head just starts screaming, "
        "'See? People don't respect your time. You're not a priority to anyone.' And I could feel my stomach just tie "
        "into a knot. My jaw was clenched. I was so mad. But then I realized, wait, I'm letting my Roommate completely "
        "ruin my morning over an expectation."
    )
    
    print("Running SBR ETL Pipeline... (This may take a few seconds as it generates 3 assets simultaneously)\n")
    final_state = pipeline.invoke({"raw_transcript": dummy_transcript})
    
    # 8. The "Load" Phase: Save to a File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SBR_Content_Pack_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("      SBR CONTENT ASSET GENERATOR       \n")
        f.write("========================================\n\n")
        
        f.write("--- 1. EXTRACTED DATA ---\n")
        f.write(f"CORE THEME: {final_state['core_theme']}\n")
        f.write(f"ROOMMATE NARRATIVE: {final_state['roommate_narrative']}\n")
        f.write(f"VESSEL REACTION: {final_state['vessel_reaction']}\n\n")
        
        f.write("========================================\n")
        f.write("--- 2. TIKTOK/SHORTS SCRIPT ---\n\n")
        f.write(f"{final_state['tiktok_script']}\n\n")
        
        f.write("========================================\n")
        f.write("--- 3. NEWSLETTER DRAFT ---\n\n")
        f.write(f"{final_state['newsletter_body']}\n\n")
        
        f.write("========================================\n")
        f.write("--- 4. SOCIAL MEDIA THREAD ---\n\n")
        f.write(f"{final_state['social_thread']}\n")
        
    print(f"Success! Content package saved to your folder as: {filename}")