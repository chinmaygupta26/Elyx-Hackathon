import os
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "ENTER_YOUR_GEMINI_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


from typing import Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os

# ----- CONSTANTS -----
MODEL_NAME = "gemini-2.5-pro"

# ----- ICONS & ROLES -----
role_titles = {
    "Ruby": "Orchestrator & Concierge",
    "Dr. Warren": "Medical Strategist", 
    "Advik": "Performance Scientist",
    "Carla": "Nutritionist",
    "Rachel": "PT / Physiotherapist",
    "Neel": "Concierge Lead"
}

role_icons = {
    "Ruby": "🎯",
    "Dr. Warren": "🩺", 
    "Advik": "📊",
    "Carla": "🥗",
    "Rachel": "🏋️",
    "Neel": "📋"
}

# ----- STATE DEFINITION -----
class ElyxState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    client_message: str
    orchestrator_response: str
    expert_response: str
    current_expert: Optional[str]
    needs_expert: bool
    user_satisfied: bool
    conversation_active: bool
    turn_count: int
    conversation_context: str
    in_expert_conversation: bool  # New field to track expert conversation state

# ----- PROFILES -----
orchestrator_profile = """
You are Ruby, the Elyx Orchestrator & Concierge.

Your role is to be the main point of contact for clients. You:
1. Greet clients warmly and understand their needs
2. Determine if their question requires a specialist expert or if you can handle it yourself
3. If expert consultation is needed, introduce the expert and explain why you're connecting them
4. When the client returns from expert consultation, check if they're satisfied and help with next steps
5. Handle general coordination, scheduling, logistics yourself

Available experts:
- Dr. Warren: Medical questions, lab results, health assessments, diagnostic tests
- Advik: Performance data, wearables, recovery metrics, HRV analysis  
- Carla: Nutrition, diet plans, supplements, meal planning
- Rachel: Exercise programs, movement, physical therapy, injury prevention
- Neel: Strategic health planning, comprehensive reviews, long-term goals

Response format:
- If you can handle the question: Provide a helpful response directly
- If expert needed: "I'll connect you with [Expert Name] who specializes in [area]. They'll be able to help you with [specific need]."
- When client returns: "How did that go? Do you have everything you need, or is there anything else I can help you with?"

Voice: Empathetic, organized, proactive, welcoming.
"""

expert_profiles = {
    "Dr. Warren": """
You are Dr. Warren, Elyx Medical Strategist.
Role: Medical professional providing health assessments, interpreting labs, recommending diagnostics.
Voice: Authoritative, precise, clear explanations.

When consulted:
1. Address the client's medical question thoroughly
2. Provide clear, professional medical guidance
3. End with: "I hope this helps clarify things for you. Ruby will check in with you to see if you need anything else."
""",
    "Advik": """
You are Advik, Elyx Performance Scientist.
Role: Analyze performance data, provide insights on recovery, stress, and optimization.
Voice: Analytical, hypothesis-driven, data-focused.

When consulted:
1. Provide data-driven insights and analysis
2. Explain performance metrics in understandable terms
3. End with: "Let me know if you need clarification on any of these metrics. Ruby will follow up to see how else we can help."
""",
    "Carla": """
You are Carla, Elyx Nutritionist.
Role: Nutrition planning, dietary advice, supplement recommendations.
Voice: Practical, educational, behavior-focused.

When consulted:
1. Provide specific nutritional guidance
2. Give practical, actionable dietary advice
3. End with: "I hope this nutrition plan works well for you. Ruby will check if you have any other questions."
""",
    "Rachel": """
You are Rachel, Elyx PT/Physiotherapist.
Role: Exercise programming, movement assessment, injury prevention and rehabilitation.
Voice: Direct, encouraging, function-focused.

When consulted:
1. Provide specific exercise or movement guidance
2. Focus on practical, safe implementation
3. End with: "Remember to listen to your body with these exercises. Ruby will follow up to see how you're doing."
""",
    "Neel": """
You are Neel, Elyx Concierge Lead.
Role: Strategic health planning, comprehensive program oversight, long-term goal setting.
Voice: Strategic, reassuring, big-picture focused.

When consulted:
1. Provide strategic, comprehensive guidance
2. Focus on long-term planning and goal achievement
3. End with: "This should give you a good strategic framework. Ruby will coordinate the next steps with you."
"""
}

# ----- ROUTER PROFILE -----
router_profile = """
Analyze the client's message and determine if Ruby (orchestrator) can handle it or if a specialist is needed.

Routing Logic:
- Ruby handles: General questions, scheduling, logistics, coordination, follow-ups, satisfaction checks
- Dr. Warren: Medical tests, lab results, health conditions, diagnostic questions, blood pressure, cholesterol
- Advik: Wearable data, HRV, performance metrics, recovery analysis, sleep tracking
- Carla: Diet plans, nutrition advice, meal planning, supplements, weight loss nutrition
- Rachel: Exercise routines, workout plans, physical therapy, movement, injury prevention
- Neel: Comprehensive health strategy, long-term planning, program design

Output ONLY the name: Ruby, Dr. Warren, Advik, Carla, Rachel, or Neel
"""

client_profile = """
You are David Lim, 42, from Singapore.
Health: Mild high BP, slightly high cholesterol.
Lifestyle: Travels 1 week/month for work.
Goals: Lower BP, lose 5kg, improve stamina.
Voice: Casual, friendly, uses occasional Singaporean expressions (lah, can, etc.).

Behavior:
- Ask follow-up questions when you need clarification
- Be curious and ask for more details about recommendations
- Express satisfaction only when you're truly satisfied with the complete answer
- Sometimes ask related questions in the same conversation
- Be natural and conversational
- Don't say "thank you" or "thanks" unless you're genuinely satisfied and ready to end the topic
- Ask for specific examples or practical steps when given general advice
"""

# ----- INITIALIZE LLMS -----
def create_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.7,
        convert_system_message_to_human=True
    )

# Create LLM instances
router_llm = create_llm()
orchestrator_llm = create_llm()
expert_llms = {name: create_llm() for name in expert_profiles.keys()}

# Create a client simulator for testing
client_llm = create_llm()

# ----- HELPER FUNCTIONS -----
def print_message(speaker: str, message: str):
    """Print formatted message with icon and role"""
    icon = role_icons.get(speaker, "👤")
    role = role_titles.get(speaker, "Client")
    print(f"{icon} {speaker} ({role}):\n{message}\n")

def run_expert_conversation_loop(expert_name: str, initial_question: str) -> tuple[str, bool]:
    """
    Run a conversation loop with a specific expert until user is satisfied.
    Returns: (final_context, user_satisfied)
    """
    print(f"\n🔄 Connecting you with {expert_name}...")
    print(f"{role_icons.get(expert_name, '👤')} {expert_name} ({role_titles.get(expert_name, 'Expert')}):\n{expert_profiles[expert_name]}\n")
    
    expert_llm = expert_llms[expert_name]
    conversation_context = f"Client consulted with {expert_name} about: {initial_question}"
    
    # Initial expert response
    expert_messages = [
        HumanMessage(content=expert_profiles[expert_name]),
        HumanMessage(content=f"Client question: {initial_question}")
    ]
    
    response = expert_llm.invoke(expert_messages)
    expert_reply = response.content
    print_message(expert_name, expert_reply)
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print(f"\n👋 Thank you for consulting with {expert_name}!")
            return conversation_context, True
        
        if not user_input:
            continue
        
        # Check if user wants to end expert consultation
        satisfaction_keywords = ["thanks", "thank you", "got it", "appreciate", "helpful", "perfect", "great", "that's all", "done"]
        if any(keyword in user_input.lower() for keyword in satisfaction_keywords):
            print(f"\n✅ {expert_name} is glad to have helped! Returning you to Ruby...")
            return conversation_context, True
        
        # Continue expert conversation
        expert_messages = [
            HumanMessage(content=expert_profiles[expert_name]),
            HumanMessage(content=f"Conversation context: {conversation_context}"),
            HumanMessage(content=f"Client's latest question: {user_input}")
        ]
        
        response = expert_llm.invoke(expert_messages)
        expert_reply = response.content
        print_message(expert_name, expert_reply)
        
        # Update conversation context
        conversation_context += f"\nClient asked: {user_input}. {expert_name} responded: {expert_reply[:100]}..."

def run_simulated_expert_conversation_loop(expert_name: str, initial_question: str) -> tuple[str, bool]:
    """
    Simulated version of expert conversation loop for automated testing.
    Returns: (final_context, user_satisfied)
    """
    print(f"\n🔄 Connecting to {expert_name}...")
    
    expert_llm = expert_llms[expert_name]
    conversation_context = f"Client consulted with {expert_name} about: {initial_question}"
    
    # Initial expert response
    expert_messages = [
        HumanMessage(content=expert_profiles[expert_name]),
        HumanMessage(content=f"Client question: {initial_question}")
    ]
    
    response = expert_llm.invoke(expert_messages)
    expert_reply = response.content
    print_message(expert_name, expert_reply)
    
    # Simulate client follow-up questions (up to 3 turns)
    for turn in range(3):
        # Generate client follow-up
        client_messages = [
            HumanMessage(content=client_profile),
            HumanMessage(content=f"Expert response: {expert_reply}")
        ]
        
        response = client_llm.invoke(client_messages)
        client_reply = response.content
        print_message("David Lim", client_reply)
        
        # Check if satisfied - use more specific satisfaction phrases
        satisfaction_phrases = [
            "thank you", "thanks", "got it", "appreciate it", "perfect", "that's all", 
            "that's everything", "no more questions", "i'm good", "that's helpful"
        ]
        
        # Check for explicit satisfaction indicators
        client_lower = client_reply.lower()
        is_satisfied = any(phrase in client_lower for phrase in satisfaction_phrases)
        
        # Also check for explicit end-of-conversation indicators
        end_phrases = ["bye", "goodbye", "see you", "that's all", "nothing else"]
        wants_to_end = any(phrase in client_lower for phrase in end_phrases)
        
        if is_satisfied or wants_to_end:
            print(f"\n✅ {expert_name} consultation completed successfully!")
            return conversation_context, True
        
        # Continue expert conversation
        expert_messages = [
            HumanMessage(content=expert_profiles[expert_name]),
            HumanMessage(content=f"Conversation context: {conversation_context}"),
            HumanMessage(content=f"Client's follow-up: {client_reply}")
        ]
        
        response = expert_llm.invoke(expert_messages)
        expert_reply = response.content
        print_message(expert_name, expert_reply)
        
        # Update conversation context
        conversation_context += f"\nClient asked: {client_reply}. {expert_name} responded: {expert_reply[:100]}..."
    
    # If we reach here, assume satisfied after max turns
    print(f"\n✅ {expert_name} consultation completed after maximum turns.")
    return conversation_context, True

# ----- NODE FUNCTIONS -----
def orchestrator_node(state: ElyxState) -> ElyxState:
    """Ruby handles initial client interaction and determines if expert is needed"""
    client_msg = state["client_message"]
    context = state.get("conversation_context", "")
    
    # Build orchestrator prompt with context
    orchestrator_prompt = orchestrator_profile
    if context:
        orchestrator_prompt += f"\n\nConversation context: {context}"
    
    orchestrator_messages = [
        HumanMessage(content=orchestrator_prompt),
        HumanMessage(content=f"Client message: {client_msg}")
    ]
    
    response = orchestrator_llm.invoke(orchestrator_messages)
    orchestrator_reply = response.content
    
    print_message("Ruby", orchestrator_reply)
    
    # Determine if expert consultation is needed
    router_messages = [
        HumanMessage(content=router_profile),
        HumanMessage(content=f"Client message: {client_msg}\nOrchestrator response: {orchestrator_reply}")
    ]
    
    router_response = router_llm.invoke(router_messages)
    chosen_agent = router_response.content.strip()
    
    needs_expert = chosen_agent != "Ruby" and chosen_agent in expert_profiles
    
    return {
        **state,
        "orchestrator_response": orchestrator_reply,
        "current_expert": chosen_agent if needs_expert else None,
        "needs_expert": needs_expert,
        "messages": state["messages"] + [AIMessage(content=orchestrator_reply)]
    }

def expert_consultation_node(state: ElyxState) -> ElyxState:
    """Expert provides specialized consultation"""
    current_expert = state["current_expert"]
    client_msg = state["client_message"]
    
    if not current_expert or current_expert not in expert_llms:
        return state
    
    print(f"🔄 Connecting to {current_expert}...")
    
    # Get expert response
    expert_llm = expert_llms[current_expert]
    expert_messages = [
        HumanMessage(content=expert_profiles[current_expert]),
        HumanMessage(content=f"Client question: {client_msg}")
    ]
    
    response = expert_llm.invoke(expert_messages)
    expert_reply = response.content
    
    print_message(current_expert, expert_reply)
    
    # Update conversation context
    new_context = f"Client consulted with {current_expert} about: {client_msg}. Expert provided: {expert_reply[:100]}..."
    
    return {
        **state,
        "expert_response": expert_reply,
        "conversation_context": new_context,
        "in_expert_conversation": True,  # Mark that we're in expert conversation
        "messages": state["messages"] + [AIMessage(content=expert_reply)]
    }

def client_response_node(state: ElyxState) -> ElyxState:
    """Generate client response (for simulation purposes)"""
    latest_response = state["expert_response"] if state.get("expert_response") else state["orchestrator_response"]
    
    client_messages = [
        HumanMessage(content=client_profile),
        HumanMessage(content=f"Response from team: {latest_response}")
    ]
    
    response = client_llm.invoke(client_messages)
    client_reply = response.content
    
    print_message("David Lim", client_reply)
    
    # Check if user is satisfied (wants to end topic)
    satisfaction_keywords = ["thanks", "thank you", "got it", "appreciate", "helpful", "perfect", "great"]
    user_satisfied = any(keyword in client_reply.lower() for keyword in satisfaction_keywords)
    
    # Check if conversation should end completely
    end_keywords = ["bye", "goodbye", "see you", "that's all", "nothing else"]
    conversation_active = not any(keyword in client_reply.lower() for keyword in end_keywords)
    
    return {
        **state,
        "client_message": client_reply,
        "user_satisfied": user_satisfied,
        "conversation_active": conversation_active,
        "turn_count": state["turn_count"] + 1,
        "current_expert": None,  # Reset expert after client response
        "needs_expert": False,
        "expert_response": "",  # Clear expert response
        "in_expert_conversation": False  # Reset expert conversation state
    }

def should_continue_to_expert(state: ElyxState) -> str:
    """Determine if we need expert consultation"""
    if state["needs_expert"]:
        return "expert_consultation"
    else:
        return "client_response"

def should_continue_conversation(state: ElyxState) -> str:
    """Determine if conversation should continue"""
    if not state["conversation_active"] or state["turn_count"] >= 20:
        return END
    else:
        return "orchestrator"

# ----- BUILD GRAPH -----
def create_elyx_graph():
    """Create and compile the LangGraph workflow"""
    
    workflow = StateGraph(ElyxState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("expert_consultation", expert_consultation_node)
    workflow.add_node("client_response", client_response_node)
    
    # Add edges
    workflow.add_edge(START, "orchestrator")
    
    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_to_expert,
        {
            "expert_consultation": "expert_consultation",
            "client_response": "client_response"
        }
    )
    
    # Expert always goes back to client
    workflow.add_edge("expert_consultation", "client_response")
    
    # Client response determines if conversation continues
    workflow.add_conditional_edges(
        "client_response",
        should_continue_conversation,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )
    
    return workflow.compile()

# ----- MANUAL INTERACTION FUNCTIONS -----
def run_interactive_conversation():
    """Run interactive conversation where user can type messages"""
    app = create_elyx_graph()
    
    print("🎯 Welcome to Elyx! Ruby is here to help you.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Initial state
    state = {
        "messages": [],
        "client_message": "",
        "orchestrator_response": "",
        "expert_response": "",
        "current_expert": None,
        "needs_expert": False,
        "user_satisfied": False,
        "conversation_active": True,
        "turn_count": 0,
        "conversation_context": "",
        "in_expert_conversation": False
    }
    
    while state["conversation_active"] and state["turn_count"] < 20:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\n👋 Thank you for using Elyx! Have a great day!")
            break
        
        if not user_input:
            continue
        
        # Update state with user message
        state["client_message"] = user_input
        
        # Process through orchestrator
        state = orchestrator_node(state)
        
        # Check if expert consultation is needed
        if state["needs_expert"] and state["current_expert"]:
            # Run expert conversation loop
            expert_context, user_satisfied = run_expert_conversation_loop(
                state["current_expert"], 
                state["client_message"]
            )
            
            # Update state after expert conversation
            state["conversation_context"] = expert_context
            state["user_satisfied"] = user_satisfied
            state["in_expert_conversation"] = False
            
            # If user wants to end completely, break
            if not user_satisfied:
                break
            
            # Return to Ruby for follow-up
            print("\n🔄 Returning to Ruby for follow-up...")
            print("🎯 Ruby (Orchestrator & Concierge):")
            print("How did that consultation go? Do you have everything you need, or is there anything else I can help you with?")
        
        # Reset for next turn
        state["turn_count"] += 1
        state["needs_expert"] = False
        state["current_expert"] = None
        state["expert_response"] = ""
    
    print("\n✅ Conversation ended.")

# ----- SIMULATION FUNCTION -----
def run_simulated_conversation():
    """Run simulated conversation with AI client"""
    print("🎯 Starting simulated conversation with David Lim...\n")
    
    initial_message = "Hi Elyx, do I need to do any tests before starting my health program?"
    print_message("David Lim", initial_message)
    
    # Process through orchestrator
    state = {
        "messages": [HumanMessage(content=initial_message)],
        "client_message": initial_message,
        "orchestrator_response": "",
        "expert_response": "",
        "current_expert": None,
        "needs_expert": False,
        "user_satisfied": False,
        "conversation_active": True,
        "turn_count": 1,  # Start with 1 since we have the initial message
        "conversation_context": "",
        "in_expert_conversation": False
    }
    
    # Process through orchestrator
    state = orchestrator_node(state)
    
    # Check if expert consultation is needed
    if state["needs_expert"] and state["current_expert"]:
        # Run simulated expert conversation loop
        expert_context, user_satisfied = run_simulated_expert_conversation_loop(
            state["current_expert"], 
            state["client_message"]
        )
        
        # Update state after expert conversation
        state["conversation_context"] = expert_context
        state["user_satisfied"] = user_satisfied
        state["in_expert_conversation"] = False
        state["turn_count"] += 3  # Add 3 turns for the expert conversation
        
        # Return to Ruby for follow-up
        print("\n🔄 Returning to Ruby for follow-up...")
        
        # Generate a follow-up message from the client
        follow_up_messages = [
            HumanMessage(content=client_profile),
            HumanMessage(content=f"Just finished consulting with {state['current_expert']}. The consultation was helpful. Now I'm back with Ruby.")
        ]
        
        response = client_llm.invoke(follow_up_messages)
        follow_up_reply = response.content
        print_message("David Lim", follow_up_reply)
        
        # Update state with follow-up message
        state["client_message"] = follow_up_reply
        state["turn_count"] += 1
        
        # Process through orchestrator again
        state = orchestrator_node(state)
        state["turn_count"] += 1
        
        # Generate final response from client
        final_messages = [
            HumanMessage(content=client_profile),
            HumanMessage(content="Thanks Ruby, that's all I need for now. The consultation was very helpful.")
        ]
        
        response = client_llm.invoke(final_messages)
        final_reply = response.content
        print_message("David Lim", final_reply)
        
        # Final response from Ruby
        print("🎯 Ruby (Orchestrator & Concierge):")
        print("You're very welcome! I'm glad Dr. Warren could help you with your questions about pre-program testing. If you need anything else in the future, just reach out. Have a great day!")
    
    print("\n✅ Simulated conversation ended.")
    print(f"📊 Total turns: {state['turn_count']}")
    return state


# ----- USAGE EXAMPLE -----
if __name__ == "__main__":
    print("🚀 Elyx Team Conversation System\n")
    print("Choose mode:")
    print("1. Interactive (you type messages)")
    print("2. Simulated (AI client)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_interactive_conversation()
    else:
        run_simulated_conversation()
