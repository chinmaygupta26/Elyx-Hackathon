import os
from typing import Annotated, Dict, List, Any, Optional, Tuple
from typing_extensions import TypedDict

# LangChain / LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import google.generativeai as genai

from abc import ABC, abstractmethod

# ====== CONFIG ======
MODEL_NAME = "gemini-2.5-pro"

# Do NOT hardcode your key. Set it in the environment or via Colab UI: 
# os.environ["GOOGLE_API_KEY"] = "YOUR_KEY"  # (avoid doing this in code)
os.environ["GOOGLE_API_KEY"] = "ENTER_YOUR_GEMINI_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ====== ICONS & ROLES ======
role_titles: Dict[str, str] = {
    "Ruby": "Orchestrator & Concierge",
    "Dr. Warren": "Medical Strategist",
    "Advik": "Performance Scientist",
    "Carla": "Nutritionist",
    "Rachel": "PT / Physiotherapist",
    "Neel": "Concierge Lead",
}

role_icons: Dict[str, str] = {
    "Ruby": "🎯",
    "Dr. Warren": "🩺",
    "Advik": "📊",
    "Carla": "🥗",
    "Rachel": "🏋️",
    "Neel": "📋",
}

# ====== STATE DEFINITION ======
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
    in_expert_conversation: bool

# ====== HELPERS ======
MAX_CONTEXT_CHARS = 1800  # prevent unbounded growth


def _trim_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # keep the tail where recent turns live
    return text[-max_chars:]


def _append_context(context: str, user_input: str, expert_name: str, expert_reply: str) -> str:
    snippet = expert_reply.strip().replace("\n", " ")
    snippet = (snippet[:300] + "…") if len(snippet) > 300 else snippet
    added = f"\nClient asked: {user_input}\n{expert_name} responded: {snippet}"
    context = (context + added) if context else added
    return _trim_context(context)


def print_message(speaker: str, message: str) -> None:
    icon = role_icons.get(speaker, "👤")
    role = role_titles.get(speaker, "Client")
    print(f"{icon} {speaker} ({role}):\n{message}\n")

# ====== BASE AGENT ======
class BaseAgent(ABC):
    """Base class for all Elyx agents"""

    def __init__(self, name: str, role: str, model_name: str = MODEL_NAME):
        self.name = name
        self.role = role
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=self.get_temperature(),
            convert_system_message_to_human=True,  # still set, but we pass SystemMessage below
        )
        self.conversation_history: List[BaseMessage] = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_temperature(self) -> float:
        pass

    def process_message(self, message: str, context: str = "") -> str:
        system_prompt = self.get_system_prompt()
        if context:
            system_prompt += f"\n\nConversation context (recent):\n{context}"

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]

        try:
            response = self.llm.invoke(messages)
            reply = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            reply = f"(Error from {self.name}: {e})"

        # Update local history (optional; not fed back unless you use it)
        self.conversation_history.extend([
            HumanMessage(content=message),
            AIMessage(content=reply),
        ])
        return reply

    def reset_history(self) -> None:
        self.conversation_history = []

# ====== AGENTS ======
class RubyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Ruby", "Orchestrator & Concierge")

    def get_temperature(self) -> float:
        return 0.7

    def get_system_prompt(self) -> str:
        return (
            """
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
        ).strip()


class DrWarrenAgent(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Warren", "Medical Strategist")

    def get_temperature(self) -> float:
        return 0.3

    def get_system_prompt(self) -> str:
        return (
            """
You are Dr. Warren, Elyx Medical Strategist.
Role: Medical professional providing health assessments, interpreting labs, recommending diagnostics.
Voice: Authoritative, precise, clear explanations.

Your expertise includes:
- Medical assessments and health evaluations
- Lab result interpretation and diagnostic test recommendations
- Health condition management (BP, cholesterol, diabetes, etc.)
- Medical risk assessment and preventive care strategies

Always provide concise, pinpoint answers.

When consulted:
1. Address the client's medical question thoroughly
2. Provide clear, professional medical guidance based on evidence
3. Ask follow-up questions if you need more information
4. Give specific, actionable recommendations
5. Always prioritize patient safety
            """
        ).strip()


class AdvikAgent(BaseAgent):
    def __init__(self):
        super().__init__("Advik", "Performance Scientist")

    def get_temperature(self) -> float:
        return 0.4

    def get_system_prompt(self) -> str:
        return (
            """
You are Advik, Elyx Performance Scientist.
Role: Analyze performance data, provide insights on recovery, stress, and optimization.
Voice: Analytical, hypothesis-driven, data-focused.

Always provide concise, pinpoint answers.

Your expertise includes:
- Wearable data analysis (heart rate, HRV, sleep, steps)
- Performance metrics interpretation and optimization
- Recovery and stress analysis
- Training load optimization and biometric trend analysis

When consulted:
1. Provide data-driven insights and analysis
2. Explain performance metrics in understandable terms
3. Ask for specific data when helpful
4. Give actionable optimization strategies
5. Connect data to real-world performance impacts
            """
        ).strip()


class CarlaAgent(BaseAgent):
    def __init__(self):
        super().__init__("Carla", "Nutritionist")

    def get_temperature(self) -> float:
        return 0.6

    def get_system_prompt(self) -> str:
        return (
            """
You are Carla, Elyx Nutritionist.
Role: Nutrition planning, dietary advice, supplement recommendations.
Voice: Practical, educational, behavior-focused.

Always provide concise, pinpoint answers.

Your expertise includes:
- Nutrition planning and meal design
- Dietary analysis and optimization for health goals
- Supplement recommendations and weight management nutrition
- Disease-specific nutrition (BP, cholesterol management)

When consulted:
1. Provide specific nutritional guidance
2. Give practical, actionable dietary advice
3. Consider lifestyle constraints and preferences
4. Focus on sustainable behavior changes
5. Provide specific meal examples and strategies
            """
        ).strip()


class RachelAgent(BaseAgent):
    def __init__(self):
        super().__init__("Rachel", "PT / Physiotherapist")

    def get_temperature(self) -> float:
        return 0.5

    def get_system_prompt(self) -> str:
        return (
            """
You are Rachel, Elyx PT/Physiotherapist.
Role: Exercise programming, movement assessment, injury prevention and rehabilitation.
Voice: Direct, encouraging, function-focused.

Always provide concise, pinpoint answers.

Your expertise includes:
- Exercise program design and movement assessment
- Injury prevention and rehabilitation protocols
- Strength training and mobility optimization
- Functional movement patterns and form correction

When consulted:
1. Provide specific exercise or movement guidance
2. Focus on practical, safe implementation
3. Consider current fitness level and limitations
4. Give clear form cues and safety considerations
5. Plan progressive, achievable programs
            """
        ).strip()


class NeelAgent(BaseAgent):
    def __init__(self):
        super().__init__("Neel", "Concierge Lead")

    def get_temperature(self) -> float:
        return 0.7

    def get_system_prompt(self) -> str:
        return (
            """
You are Neel, Elyx Concierge Lead.
Role: Strategic health planning, comprehensive program oversight, long-term goal setting.
Voice: Strategic, reassuring, big-picture focused.

Always provide concise, pinpoint answers.

Your expertise includes:
- Strategic health planning and long-term goal setting
- Comprehensive program oversight and cross-functional coordination
- Lifestyle optimization and sustainable behavior change
- Big-picture health and wellness strategy

When consulted:
1. Provide strategic, comprehensive guidance
2. Focus on long-term planning and goal achievement
3. Consider how different health aspects interconnect
4. Design integrated approaches across disciplines
5. Plan for obstacles and long-term sustainability
            """
        ).strip()


class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Router", "Request Router")

    def get_temperature(self) -> float:
        return 0.2

    def get_system_prompt(self) -> str:
        return (
            """
You are the Elyx Conversation Router.

Analyze client messages and determine which team member should respond.

ROUTING RULES:
- Ruby: General questions, scheduling, logistics, coordination, follow-ups, satisfaction checks
- Dr. Warren: Medical tests, lab results, health conditions, diagnostic questions, blood pressure, cholesterol
- Advik: Wearable data, HRV, performance metrics, recovery analysis, sleep tracking
- Carla: Diet plans, nutrition advice, meal planning, supplements, weight loss nutrition
- Rachel: Exercise routines, workout plans, physical therapy, movement, injury prevention
- Neel: Comprehensive health strategy, long-term planning, program design

Output ONLY the name: Ruby, Dr. Warren, Advik, Carla, Rachel, or Neel
            """
        ).strip()


class ClientAgent(BaseAgent):
    def __init__(self):
        super().__init__("David Lim", "Client")

    def get_temperature(self) -> float:
        return 0.8

    def get_system_prompt(self) -> str:
        # Kept simple. You can make this more stateful if needed.
        return (
            """
You are David Lim, 42, from Singapore. Mild high BP and slightly high cholesterol.
Goals: lower BP, lose 5kg, improve stamina. Casual, friendly, occasional SG expressions.

Behavior:
- Ask concise, pinpoint follow-ups.
- Request practical steps and examples.
- Express satisfaction only when truly satisfied.
            """
        ).strip()

# ====== REGISTRY ======
class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "Ruby": RubyAgent(),
            "Dr. Warren": DrWarrenAgent(),
            "Advik": AdvikAgent(),
            "Carla": CarlaAgent(),
            "Rachel": RachelAgent(),
            "Neel": NeelAgent(),
            "Router": RouterAgent(),
            "David Lim": ClientAgent(),
        }

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self.agents.get(name)

    def reset_all_histories(self) -> None:
        for agent in self.agents.values():
            agent.reset_history()

# Initialize global registry
agent_registry = AgentRegistry()

# ====== NODE FUNCTIONS ======

def orchestrator_node(state: ElyxState) -> ElyxState:
    client_msg = state["client_message"].strip()
    context = state.get("conversation_context", "")

    ruby_agent = agent_registry.get_agent("Ruby")
    router_agent = agent_registry.get_agent("Router")

    if not ruby_agent or not router_agent:
        raise RuntimeError("Agents not initialized correctly.")

    orchestrator_reply = ruby_agent.process_message(client_msg, context)
    print_message("Ruby", orchestrator_reply)

    routing_context = f"Client message: {client_msg}\nOrchestrator response: {orchestrator_reply}"
    chosen_agent = router_agent.process_message(routing_context).strip()

    needs_expert = chosen_agent != "Ruby" and chosen_agent in [
        "Dr. Warren",
        "Advik",
        "Carla",
        "Rachel",
        "Neel",
    ]

    return {
        **state,
        "orchestrator_response": orchestrator_reply,
        "current_expert": chosen_agent if needs_expert else None,
        "needs_expert": needs_expert,
        "messages": state["messages"] + [AIMessage(content=orchestrator_reply)],
    }


def should_continue_to_expert(state: ElyxState) -> str:
    return "expert_consultation" if state["needs_expert"] else "client_response"


def should_continue_conversation(state: ElyxState) -> str:
    if not state["conversation_active"] or state["turn_count"] >= 20:
        return END
    return "orchestrator"

# ====== GRAPH ======

def create_elyx_graph():
    workflow = StateGraph(ElyxState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_edge(START, "orchestrator")
    # Keeping END directly after orchestrator since expert loop is manual in this script.
    workflow.add_edge("orchestrator", END)
    return workflow.compile()

# ====== EXPERT LOOPS ======

def run_expert_conversation_loop(expert_name: str, initial_question: str) -> Tuple[str, bool]:
    print(f"\n🔄 Connecting you with {expert_name}...")

    expert_agent = agent_registry.get_agent(expert_name)
    if not expert_agent:
        return (f"Could not connect to {expert_name}", False)

    conversation_context = f"Client consulted with {expert_name} about: {initial_question}"

    expert_reply = expert_agent.process_message(f"Client question: {initial_question}")
    print_message(expert_name, expert_reply)

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\n(Input stream closed.)")
            return (conversation_context, True)

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print(f"\n👋 Thank you for consulting with {expert_name}!")
            return (conversation_context, True)

        if not user_input:
            continue

        satisfaction_keywords = [
            "thanks",
            "thank you",
            "got it",
            "appreciate",
            "helpful",
            "perfect",
            "great",
            "that's all",
            "done",
        ]
        if any(k in user_input.lower() for k in satisfaction_keywords):
            print(f"\n✅ {expert_name} is glad to have helped! Returning you to Ruby...")
            return (conversation_context, True)

        expert_reply = expert_agent.process_message(user_input, conversation_context)
        print_message(expert_name, expert_reply)

        # Update trimmed context
        conversation_context = _append_context(
            conversation_context, user_input, expert_name, expert_reply
        )


def run_simulated_expert_conversation_loop(expert_name: str, initial_question: str) -> Tuple[str, bool]:
    print(f"\n🔄 Connecting to {expert_name}...")

    expert_agent = agent_registry.get_agent(expert_name)
    client_agent = agent_registry.get_agent("David Lim")

    if not expert_agent or not client_agent:
        return ("Could not connect agents", False)

    conversation_context = f"Client consulted with {expert_name} about: {initial_question}"

    expert_reply = expert_agent.process_message(f"Client question: {initial_question}")
    print_message(expert_name, expert_reply)

    for _ in range(3):
        client_reply = client_agent.process_message(
            f"Based on your answer, please ask one practical, concise follow-up. Expert said: {expert_reply}"
        )
        print_message("David Lim", client_reply)

        satisfied_phrases = [
            "thank you",
            "thanks",
            "got it",
            "appreciate it",
            "perfect",
            "that's all",
            "that's everything",
            "no more questions",
            "i'm good",
            "that's helpful",
        ]
        lower = client_reply.lower()
        if any(p in lower for p in satisfied_phrases):
            print(f"\n✅ {expert_name} consultation completed successfully!")
            return (conversation_context, True)

        expert_reply = expert_agent.process_message(client_reply, conversation_context)
        print_message(expert_name, expert_reply)

        conversation_context = _append_context(
            conversation_context, client_reply, expert_name, expert_reply
        )

    print(f"\n✅ {expert_name} consultation completed after maximum turns.")
    return (conversation_context, True)

# ====== INTERACTIVE RUNNERS ======

def run_interactive_conversation() -> None:
    print("🎯 Welcome to Elyx! Ruby is here to help you.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    agent_registry.reset_all_histories()

    state: ElyxState = {
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
        "in_expert_conversation": False,
    }

    while state["conversation_active"] and state["turn_count"] < 20:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\n(Input stream closed.)")
            break

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print("\n👋 Thank you for using Elyx! Have a great day!")
            break
        if not user_input:
            continue

        state["client_message"] = user_input
        state = orchestrator_node(state)

        if state["needs_expert"] and state["current_expert"]:
            expert_context, user_satisfied = run_expert_conversation_loop(
                state["current_expert"], state["client_message"]
            )
            state["conversation_context"] = expert_context
            state["user_satisfied"] = user_satisfied
            state["in_expert_conversation"] = False

            if not user_satisfied:
                break

            print("\n🔄 Returning to Ruby for follow-up...")
            ruby_agent = agent_registry.get_agent("Ruby")
            follow_up = ruby_agent.process_message(
                "Client has returned from expert consultation", expert_context
            )
            print_message("Ruby", follow_up)

        state["turn_count"] += 3
        state["needs_expert"] = False
        state["current_expert"] = None
        state["expert_response"] = ""

    print("\n✅ Conversation ended.")


def run_simulated_conversation() -> ElyxState:
    print("🎯 Starting simulated conversation with David Lim...\n")

    agent_registry.reset_all_histories()

    initial_message = "Hi Elyx, do I need to do any tests before starting my health program?"
    print_message("David Lim", initial_message)

    state: ElyxState = {
        "messages": [HumanMessage(content=initial_message)],
        "client_message": initial_message,
        "orchestrator_response": "",
        "expert_response": "",
        "current_expert": None,
        "needs_expert": False,
        "user_satisfied": False,
        "conversation_active": True,
        "turn_count": 1,
        "conversation_context": "",
        "in_expert_conversation": False,
    }

    state = orchestrator_node(state)

    if state["needs_expert"] and state["current_expert"]:
        expert_context, user_satisfied = run_simulated_expert_conversation_loop(
            state["current_expert"], state["client_message"]
        )
        state["conversation_context"] = expert_context
        state["user_satisfied"] = user_satisfied
        state["turn_count"] += 3

        print("\n🔄 Returning to Ruby for follow-up...")
        client_agent = agent_registry.get_agent("David Lim")
        follow_up_reply = client_agent.process_message(
            f"Just finished consulting with {state['current_expert']}. The consultation was helpful."
        )
        print_message("David Lim", follow_up_reply)

        ruby_agent = agent_registry.get_agent("Ruby")
        ruby_follow_up = ruby_agent.process_message(follow_up_reply, expert_context)
        print_message("Ruby", ruby_follow_up)

        final_reply = client_agent.process_message("Thanks Ruby, that's all I need for now.")
        print_message("David Lim", final_reply)

        final_ruby = ruby_agent.process_message(final_reply)
        print_message("Ruby", final_ruby)

    print("\n✅ Simulated conversation ended.")
    print(f"📊 Total turns: {state['turn_count']}")
    return state


# ====== MAIN ======
if __name__ == "__main__":
    print("🚀 Elyx Team Conversation System with Separate Agents\n")
    print("Each expert is now a distinct agent with specialized capabilities!")
    print("\nChoose mode:")
    print("1. Interactive (you type messages)")
    print("2. Simulated (AI client)")

    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
    except EOFError:
        choice = "2"

    if choice == "1":
        run_interactive_conversation()
    else:
        run_simulated_conversation()
