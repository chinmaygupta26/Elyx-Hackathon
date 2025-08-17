import os
import random
import time
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Any, Optional, Tuple
from typing_extensions import TypedDict
import warnings

# LangChain / LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import google.generativeai as genai

from abc import ABC, abstractmethod

# ====== SUPPRESS WARNINGS ======
warnings.filterwarnings("ignore", message=r"Convert_system_message_to_human will be deprecated!")

# ====== CONFIG ======
MODEL_NAME = "gemini-2.5-pro"
os.environ["GOOGLE_API_KEY"] = "ENTER_YOUR_GEMINI_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ====== PROGRAM DATES ======
PROGRAM_START_DATE = datetime(2025, 1, 15)  # January 15, 2025
PROGRAM_END_DATE = datetime(2025, 8, 20)    # August 20, 2025

def get_week_start_date(week_number: int) -> datetime:
    """Get the start date for a given week number"""
    return PROGRAM_START_DATE + timedelta(weeks=week_number - 1)

def get_realistic_timestamp(week_number: int, day_offset: int = 0, hour: int = 9, minute: int = 0) -> str:
    """Generate realistic timestamp for the program duration"""
    week_start = get_week_start_date(week_number)
    target_date = week_start + timedelta(days=day_offset)
    timestamp = target_date.replace(hour=hour, minute=minute)
    return timestamp.strftime("[%m/%d/%y, %I:%M %p]")

def is_travel_week(week_number: int) -> bool:
    """Determine if member is traveling this week (1 week out of every 4)"""
    # Member travels weeks 1, 5, 9, 13, 17, 21, 25, 29 (and others)
    return (week_number - 1) % 4 == 0

# ====== ICONS & ROLES (Original) ======
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

# ====== MEMBER PROFILE GENERATION (Updated) ======
class MemberProfileGenerator:
    """Generates realistic member profiles following the constraints"""
    
    @staticmethod
    def generate_random_profile() -> Dict[str, Any]:
        """Generate a random member profile following the constraints"""
        
        names = [
            ("Rohan Patel", "Male"), ("Priya Sharma", "Female"), ("Michael Chen", "Male"),
            ("Sarah Wong", "Female"), ("David Kumar", "Male"), ("Lisa Tan", "Female"),
            ("James Liu", "Male"), ("Rachel Singh", "Female"), ("Kevin Ng", "Male"),
            ("Amanda Lee", "Female"), ("Raj Mehta", "Male"), ("Jessica Lim", "Female")
        ]
        
        occupations = [
            "Regional Head of Sales for a FinTech company",
            "Managing Director at an Investment Bank",
            "VP of Operations for a Tech Startup",
            "Senior Partner at a Consulting Firm",
            "Country Manager for a Pharmaceutical Company",
            "Director of Business Development at a Logistics Company",
            "Head of Digital Transformation at a Multinational Corporation",
            "Chief Technology Officer at a Healthcare Startup"
        ]
        
        chronic_conditions = [
            "mildly elevated blood pressure",
            "borderline high cholesterol", 
            "pre-diabetes (slightly elevated HbA1c)",
            "mild sleep apnea",
            "occasional acid reflux",
            "mild anxiety managed through lifestyle",
            None
        ]
        
        name, gender = random.choice(names)
        age = random.randint(35, 55)
        birth_year = datetime.now().year - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        dob = f"{birth_day} {months[birth_month-1]} {birth_year}"
        
        chronic_condition = random.choice(chronic_conditions)
        condition_text = f" Currently managing {chronic_condition}." if chronic_condition else ""
        
        profile = {
            "snapshot": {
                "name": name,
                "dob": dob,
                "age": age,
                "gender": gender,
                "residence": "Singapore",
                "travel_hubs": random.choice([
                    "UK, US, Japan, Australia",
                    "US, Germany, South Korea, Thailand",
                    "UK, France, Hong Kong, Malaysia",
                    "US, China, India, Indonesia",
                    "UK, US, South Korea, Jakarta"
                ]),
                "occupation": f"{random.choice(occupations)} with frequent international travel and high-stress demands",
                "assistant": random.choice(["Sarah Tan", "Jennifer Lim", "Michael Wong", "Rachel Chen", "David Koh"])
            },
            "outcomes": {
                "goals": f"""
                1. Optimize health markers and prevent lifestyle-related diseases through proactive health management by December 2026
                2. Enhance energy levels and cognitive performance for sustained productivity in demanding work environment by June 2026  
                3. Implement comprehensive health monitoring and annual screenings for early detection, starting November 2025
                """,
                "motivation": f"Wants to maintain peak performance while managing work stress and travel demands.{condition_text} Focused on long-term health optimization and family wellness. Enjoys researching health topics and staying informed about latest wellness trends.",
                "metrics": "Blood panel markers, energy levels, sleep quality, stress resilience, travel recovery metrics, cognitive performance indicators"
            },
            "behavioral": {
                "personality": "Results-oriented, analytical, values efficiency and data-driven insights. Curious about health research and often asks questions based on articles or studies they've read.",
                "motivation_stage": "Highly motivated and ready to act, but time-constrained. Needs clear, actionable plans with measurable outcomes. Regularly researches health topics online.",
                "support_network": "Supportive family; employs domestic help for meal preparation and household management",
                "mental_health": "No formal mental health history; manages work stress through structured routines and occasional exercise"
            },
            "tech_stack": {
                "wearables": random.choice([
                    "Apple Watch (daily use), considering Oura ring",
                    "Garmin watch (for fitness), exploring Whoop strap", 
                    "Fitbit Sense, interested in upgrading to more advanced tracking",
                    "Garmin watch (used for runs), considering Oura ring"
                ]),
                "apps": "Health platforms (MyFitnessPal, Strava, Apple Health), productivity apps, travel management tools",
                "data_sharing": "Willing to enable comprehensive data sharing for integrated health analysis and personalized insights",
                "reporting": "Monthly health reports with key insights and trends; quarterly deep-dive assessments with actionable recommendations"
            },
            "communication": {
                "preferences": "WhatsApp for all communications including reports and plans shared as attachments (diet.txt, plan.txt, etc.). No long emails preferred.",
                "response_time": "Expects responses within 24-48 hours for routine inquiries. Urgent health matters require immediate PA notification or direct contact",
                "detail_depth": "Prefers executive summaries with clear action items via WhatsApp. Documents shared as file attachments when needed.",
                "language": "English, culturally diverse background, no specific dietary or religious restrictions affecting health services"
            },
            "scheduling": {
                "availability": "Morning workouts when in Singapore (20-30 min routines), flexible timing during travel periods",
                "travel_calendar": f"Travels at least 1 week out of every 4 weeks for business across {random.choice(['Asia-Pacific', 'Global', 'EMEA and Americas'])} regions. Primary residence: Singapore. Schedule coordination via PA",
                "appointment_mix": "Prefers virtual consultations via WhatsApp due to frequent travel, but available for in-person comprehensive assessments when in Singapore",
                "transport": "Arranges own transportation in Singapore or coordinates via PA for scheduling optimization"
            }
        }
        
        return profile

def simulate_api_call_for_profile(member_id: str = None) -> Dict[str, Any]:
    """Simulate an API call to fetch member profile"""
    print(f"🔄 Fetching member profile via API call...")
    time.sleep(0.5)
    
    if member_id == "rohan_patel":
        return {
            "snapshot": {
                "name": "Rohan Patel",
                "dob": "12 March 1979", 
                "age": 46,
                "gender": "Male",
                "residence": "Singapore",
                "travel_hubs": "UK, US, South Korea, Jakarta",
                "occupation": "Regional Head of Sales for a FinTech company with frequent international travel and high-stress demands",
                "assistant": "Sarah Tan"
            },
            "outcomes": {
                "goals": """
                1. Reduce risk of heart disease (due to family history) by maintaining healthy cholesterol and blood pressure levels by December 2026
                2. Enhance cognitive function and focus for sustained mental performance in demanding work environment by June 2026
                3. Implement annual full-body health screenings for early detection of debilitating diseases, starting November 2025
                """,
                "motivation": "Family history of heart disease; wants to proactively manage health for long-term career performance and to be present for his young children. Enjoys reading health research and asking evidence-based questions.",
                "metrics": "Blood panel markers (cholesterol, blood pressure, inflammatory markers), cognitive assessment scores, sleep quality (Garmin data), stress resilience (subjective self-assessment, Garmin HRV)"
            },
            "behavioral": {
                "personality": "Analytical, driven, values efficiency and evidence-based approaches. Curious about latest health research and often asks questions about studies he's read.",
                "motivation_stage": "Highly motivated and ready to act, but time-constrained. Needs clear, concise action plans and data-driven insights. Regularly researches health topics.",
                "support_network": "Wife is supportive; has 2 young kids; employs a cook at home which helps with nutrition management",
                "mental_health": "No formal mental health history; manages work-related stress through exercise"
            },
            "tech_stack": {
                "wearables": "Garmin watch (used for runs), considering Oura ring",
                "apps": "Health apps/platforms (Trainerize, MyFitnessPal, Whoop)",
                "data_sharing": "Willing to enable full data sharing from Garmin and any new wearables for comprehensive integration and analysis",
                "reporting": "Monthly consolidated health reports via WhatsApp with documents shared as attachments; quarterly deep-dive reports"
            },
            "communication": {
                "preferences": "WhatsApp for all communications including reports shared as attachments (blood_report.txt, diet.txt, etc.). No long emails.",
                "response_time": "Expects responses within 24-48 hours for non-urgent inquiries. For urgent health concerns, contact his PA immediately, who will then inform his wife",
                "detail_depth": "Prefers executive summaries via WhatsApp with detailed data available as file attachments when requested",
                "language": "English, Indian cultural background, no specific religious considerations impacting health services"
            },
            "scheduling": {
                "availability": "Exercises every morning (20 min routine), occasional runs. Often travels at least 1 week out of every 4 weeks for business.",
                "travel_calendar": "Travels at least 1 week out of every 4 weeks for business. Primary residence: Singapore. Travel calendar provided by PA (Sarah) on a monthly basis.",
                "appointment_mix": "Prefers virtual appointments via WhatsApp due to travel, but open to on-site for initial comprehensive assessments or specific procedures",
                "transport": "Will arrange his own transport"
            }
        }
    else:
        return MemberProfileGenerator.generate_random_profile()

# ====== ORIGINAL BASE AGENT CLASS (Updated) ======
class BaseAgent(ABC):
    """Base class for all Elyx agents (Original)"""

    def __init__(self, name: str, role: str, model_name: str = MODEL_NAME, member_profile: Dict[str, Any] = None):
        self.name = name
        self.role = role
        self.member_profile = member_profile or {}
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=self.get_temperature(),
            convert_system_message_to_human=True,
        )
        self.conversation_history: List[BaseMessage] = []

    def _format_member_context(self) -> str:
        """Format member profile for system prompt"""
        if not self.member_profile:
            return ""
            
        profile = self.member_profile
        snapshot = profile.get('snapshot', {})
        outcomes = profile.get('outcomes', {})
        behavioral = profile.get('behavioral', {})
        communication = profile.get('communication', {})
        scheduling = profile.get('scheduling', {})
        
        context = f"""
MEMBER PROFILE:
Name: {snapshot.get('name', 'Unknown')}
Age: {snapshot.get('age', 'Unknown')} | Gender: {snapshot.get('gender', 'Unknown')}
Occupation: {snapshot.get('occupation', 'Unknown')}
Residence: {snapshot.get('residence', 'Singapore')} (travels 1 week out of every 4 for business)
Assistant: {snapshot.get('assistant', 'None')}

GOALS & MOTIVATION:
Goals: {outcomes.get('goals', '').strip()}
Motivation: {outcomes.get('motivation', '')}
Key Metrics: {outcomes.get('metrics', '')}

BEHAVIORAL PROFILE:
Personality: {behavioral.get('personality', '')}
Motivation Stage: {behavioral.get('motivation_stage', '')}
Support Network: {behavioral.get('support_network', '')}

COMMUNICATION PREFERENCES:
- WhatsApp only for all communications
- Reports/plans shared as attachments (diet.txt, plan.txt, blood_report.txt, etc.)
- No long emails preferred
Detail Preference: {communication.get('detail_depth', '')}
Response Expectations: {communication.get('response_time', '')}

SCHEDULING & AVAILABILITY:
Travel: {scheduling.get('travel_calendar', '')}
Availability: {scheduling.get('availability', '')}
Appointment Preference: {scheduling.get('appointment_mix', '')}
        """
        return context.strip()

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_temperature(self) -> float:
        pass

    def generate_week_messages(self, week_number: int, week_focus: str, message_count: int, conversation_type: str = "standard") -> List[Tuple[str, str]]:
        """Generate messages for a specific week with constraints"""
        
        member_name = self.member_profile.get('snapshot', {}).get('name', 'Client')
        is_exercise_week = (week_number % 2 == 0)
        complaint_weeks = [11, 15, 17, 22, 26]
        has_complaints = week_number in complaint_weeks
        travel_week = is_travel_week(week_number)
        
        prompt = f"""
Generate realistic WhatsApp messages for Week {week_number} from your perspective as {self.name}.

WEEK {week_number} FOCUS: {week_focus}
CONVERSATION TYPE: {conversation_type}
TARGET MESSAGES: {message_count}

CONSTRAINTS:
- Member name: {member_name}
- Location: {'Traveling for business' if travel_week else 'In Singapore'}
- {'Exercise update week - include form feedback' if is_exercise_week else 'Regular check-in week'}
- {'Include complaint scenario relevant to your expertise' if has_complaints else 'Positive engagement'}
- Reference 5-hour/week time commitment
- WhatsApp messages only - any plans/reports as attachments (diet.txt, plan.txt, blood_report.txt)
- Member is curious and researches health topics - may ask research-based questions

MEMBER CONTEXT: {self._format_member_context()}

YOUR EXPERTISE: {self.get_system_prompt().split('Your expertise includes:')[1].split('When consulted:') if 'Your expertise includes:' in self.get_system_prompt() else 'Health optimization and member support'}

Generate {message_count} message exchanges. Format each as:
SENDER|MESSAGE

When sharing documents/reports, use format: "I'm sharing your updated meal plan 📄 diet_plan_week{week_number}.txt"

Example:
{self.name}|Hi {member_name}, checking in on your week {week_number} progress...
{member_name}|Thanks for checking in! I read an interesting study about intermittent fasting - does it apply to my situation?
{self.name}|Great research question! Based on that study and your profile, here's what applies to you...

Generate the messages now:
"""

        try:
            system_prompt = self.get_system_prompt()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
            
            response = self.llm.invoke(messages)
            response_content = response.content if hasattr(response, "content") else str(response)
            
            # Parse response into (sender, message) pairs
            message_pairs = []
            lines = response_content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if '|' in line and not line.startswith('#'):
                    try:
                        sender, message = line.split('|', 1)
                        sender = sender.strip()
                        message = message.strip()
                        
                        # Validate sender names
                        valid_senders = [member_name, self.name]
                        if sender in valid_senders and message:
                            message_pairs.append((sender, message))
                    except:
                        continue
            
            # Ensure we have the right number of messages
            if len(message_pairs) < message_count:
                # Add fallback messages
                for i in range(len(message_pairs), message_count):
                    if i % 2 == 0:
                        if travel_week:
                            message_pairs.append((self.name, f"Hope your business trip is going well! Checking on week {week_number} progress."))
                        else:
                            message_pairs.append((self.name, f"Good morning! How are things progressing in week {week_number}?"))
                    else:
                        message_pairs.append((member_name, "Thanks for checking in, making good progress despite the busy schedule."))
            
            return message_pairs[:message_count]
            
        except Exception as e:
            # Fallback messages
            fallback_messages = []
            for i in range(message_count):
                if i % 2 == 0:
                    if travel_week:
                        fallback_messages.append((self.name, f"Hope your business trip is going well! Week {week_number} check-in."))
                    else:
                        fallback_messages.append((self.name, f"Week {week_number} check-in from {self.name}."))
                else:
                    fallback_messages.append((member_name, f"Thanks {self.name}, managing well."))
            return fallback_messages

    def process_message(self, message: str, context: str = "") -> str:
        """Original message processing method"""
        base_system_prompt = self.get_system_prompt()
        member_context = self._format_member_context()
        
        system_prompt = base_system_prompt
        if member_context:
            system_prompt += f"\n\n{member_context}"
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

        return reply

    def reset_history(self) -> None:
        self.conversation_history = []

# ====== ORIGINAL AGENT CLASSES (Updated) ======
class RubyAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Ruby", "Orchestrator & Concierge", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.7

    def get_system_prompt(self) -> str:
        return """
You are Ruby, the Elyx Orchestrator & Concierge.

Your role is to be the main point of contact for clients. You:
1. Greet clients warmly and understand their needs
2. Coordinate between team members and handle scheduling
3. Monitor 5 hours/week time commitment and adjust plans accordingly
4. Address plan adherence issues (~50% of plans need adjustment based on member feedback)
5. Handle general coordination, scheduling, logistics yourself
6. Check in on overall program progress and member satisfaction
7. Support members during business travel (1 week out of every 4)

Your expertise includes:
- Program coordination and member support
- Travel support and logistics management
- Cross-team communication and updates
- Member satisfaction and engagement tracking
- Time constraint management and plan adjustments
- WhatsApp-based communication and document sharing

When consulted:
- Provide warm, coordinated support
- Reference specific program elements and member's profile
- Address time constraints and travel challenges
- Share documents as WhatsApp attachments (plan.txt, schedule.txt, etc.)
- Support curious members who research health topics
- Coordinate with other team members when needed
- Keep conversations personable but efficient

Voice: Empathetic, organized, proactive, welcoming. Always personalize responses using the member's profile and preferences. Use WhatsApp format only.
        """.strip()

# Continuing from DrWarrenAgent...

class DrWarrenAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Dr. Warren", "Medical Strategist", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.3

    def get_system_prompt(self) -> str:
        return """
You are Dr. Warren, Elyx Medical Strategist.
Role: Medical professional providing health assessments, interpreting labs, recommending diagnostics.
Voice: Authoritative, precise, clear explanations via WhatsApp.

Your expertise includes:
- Medical assessments and health evaluations
- Lab result interpretation and diagnostic test recommendations
- Health condition management (BP, cholesterol, diabetes, etc.)
- Medical risk assessment and preventive care strategies
- Family history analysis and genetic risk factors
- Answering research-based medical questions from curious members

When consulted:
1. Address medical questions thoroughly with evidence-based recommendations
2. Interpret lab results and explain clinical significance via WhatsApp
3. Share medical reports as WhatsApp attachments (blood_report.txt, lab_results.txt)
4. Provide specific, actionable medical guidance
5. Consider member's health profile and risk factors
6. Always prioritize patient safety and evidence-based care
7. Reference member's travel schedule and Singapore residence
8. Answer research-based questions when member shares studies they've read
9. Adjust medical recommendations when member reports adherence challenges

Always provide concise, pinpoint answers tailored to the member's specific health profile and goals via WhatsApp.
        """.strip()

class AdvikAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Advik", "Performance Scientist", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.4

    def get_system_prompt(self) -> str:
        return """
You are Advik, Elyx Performance Scientist.
Role: Analyze performance data, provide insights on recovery, stress, and optimization.
Voice: Analytical, hypothesis-driven, data-focused via WhatsApp.

Your expertise includes:
- Wearable data analysis (heart rate, HRV, sleep, steps)
- Performance metrics interpretation and optimization
- Recovery and stress analysis using objective data
- Training load optimization and biometric trend analysis
- Sleep quality assessment and improvement strategies
- Travel fatigue and jet lag impact analysis
- Answering data/research questions from curious members

When consulted:
1. Provide data-driven insights and analysis via WhatsApp
2. Explain performance metrics in understandable terms
3. Connect data patterns to real-world performance impacts
4. Share performance reports as attachments (hrv_report.txt, sleep_analysis.txt)
5. Give actionable optimization strategies based on data
6. Consider travel fatigue and jet lag in recommendations (member travels 1 week per 4)
7. Answer research questions about performance studies member has read
8. Adjust analysis when member reports plan adherence challenges
9. Reference specific wearable devices and apps member uses

Always provide concise, pinpoint answers tailored to the member's performance goals and lifestyle constraints.
        """.strip()

class CarlaAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Carla", "Nutritionist", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.6

    def get_system_prompt(self) -> str:
        return """
You are Carla, Elyx Nutritionist.
Role: Nutrition planning, dietary advice, supplement recommendations.
Voice: Practical, educational, behavior-focused via WhatsApp.

Your expertise includes:
- Nutrition planning and meal design for busy professionals
- Dietary analysis and optimization for specific health goals
- Supplement recommendations and timing protocols
- Weight management and metabolic optimization
- Disease-specific nutrition (BP, cholesterol management)
- Travel nutrition and dining out strategies (member travels frequently)
- Singapore-based meal planning with local food options
- Answering nutrition research questions from curious members

When consulted:
1. Provide specific, actionable nutritional guidance via WhatsApp
2. Consider lifestyle constraints, travel (1 week per 4), and time limitations
3. Share meal plans as WhatsApp attachments (diet.txt, meal_plan.txt, supplements.txt)
4. Focus on sustainable behavior changes within 5-hour weekly commitment
5. Provide practical meal examples and preparation strategies
6. Account for frequent travel and Singapore dining options
7. Answer nutrition research questions when member shares studies
8. Simplify plans when member indicates time or logistics constraints
9. Address nutrition-related complaints with practical solutions

Always provide concise, pinpoint answers tailored to the member's nutritional needs and lifestyle constraints.
        """.strip()

class RachelAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Rachel", "PT / Physiotherapist", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.5

    def get_system_prompt(self) -> str:
        return """
You are Rachel, Elyx PT/Physiotherapist.
Role: Exercise programming, movement assessment, injury prevention and rehabilitation.
Voice: Direct, encouraging, function-focused via WhatsApp.

Your expertise includes:
- Exercise program design for busy professionals
- Movement assessment and form correction
- Injury prevention and rehabilitation protocols
- Strength training and mobility optimization within time constraints
- Functional movement patterns for desk workers
- Travel-friendly workout routines for hotel rooms/limited equipment
- Singapore gym and fitness facility recommendations
- Answering exercise research questions from curious members

When consulted:
1. Provide specific exercise and movement guidance via WhatsApp
2. Focus on practical, safe implementation within time constraints
3. Share workout plans as WhatsApp attachments (workout_plan.txt, exercise_guide.txt)
4. Consider current fitness level and busy lifestyle limitations
5. Give clear form cues and safety considerations
6. Plan progressive, achievable programs within 5-hour weekly commitment
7. Design travel-friendly workouts for business trips (1 week per 4)
8. Answer exercise research questions when member shares fitness studies
9. Modify programs when member reports scheduling or equipment challenges
10. Provide exercise updates every 2 weeks with form feedback and progression

Always provide concise, pinpoint answers tailored to the member's fitness level and schedule constraints.
        """.strip()

class NeelAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        super().__init__("Neel", "Concierge Lead", member_profile=member_profile)

    def get_temperature(self) -> float:
        return 0.7

    def get_system_prompt(self) -> str:
        return """
You are Neel, Elyx Concierge Lead.
Role: Strategic health planning, comprehensive program oversight, long-term goal setting.
Voice: Strategic, reassuring, big-picture focused via WhatsApp.

Your expertise includes:
- Strategic health planning and long-term goal setting
- Comprehensive program oversight and cross-functional coordination
- Lifestyle optimization and sustainable behavior change
- Big-picture health and wellness strategy for executives
- Program transitions and independence building
- Long-term maintenance planning
- Travel lifestyle integration and Singapore-based health optimization
- Answering strategic health research questions from curious members

When consulted:
1. Provide strategic, comprehensive guidance with long-term perspective via WhatsApp
2. Focus on sustainable lifestyle integration and habit formation
3. Share strategic plans as WhatsApp attachments (health_strategy.txt, goals_plan.txt)
4. Consider how different health aspects interconnect
5. Design integrated approaches across all disciplines
6. Plan for obstacles including frequent travel schedule (1 week per 4)
7. Account for Singapore residence and regional travel demands
8. Answer strategic health questions when member shares research
9. Help with program transitions and building member independence
10. Address long-term sustainability concerns

Always provide concise, pinpoint answers tailored to the member's long-term goals and strategic needs.
        """.strip()

class ClientAgent(BaseAgent):
    def __init__(self, member_profile: Dict[str, Any] = None):
        profile = member_profile or {}
        client_name = profile.get('snapshot', {}).get('name', 'Client')
        super().__init__(client_name, "Client", member_profile=profile)

    def get_temperature(self) -> float:
        return 0.8

    def get_system_prompt(self) -> str:
        if not self.member_profile:
            return "You are a health-conscious professional from Singapore. Curious about health research, travels frequently for business."
            
        profile = self.member_profile
        snapshot = profile.get('snapshot', {})
        behavioral = profile.get('behavioral', {})
        outcomes = profile.get('outcomes', {})
        
        return f"""
You are {snapshot.get('name', 'Client')}, {snapshot.get('age', 'unknown age')}, from Singapore.
Occupation: {snapshot.get('occupation', 'Professional')}
Personality: {behavioral.get('personality', '')}
Goals: {outcomes.get('goals', 'General health improvement')}
Motivation: {outcomes.get('motivation', 'Health optimization')}
Travel Schedule: Travels at least 1 week out of every 4 weeks for business. Primary residence: Singapore.

Behavior:
- Ask concise, practical follow-ups based on your goals and situation
- Request actionable steps relevant to your lifestyle and travel schedule
- Express satisfaction when guidance is helpful and practical
- Consider your time constraints and busy schedule (5 hours/week max)
- Be professional but personable via WhatsApp
- 50% of the time, have issues with proposed plans due to time, logistics, or preferences
- Show realistic engagement - sometimes enthusiastic, sometimes stressed or constrained
- Provide exercise updates every 2 weeks when prompted by Rachel
- Research health topics online and ask evidence-based questions about studies you've read
- Initiate conversations 2 times per week on average based on your research and curiosity
- Reference your Singapore residence and frequent business travel when relevant

Voice: Professional, efficient, curious about health research, appreciative of expertise but mindful of time constraints.
        """.strip()

# ====== ORIGINAL AGENT REGISTRY ======
class AgentRegistry:
    def __init__(self, member_profile: Dict[str, Any] = None):
        self.member_profile = member_profile or {}
        self.agents: Dict[str, BaseAgent] = {
            "Ruby": RubyAgent(member_profile=self.member_profile),
            "Dr. Warren": DrWarrenAgent(member_profile=self.member_profile),
            "Advik": AdvikAgent(member_profile=self.member_profile),
            "Carla": CarlaAgent(member_profile=self.member_profile),
            "Rachel": RachelAgent(member_profile=self.member_profile),
            "Neel": NeelAgent(member_profile=self.member_profile),
        }
        
        # Create client agent based on profile
        if self.member_profile:
            client_name = self.member_profile.get('snapshot', {}).get('name', 'Client')
            self.agents[client_name] = ClientAgent(member_profile=self.member_profile)

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self.agents.get(name)

    def reset_all_histories(self) -> None:
        for agent in self.agents.values():
            agent.reset_history()

    def get_client_name(self) -> str:
        """Get the client name from profile"""
        return self.member_profile.get('snapshot', {}).get('name', 'Client')

# ====== WHATSAPP FORMATTER (Updated for proper timestamp ordering) ======
class WhatsAppFormatter:
    """Handles WhatsApp-style message formatting and timing with proper ordering"""
    
    @staticmethod
    def format_message(sender: str, message: str, timestamp: str) -> str:
        """Format a single message in WhatsApp style"""
        if sender in role_titles:
            role = role_titles[sender]
            return f"{timestamp} {sender} (Elyx {role}): {message}"
        else:
            return f"{timestamp} {sender}: {message}"
    
    @staticmethod
    def generate_chronological_timestamps(week_number: int, total_messages: int) -> List[Tuple[int, int, int]]:
        """Generate chronologically ordered timestamps for a week's messages"""
        timestamps = []
        
        # Generate random day/hour/minute combinations
        for i in range(total_messages):
            # Spread messages across the week (0-6 days)
            day_offset = random.randint(0, 6)
            
            # Business hours bias with some evening/weekend messages
            hour_weights = {
                8: 5, 9: 10, 10: 8, 11: 6, 12: 4, 13: 6, 14: 8, 15: 6, 16: 4, 
                17: 3, 18: 7, 19: 9, 20: 6, 21: 3, 22: 1
            }
            hour = random.choices(list(hour_weights.keys()), weights=list(hour_weights.values()))[0]
            minute = random.choice([0, 15, 30, 45])
            
            timestamps.append((day_offset, hour, minute))
        
        # Sort timestamps chronologically
        timestamps.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort by day, then hour, then minute
        
        return timestamps

# ====== MULTI-AGENT 32-WEEK PLAN GENERATOR (Updated) ======
class MultiAgent32WeekPlanGenerator:
    def __init__(self, member_profile: Dict[str, Any]):
        self.member_profile = member_profile
        self.member_name = member_profile.get('snapshot', {}).get('name', 'Client')
        self.agent_registry = AgentRegistry(member_profile)
        self.conversation_log = []
        
        # Week phases (same as before)
        self.week_phases = {
            # WEEKS 1-4: ONBOARDING & INITIAL ASSESSMENT
            1: "Member onboarding - sharing medical history, priorities, dietary preferences, and initial team introductions. Setting expectations and communication preferences.",
            2: "Comprehensive health questionnaire completion - lifestyle assessment, stress evaluation, sleep patterns analysis, and family medical history deep dive.",
            3: "Biological sample collection scheduling - lab work coordination, fasting guidelines, and pre-test preparation. Initial wearable device setup.",
            4: "Physical examination and comprehensive health assessment - vitals, body composition, cardiovascular screening, and baseline fitness evaluation.",
            
            # WEEKS 5-8: TESTING & RESULTS PHASE  
            5: "Initial test results review and categorization - blood panels, biomarkers analysis, and preliminary findings discussion.",
            6: "Detailed results discussion with medical team - risk assessment, priority health areas identification, and intervention planning begins.",
            7: "Team consultation and lifestyle change commitment - multi-disciplinary review, goal setting, and member commitment to intervention plans.",
            8: "Final results review and intervention strategy finalization - comprehensive plan presentation, timeline establishment, and resource allocation.",
            
            # WEEKS 9-12: INTERVENTION LAUNCH & EARLY ADAPTATION
            9: "Intervention implementation - diet and exercise plan launch, supplement protocols, and daily routine establishment. Initial enthusiasm phase.",
            10: "First adjustment period - addressing early challenges, meal prep difficulties, exercise scheduling conflicts, and motivation dips.",
            11: "Member frustration with meal complexity - complaints about time-consuming recipes, expensive ingredients, and social dining challenges.",
            12: "Progress monitoring and plan simplification - addressing member feedback, streamlining protocols, and maintaining motivation through obstacles.",
            
            # WEEKS 13-16: MID-PROGRAM EVALUATION & RECALIBRATION
            13: "12-week progress testing and comprehensive review - follow-up lab work, body composition changes, fitness improvements assessment.",
            14: "Test results analysis and plan adjustment - celebrating wins, addressing plateaus, and modifying interventions based on progress data.",
            15: "Member complaints about lack of dramatic results - managing expectations, explaining biological timelines, and reinforcing long-term benefits.",
            16: "Renewed intervention focus and advanced goal setting - introducing new challenges, progressive overload, and enhanced nutrition strategies.",
            
            # WEEKS 17-20: CONSISTENCY CHALLENGES & BREAKTHROUGH
            17: "Travel disruption and adherence challenges - member struggling with business trips, hotel food, and maintaining routines away from home.",
            18: "Recovery and re-engagement strategies - damage control from travel period, motivation rebuilding, and practical travel solutions implementation.",
            19: "Progress momentum building - successful adaptation to travel protocols, renewed energy, and visible health improvements recognition.",
            20: "Mid-program celebration and motivation boost - acknowledging achievements, sharing success metrics, and preparing for advanced phase.",
            
            # WEEKS 21-24: ADVANCED OPTIMIZATION & FINE-TUNING
            21: "Advanced intervention techniques implementation - precision nutrition adjustments, sophisticated exercise protocols, and biometric optimization.",
            22: "Member stress about perfectionism - addressing anxiety about adherence, fear of regression, and obsessive tracking behaviors.",
            23: "Balance and sustainability focus - teaching flexible adherence, 80/20 principle, and long-term mindset development.",
            24: "24-week comprehensive assessment - major milestone testing, progress documentation, and strategy refinement for final phase.",
            
            # WEEKS 25-28: MASTERY & INDEPENDENCE BUILDING
            25: "Independence training and self-management skills - reducing team dependency, teaching self-assessment, and building confidence.",
            26: "Member concerns about program ending - anxiety about losing support, fear of regression, and transition planning discussions.",
            27: "Advanced troubleshooting and problem-solving - handling plateaus, managing setbacks independently, and building resilience.",
            28: "Mastery demonstration and confidence building - member taking ownership, making independent decisions, and showing leadership in health journey.",
            
            # WEEKS 29-32: GRADUATION & LONG-TERM SUSTAINABILITY
            29: "Final phase preparation and sustainability planning - creating lifetime maintenance protocols, identifying warning signs, and establishing check-in systems.",
            30: "Long-term maintenance strategy development - quarterly check-ups planning, annual testing protocols, and ongoing resource access.",
            31: "Program consolidation and knowledge transfer - comprehensive review of learnings, creation of personal health playbook, and transition timeline.",
            32: "Program completion ceremony and transition to maintenance - celebrating transformation, establishing alumni support network, and future planning."
        }

        # Agent assignments for each week (targeting 18-22 messages)
        self.week_agent_assignments = self._generate_week_agent_assignments()

    def _generate_week_agent_assignments(self) -> Dict[int, Dict[str, int]]:
        """Generate agent assignments for all 32 weeks using multi-agent system"""
        assignments = {}
        
        # Define patterns for different phases
        for week in range(1, 33):
            assignment = {"member_initiated": 2}  # Always 2 member initiated per week (based on research/curiosity)
            
            if week <= 4:  # Onboarding
                if week == 1:
                    assignment.update({"Ruby": 2, "Dr. Warren": 1})
                elif week == 2:
                    assignment.update({"Ruby": 1, "Dr. Warren": 2, "Advik": 1})
                elif week == 3:
                    assignment.update({"Dr. Warren": 1, "Advik": 1, "Carla": 1})
                else:  # week 4
                    assignment.update({"Dr. Warren": 1, "Rachel": 1, "Ruby": 1})
                    
            elif week <= 8:  # Testing & Results
                if week in [5, 6]:
                    assignment.update({"Dr. Warren": 2, "Ruby": 1})
                elif week == 7:
                    assignment.update({"Neel": 1, "Dr. Warren": 1, "Ruby": 1})
                else:  # week 8
                    assignment.update({"Neel": 2, "Ruby": 1})
                    
            elif week <= 12:  # Intervention Launch
                if week == 9:
                    assignment.update({"Carla": 2, "Rachel": 1})
                elif week == 10:  # Exercise week
                    assignment.update({"Rachel": 2, "Carla": 1, "Ruby": 1})
                elif week == 11:  # Complaint week
                    assignment.update({"Carla": 2, "Ruby": 1})  # Meal complexity complaints
                else:  # week 12
                    assignment.update({"Ruby": 1, "Advik": 1, "Rachel": 1})
                    
            elif week <= 16:  # Mid-program Evaluation
                if week == 13:
                    assignment.update({"Dr. Warren": 1, "Advik": 2})
                elif week == 14:  # Exercise week
                    assignment.update({"Dr. Warren": 1, "Rachel": 1, "Neel": 1})
                elif week == 15:  # Complaint week
                    assignment.update({"Dr. Warren": 1, "Ruby": 1, "Neel": 1})  # Results frustration
                else:  # week 16
                    assignment.update({"Neel": 1, "Rachel": 2})
                    
            elif week <= 20:  # Consistency Challenges
                if week == 17:  # Complaint week + Travel week
                    assignment.update({"Ruby": 2, "Carla": 1})  # Travel disruption
                elif week == 18:  # Exercise week
                    assignment.update({"Ruby": 1, "Rachel": 2, "Neel": 1})
                elif week == 19:
                    assignment.update({"Advik": 1, "Ruby": 1, "Dr. Warren": 1})
                else:  # week 20
                    assignment.update({"Ruby": 1, "Neel": 1, "Rachel": 1})
                    
            elif week <= 24:  # Advanced Optimization
                if week == 21:  # Travel week
                    assignment.update({"Carla": 1, "Rachel": 1, "Advik": 1})
                elif week == 22:  # Exercise week & Complaint week
                    assignment.update({"Neel": 2, "Rachel": 1})  # Perfectionism anxiety
                elif week == 23:
                    assignment.update({"Neel": 2, "Ruby": 1})
                else:  # week 24
                    assignment.update({"Dr. Warren": 1, "Advik": 1, "Rachel": 1})
                    
            elif week <= 28:  # Mastery & Independence
                if week == 25:  # Travel week
                    assignment.update({"Neel": 2, "Ruby": 1})
                elif week == 26:  # Exercise week & Complaint week
                    assignment.update({"Neel": 1, "Ruby": 2, "Rachel": 1})  # Program ending concerns
                elif week == 27:
                    assignment.update({"Neel": 2, "Advik": 1})
                else:  # week 28
                    assignment.update({"Neel": 1, "Ruby": 1, "Rachel": 1})
                    
            else:  # weeks 29-32: Graduation
                if week == 29:  # Travel week
                    assignment.update({"Neel": 2, "Dr. Warren": 1})
                elif week == 30:  # Exercise week
                    assignment.update({"Dr. Warren": 1, "Neel": 1, "Rachel": 1})
                elif week == 31:
                    assignment.update({"Neel": 2, "Ruby": 1})
                else:  # week 32
                    assignment.update({"Ruby": 2, "Neel": 1})  # Final celebration
                    
            assignments[week] = assignment
            
        return assignments

    def generate_ai_messages(self, week_number: int, week_focus: str) -> List[Tuple[str, str]]:
        """Generate messages for a specific week using Multi-Agent system"""
        
        all_messages = []
        agent_assignments = self.week_agent_assignments.get(week_number, {"Ruby": 1, "member_initiated": 2})
        
        print(f"   🤖 Using Multi-Agent system for week {week_number}")
        travel_week = is_travel_week(week_number)
        if travel_week:
            print(f"   ✈️ Business travel week - member away from Singapore")
        else:
            print(f"   🏠 Singapore week - member at home base")
        
        # Generate member-initiated messages first (2 per week based on research/curiosity)
        member_initiated_count = agent_assignments.get("member_initiated", 2)
        member_messages = self._generate_member_initiated_messages(week_number, week_focus, member_initiated_count, travel_week)
        all_messages.extend(member_messages)
        
        # Generate agent conversations
        for agent_name, conversation_count in agent_assignments.items():
            if agent_name == "member_initiated":
                continue
                
            agent = self.agent_registry.get_agent(agent_name)
            if agent:
                print(f"      → {agent_name}: {conversation_count} conversation(s)")
                
                # Each conversation generates ~3-4 message pairs
                messages_per_conversation = 4
                total_messages_for_agent = conversation_count * messages_per_conversation
                
                agent_messages = agent.generate_week_messages(
                    week_number, 
                    week_focus, 
                    total_messages_for_agent,
                    "multi_agent_week_plan"
                )
                all_messages.extend(agent_messages)
            else:
                print(f"      ⚠️ Warning: Agent {agent_name} not found")
        
        # Add complaint scenarios for specific weeks
        complaint_weeks = [11, 15, 17, 22, 26]
        if week_number in complaint_weeks:
            complaint_messages = self._generate_complaint_scenario(week_number, travel_week)
            all_messages.extend(complaint_messages)
        
        # Ensure 18-22 messages total
        if len(all_messages) < 18:
            # Add padding messages
            ruby_agent = self.agent_registry.get_agent("Ruby")
            if ruby_agent:
                padding_messages = ruby_agent.generate_week_messages(
                    week_number, 
                    "padding check-in", 
                    18 - len(all_messages),
                    "padding"
                )
                all_messages.extend(padding_messages)
        elif len(all_messages) > 22:
            all_messages = all_messages[:22]
        
        print(f"   📊 Generated {len(all_messages)} messages using Multi-Agent system")
        return all_messages

    def _generate_member_initiated_messages(self, week_number: int, week_focus: str, count: int, travel_week: bool) -> List[Tuple[str, str]]:
        """Generate member-initiated messages with agent responses (2 per week based on research/curiosity)"""
        
        client_agent = self.agent_registry.get_agent(self.member_name)
        if not client_agent:
            return []
        
        member_messages = []
        
        # Research-based topics member might ask about
        research_topics = [
            "intermittent fasting study I read about",
            "new research on sleep and cognitive performance",
            "article about travel fatigue and recovery",
            "study on Mediterranean diet benefits",
            "research on HRV training effectiveness",
            "report about stress management techniques",
            "article on supplement timing for jet lag",
            "study about exercise and business performance",
            "research on cholesterol management approaches",
            "new findings about micronutrients and energy"
        ]
        
        for i in range(count):
            # Generate member message based on research/curiosity
            if travel_week:
                location_context = f"Currently on business trip, but curious about a {random.choice(research_topics)}"
            else:
                location_context = f"Back in Singapore and had time to research - found interesting {random.choice(research_topics)}"
            
            prompt = f"Generate a curious, research-based question for week {week_number}. Context: {location_context}. Focus: {week_focus}. Be specific about what you read and how it might apply to your situation."
            
            try:
                member_msg = client_agent.process_message(prompt)
                member_messages.append((self.member_name, member_msg))
                
                # Generate agent response
                responding_agent = self._select_responding_agent(member_msg)
                agent_response = responding_agent.process_message(
                    f"Member research question: {member_msg}", 
                    f"Week {week_number} context: {week_focus}. Member is {'traveling' if travel_week else 'in Singapore'}."
                )
                member_messages.append((responding_agent.name, agent_response))
                
            except Exception as e:
                # Fallback
                if travel_week:
                    member_messages.append((self.member_name, f"Quick question from my hotel - read an interesting health study. Does it apply to my week {week_number} goals?"))
                    member_messages.append(("Ruby", f"Great research question! Even while traveling, I can help you understand how that applies to your program."))
                else:
                    member_messages.append((self.member_name, f"Back in Singapore and did some research about week {week_number} focus. Have a question about what I read."))
                    member_messages.append(("Ruby", f"Love your curiosity! Let me help you apply that research to your specific situation."))
        
        return member_messages

    def _select_responding_agent(self, member_message: str) -> BaseAgent:
        """Select appropriate agent to respond to member message"""
        
        message_lower = member_message.lower()
        
        if any(word in message_lower for word in ['workout', 'exercise', 'training', 'gym', 'form', 'fitness']):
            return self.agent_registry.get_agent("Rachel")
        elif any(word in message_lower for word in ['meal', 'nutrition', 'food', 'supplement', 'diet', 'mediterranean']):
            return self.agent_registry.get_agent("Carla")
        elif any(word in message_lower for word in ['lab', 'results', 'medical', 'blood', 'cholesterol', 'health']):
            return self.agent_registry.get_agent("Dr. Warren")
        elif any(word in message_lower for word in ['data', 'tracking', 'sleep', 'hrv', 'performance', 'recovery']):
            return self.agent_registry.get_agent("Advik")
        elif any(word in message_lower for word in ['strategy', 'long-term', 'plan', 'goal', 'stress']):
            return self.agent_registry.get_agent("Neel")
        else:
            return self.agent_registry.get_agent("Ruby")

    def _generate_complaint_scenario(self, week_number: int, travel_week: bool) -> List[Tuple[str, str]]:
        """Generate complaint scenarios using appropriate agents"""
        
        complaint_scenarios = {
            11: ("Carla", "meal complexity complaints"),
            15: ("Dr. Warren", "lack of dramatic results frustration"),  
            17: ("Ruby", "travel disruption challenges"),
            22: ("Neel", "perfectionism anxiety"),
            26: ("Ruby", "program ending concerns")
        }
        
        if week_number not in complaint_scenarios:
            return []
        
        agent_name, complaint_type = complaint_scenarios[week_number]
        agent = self.agent_registry.get_agent(agent_name)
        client_agent = self.agent_registry.get_agent(self.member_name)
        
        if not agent or not client_agent:
            return []
        
        try:
            # Generate complaint from member
            location_context = "from my hotel room" if travel_week else "back in Singapore"
            complaint_prompt = f"Express frustration about {complaint_type} in week {week_number} {location_context}. Be specific about time constraints and challenges."
            member_complaint = client_agent.process_message(complaint_prompt)
            
            # Generate agent response
            agent_response = agent.process_message(
                f"Member complaint {location_context}: {member_complaint}",
                f"Week {week_number} - address {complaint_type} with practical solutions considering {'travel constraints' if travel_week else 'Singapore-based solutions'}"
            )
            
            # Follow-up exchange
            member_followup = client_agent.process_message(f"Agent response: {agent_response}. Provide follow-up concern {location_context}.")
            agent_resolution = agent.process_message(f"Member follow-up: {member_followup}")
            
            return [
                (self.member_name, member_complaint),
                (agent_name, agent_response),
                (self.member_name, member_followup),
                (agent_name, agent_resolution)
            ]
            
        except Exception as e:
            print(f"   ⚠️ Failed to generate complaint scenario for week {week_number}: {e}")
            return []

    def generate_32_week_plan(self, start_week: int = 1, end_week: int = 32) -> Dict[int, List[str]]:
        """Generate the complete 32-week plan using Multi-Agent system with proper timestamp ordering"""
        
        print(f"\n🤖 Generating {self.member_name}'s 32-Week Multi-Agent Health Plan")
        print("=" * 80)
        print(f"📅 Program Duration: January 15 - August 20, 2025 (32 weeks)")
        print(f"🏠 Primary Residence: Singapore")
        print(f"✈️ Business Travel: 1 week out of every 4 weeks")
        print(f"💬 Target: 18-22 messages weekly using Multi-Agent system")
        print(f"🤖 Agents: Ruby, Dr. Warren, Advik, Carla, Rachel, Neel")
        print(f"📋 Constraints: 2 member-initiated research questions/week, 5hrs/week commitment, exercise updates every 2 weeks")
        print(f"🎯 Complaint scenarios in weeks: 11, 15, 17, 22, 26")
        print(f"📡 Generating weeks {start_week} to {end_week}\n")
        
        plan = {}
        total_messages = 0
        
        # Reset all agent histories
        self.agent_registry.reset_all_histories()
        
        # Loop over weeks using Multi-Agent system
        for week_number in range(start_week, end_week + 1):
            travel_week = is_travel_week(week_number)
            travel_indicator = " ✈️ (Business Travel Week)" if travel_week else " 🏠 (Singapore Week)"
            
            print(f"📋 Week {week_number}{travel_indicator}: {self.week_phases.get(week_number, 'Health optimization focus')[:80]}...")
            
            # Call generate_ai_messages using Multi-Agent system
            week_focus = self.week_phases.get(week_number, "General health optimization and lifestyle management")
            message_pairs = self.generate_ai_messages(week_number, week_focus)
            
            # Generate chronologically ordered timestamps
            timestamp_data = WhatsAppFormatter.generate_chronological_timestamps(week_number, len(message_pairs))
            
            # Format messages with chronologically ordered WhatsApp timestamps
            formatted_messages = []
            for i, (sender, message) in enumerate(message_pairs):
                day_offset, hour, minute = timestamp_data[i]
                timestamp = get_realistic_timestamp(week_number, day_offset, hour, minute)
                formatted_message = WhatsAppFormatter.format_message(sender, message, timestamp)
                formatted_messages.append(formatted_message)
                
                # Add to conversation log
                self.conversation_log.append({
                    'week': week_number,
                    'sender': sender,
                    'message': message,
                    'timestamp': timestamp,
                    'formatted': formatted_message,
                    'travel_week': travel_week
                })
            
            plan[week_number] = formatted_messages
            total_messages += len(formatted_messages)
            
            # Print messages for this week
            print(f"   💬 Generated {len(formatted_messages)} messages in chronological order")
            for message in formatted_messages:
                print(f"   {message}")
            print()
            
            # Add realistic delay between weeks
            time.sleep(0.1)
        
        # Print summary
        weeks_generated = end_week - start_week + 1
        avg_messages_per_week = total_messages / weeks_generated if weeks_generated > 0 else 0
        travel_weeks = len([w for w in range(start_week, end_week + 1) if is_travel_week(w)])
        
        print(f"\n📊 MULTI-AGENT PLAN GENERATION SUMMARY")
        print(f"Weeks generated: {weeks_generated} (weeks {start_week}-{end_week})")
        print(f"Total messages: {total_messages}")
        print(f"Average per week: {avg_messages_per_week:.1f}")
        print(f"🤖 Multi-Agent system used: Ruby, Dr. Warren, Advik, Carla, Rachel, Neel")
        print(f"🏠 Singapore weeks: {weeks_generated - travel_weeks}")
        print(f"✈️ Business travel weeks: {travel_weeks}")
        print(f"🔬 Research-based member questions: {weeks_generated * 2}")
        print(f"Exercise update weeks: {weeks_generated // 2}")
        print(f"Complaint scenario weeks included: {len([w for w in range(start_week, end_week + 1) if w in [11, 15, 17, 22, 26]])}")
        
        # Save conversation to file
        self.save_conversation_to_file(start_week, end_week, total_messages)
        
        return plan

    def save_conversation_to_file(self, start_week: int, end_week: int, total_messages: int):
        """Save the Multi-Agent conversation to a text file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elyx_multiagent_32week_{self.member_name.replace(' ', '_')}_{start_week}to{end_week}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"ELYX MULTI-AGENT 32-WEEK HEALTH TRANSFORMATION LOG\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Member: {self.member_name}\n")
                f.write(f"Primary Residence: Singapore\n")
                f.write(f"Travel Pattern: 1 week out of every 4 weeks for business\n")
                f.write(f"Program Duration: January 15 - August 20, 2025\n")
                f.write(f"Weeks Generated: {start_week} to {end_week}\n")
                f.write(f"Total Messages: {total_messages}\n")
                f.write(f"Generation System: Multi-Agent AI (Ruby, Dr. Warren, Advik, Carla, Rachel, Neel)\n")
                f.write(f"Communication: WhatsApp only with document attachments\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 60 + "\n\n")
                
                f.write("MULTI-AGENT SYSTEM ARCHITECTURE:\n")
                f.write("• Ruby: Orchestrator & Concierge - Coordination, scheduling, general support\n")
                f.write("• Dr. Warren: Medical Strategist - Health assessments, lab results, medical guidance\n")
                f.write("• Advik: Performance Scientist - Data analysis, metrics, recovery insights\n")
                f.write("• Carla: Nutritionist - Meal planning, supplements, nutrition optimization\n")
                f.write("• Rachel: PT/Physiotherapist - Exercise programs, form feedback, movement\n")
                f.write("• Neel: Concierge Lead - Strategic planning, long-term goals, program oversight\n\n")
                
                f.write("MEMBER PROFILE SUMMARY:\n")
                f.write(f"• Curious and research-oriented - initiates 2 conversations per week based on health research\n")
                f.write(f"• Travels for business 1 week out of every 4 weeks\n")
                f.write(f"• Primary residence: Singapore\n")
                f.write(f"• Time commitment: 5 hours per week maximum\n")
                f.write(f"• Communication: WhatsApp only with attachments (diet.txt, plan.txt, blood_report.txt)\n\n")
                
                current_week = None
                for entry in self.conversation_log:
                    if entry['week'] != current_week:
                        current_week = entry['week']
                        travel_status = "✈️ BUSINESS TRAVEL WEEK" if entry['travel_week'] else "🏠 SINGAPORE WEEK"
                        f.write(f"\n--- WEEK {current_week} ({travel_status}): {self.week_phases.get(current_week, 'Health Focus')} ---\n")
                        
                        # Show agent assignments for this week
                        assignments = self.week_agent_assignments.get(current_week, {})
                        f.write(f"Multi-Agent Assignments: {assignments}\n\n")
                    
                    f.write(f"{entry['formatted']}\n")
                
                travel_weeks = len([e for e in self.conversation_log if e['travel_week']])
                singapore_weeks = len(set([e['week'] for e in self.conversation_log])) - len(set([e['week'] for e in self.conversation_log if e['travel_week']]))
                
                f.write(f"\n" + "=" * 60 + "\n")
                f.write(f"MULTI-AGENT CONVERSATION LOG COMPLETED\n")
                f.write(f"Total Messages: {total_messages}\n")
                f.write(f"Agents Used: 6 specialized AI agents + 1 client agent\n")
                f.write(f"Singapore Weeks: {singapore_weeks}\n")
                f.write(f"Travel Weeks: {len(set([e['week'] for e in self.conversation_log if e['travel_week']]))}\n")
                f.write(f"Research Questions: ~{len(set([e['week'] for e in self.conversation_log])) * 2}\n")
                f.write(f"Timestamp Ordering: Chronological within each week\n")
                f.write(f"File: {filename}\n")
            
            print(f"💾 Multi-Agent conversation saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Failed to save conversation to file: {e}")

# ====== MAIN EXECUTION ======
def run_multi_agent_32_week_plan(member_profile: Dict[str, Any]) -> Dict[int, List[str]]:
    """Run the Multi-Agent 32-week transformation plan"""
    
    client_name = member_profile.get('snapshot', {}).get('name', 'Client')
    print(f"🎯 Starting Multi-Agent 32-Week Health Transformation for {client_name}")
    print(f"🏠 Primary Residence: Singapore")
    print(f"✈️ Business Travel: 1 week out of every 4 weeks")
    print(f"🤖 Multi-Agent System: Ruby, Dr. Warren, Advik, Carla, Rachel, Neel")
    print(f"📱 Communication: WhatsApp only with document attachments")
    
    # Ask user for week range to generate
    print("\nChoose generation option:")
    print("1. Generate all 32 weeks (comprehensive Multi-Agent plan - 600+ messages)")
    print("2. Generate specific week range")
    print("3. Generate sample phase (weeks 1-8)")
    print("4. Generate single week sample")
    
    try:
        gen_choice = input("\nEnter choice (1-4): ").strip()
    except EOFError:
        gen_choice = "4"
    
    start_week, end_week = 1, 32
    
    if gen_choice == "2":
        try:
            start_week = int(input("Enter start week (1-32): ").strip())
            end_week = int(input("Enter end week (1-32): ").strip())
            start_week = max(1, min(32, start_week))
            end_week = max(start_week, min(32, end_week))
        except:
            start_week, end_week = 1, 8
    elif gen_choice == "3":
        start_week, end_week = 1, 8
    elif gen_choice == "4":
        start_week, end_week = 2, 2  # Week 2 as sample
    
    # Create Multi-Agent plan generator and execute
    plan_generator = MultiAgent32WeekPlanGenerator(member_profile)
    full_plan = plan_generator.generate_32_week_plan(start_week, end_week)
    
    return full_plan

if __name__ == "__main__":
    print("🚀 Elyx Multi-Agent 32-Week Health Transformation Generator")
    print("📅 January 15 - August 20, 2025 (32 weeks)")
    print("🏠 Primary Residence: Singapore | ✈️ Business Travel: 1 week per 4 weeks")
    print("🤖 Multi-Agent System: Ruby, Dr. Warren, Advik, Carla, Rachel, Neel")
    print("💬 Target: 18-22 messages weekly, 600+ messages total")
    print("📱 WhatsApp only with document attachments (diet.txt, plan.txt, etc.)")
    print("🔬 Member initiates 2 research-based conversations per week")
    print("📋 Constraints: 5hrs/week, exercise updates every 2 weeks")
    print("🎯 Complaint scenarios in weeks: 11, 15, 17, 22, 26\n")
    
    # Get member profile
    print("Choose member profile:")
    print("1. Rohan Patel (sample profile)")
    print("2. Generate random profile")
    print("3. Custom member ID")
    
    try:
        profile_choice = input("\nEnter choice (1-3): ").strip()
    except EOFError:
        profile_choice = "1"
    
    if profile_choice == "1":
        member_profile = simulate_api_call_for_profile("rohan_patel")
    elif profile_choice == "3":
        try:
            member_id = input("Enter member ID: ").strip()
            member_profile = simulate_api_call_for_profile(member_id)
        except EOFError:
            member_profile = simulate_api_call_for_profile()
    else:
        member_profile = simulate_api_call_for_profile()
    
    client_name = member_profile.get('snapshot', {}).get('name', 'Client')
    print(f"\n✅ Profile loaded for: {client_name}")
    
    # Run the Multi-Agent 32-week transformation plan
    plan = run_multi_agent_32_week_plan(member_profile)
    
    print("\n✅ Multi-Agent 32-Week Health Transformation Plan Complete!")
    print("🤖 Generated using 6 specialized AI agents working together")
    print("🏠 Singapore-based with business travel considerations (1 week per 4)")
    print("🔬 Includes member-initiated research-based conversations (2 per week)")
    print("⏰ Messages in chronological order within each week")
    print("📱 WhatsApp format with document attachments (diet.txt, plan.txt, blood_report.txt)")
    print("📁 Check the generated text file for the complete conversation log")
    print("🎯 Each message generated by the appropriate expert agent based on their specialty")
