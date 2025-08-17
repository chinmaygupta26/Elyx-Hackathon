MODEL_NAME = "models/gemini-2.5-pro"
import google.generativeai as genai
from datetime import datetime, timedelta
import random
import os
import time
import re

# Configure Gemini API
genai.configure(api_key="AIzaSyDxJOx6DS-WUBhXiiskYos3HkTyqUEpkZw")  # Uncomment and add your API key

# HIGH LEVEL 32-WEEK PLAN
THIRTY_TWO_WEEK_PLAN = """
WEEKS 1-4: ONBOARDING & INITIAL ASSESSMENT
- Member onboarding, medical history sharing, team introductions
- Comprehensive health questionnaire and lifestyle assessment
- Biological sample collection and physical examination
- Baseline establishment and expectation setting

WEEKS 5-8: TESTING & RESULTS PHASE  
- Test results review and categorization into: major issues, need followup, all okay
- Medical team discussions and intervention planning
- Team consultation and member commitment to lifestyle changes
- Strategy finalization and resource allocation

WEEKS 9-12: INTERVENTION LAUNCH & EARLY ADAPTATION
- Diet and exercise plan implementation
- Early challenges with meal complexity and time management
- Member frustration with complicated protocols
- Plan simplification and motivation maintenance

WEEKS 13-16: MID-PROGRAM EVALUATION & RECALIBRATION
- 12-week progress testing and comprehensive review
- Results analysis and plan adjustments
- Member complaints about slow progress and expectation management
- Renewed focus and advanced goal setting

WEEKS 17-20: CONSISTENCY CHALLENGES & BREAKTHROUGH
- Travel disruptions and adherence challenges
- Recovery strategies and practical solutions
- Progress momentum building and visible improvements
- Mid-program celebration and motivation boost

WEEKS 21-24: ADVANCED OPTIMIZATION & FINE-TUNING
- Advanced intervention techniques and precision adjustments
- Member stress about perfectionism and obsessive behaviors
- Balance and sustainability focus development
- 24-week comprehensive milestone assessment

WEEKS 25-28: MASTERY & INDEPENDENCE BUILDING
- Self-management skills and reduced team dependency
- Member anxiety about program ending
- Advanced troubleshooting and resilience building
- Confidence building and ownership demonstration

WEEKS 29-32: GRADUATION & LONG-TERM SUSTAINABILITY
- Sustainability planning and maintenance protocols
- Knowledge transfer and personal health playbook creation
- Program completion and transition to alumni support
- Future planning and celebration of transformation
"""

# Weekly intensity and focus areas for realistic progression
weekly_intensity = {
    "weeks_1_4": "Foundation building - high touch, frequent communication, lots of learning",
    "weeks_5_8": "Information overload - test results, multiple team members, decision making", 
    "weeks_9_12": "Reality check - initial enthusiasm meets practical challenges",
    "weeks_13_16": "Recalibration - addressing plateaus and managing expectations",
    "weeks_17_20": "Resilience building - overcoming obstacles and maintaining consistency",
    "weeks_21_24": "Optimization - fine-tuning and advanced strategies", 
    "weeks_25_28": "Independence - reducing dependency and building confidence",
    "weeks_29_32": "Transition - preparing for post-program sustainability"
}

# Complaint scenarios to weave into conversations
complaint_scenarios = {
    "week_11_meal_complexity": {
        "complaint": "These meal plans are taking 2+ hours daily to prep. As a busy executive, this isn't sustainable.",
        "team_response": "Simplification with meal delivery service integration and 15-minute meal options."
    },
    "week_15_slow_results": {
        "complaint": "I've been following everything perfectly for 3+ months but my weight is only down 8 pounds. Expected more dramatic results.",
        "team_response": "Focus on non-scale victories, biomarker improvements, and body composition changes explanation."
    },
    "week_17_travel_disruption": {
        "complaint": "Completely fell off plan during 10-day Asia trip. Feel like I've ruined all progress.",
        "team_response": "Perspective reset, damage assessment, and practical travel protocol development."
    },
    "week_22_perfectionism_stress": {
        "complaint": "I'm getting anxious if I miss a workout or eat something not on plan. This is becoming obsessive.",
        "team_response": "Mental health check-in, flexibility training, and sustainable mindset coaching."
    },
    "week_26_ending_anxiety": {
        "complaint": "Worried about maintaining progress without weekly check-ins. What if I regain everything?",
        "team_response": "Transition planning, confidence building, and long-term support system establishment."
    }
}

# ----- ENHANCED MEMBER PROFILE STRUCTURE -----
class MemberProfile:
    def __init__(self, profile_data):
        self.snapshot = profile_data.get("snapshot", {})
        self.outcomes = profile_data.get("outcomes", {})
        self.behavioral = profile_data.get("behavioral", {})
        self.tech_stack = profile_data.get("tech_stack", {})
        self.communication = profile_data.get("communication", {})
        self.scheduling = profile_data.get("scheduling", {})
    
    def get_profile_summary(self):
        """Generate a comprehensive profile summary for AI context"""
        return f"""
MEMBER PROFILE SUMMARY:
Name: {self.snapshot.get('name', 'Member')}
Age/Gender: {self.snapshot.get('age', 'N/A')}, {self.snapshot.get('gender', 'N/A')}
Location: {self.snapshot.get('residence', 'N/A')}
Travel: {self.snapshot.get('travel_hubs', 'N/A')}
Occupation: {self.snapshot.get('occupation', 'N/A')}

HEALTH GOALS:
{self.outcomes.get('goals', 'Not specified')}

MOTIVATION: {self.outcomes.get('motivation', 'Not specified')}

SUCCESS METRICS: {self.outcomes.get('metrics', 'Not specified')}

PERSONALITY: {self.behavioral.get('personality', 'Not specified')}
MOTIVATION LEVEL: {self.behavioral.get('motivation_stage', 'Not specified')}
SUPPORT NETWORK: {self.behavioral.get('support_network', 'Not specified')}

TECH PREFERENCES: {self.tech_stack.get('wearables', 'Not specified')}
REPORTING: {self.tech_stack.get('reporting', 'Not specified')}

COMMUNICATION: {self.communication.get('preferences', 'Not specified')}
DETAIL LEVEL: {self.communication.get('detail_depth', 'Not specified')}

AVAILABILITY: {self.scheduling.get('availability', 'Not specified')}
TRAVEL SCHEDULE: {self.scheduling.get('travel_calendar', 'Not specified')}
APPOINTMENT PREFERENCE: {self.scheduling.get('appointment_mix', 'Not specified')}
"""

class EnhancedMemberProfile(MemberProfile):
    def __init__(self, profile_data):
        super().__init__(profile_data)
        self.test_panel_frequency_weeks = 12
        self.avg_weekly_conversations = 5
        self.weekly_commitment_hours = 5
        self.exercise_update_frequency_weeks = 2
        self.travel_pattern = "travels 1 week every 4 weeks"
        self.primary_residence = "Singapore"
        self.plan_adherence_rate = 0.5
        self.chronic_condition = "managing high BP"

    def get_constraints_summary(self):
        return f"""
----- MEMBER CONSTRAINTS & CONDITIONS -----
- Diagnostic Test Panel: Every {self.test_panel_frequency_weeks} weeks (quarterly)
- Avg Weekly Conversations: {self.avg_weekly_conversations}
- Weekly Plan Commitment: {self.weekly_commitment_hours} hours
- Exercise Plan Update Frequency: every {self.exercise_update_frequency_weeks} weeks
- Travel Pattern: {self.travel_pattern}
- Primary Residence: {self.primary_residence}
- Plan Adherence Rate: {self.plan_adherence_rate*100}% (plans need adjustment 50% of the time)
- Chronic Condition: {self.chronic_condition}
- Health Status: Generally not sick, managing one chronic condition
"""

# ----- SAMPLE MEMBER DATA -----
rohan_profile_data = {
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
        "motivation": "Family history of heart disease; wants to proactively manage health for long-term career performance and to be present for his young children",
        "metrics": "Blood panel markers (cholesterol, blood pressure, inflammatory markers), cognitive assessment scores, sleep quality (Garmin data), stress resilience (subjective self-assessment, Garmin HRV)"
    },
    "behavioral": {
        "personality": "Analytical, driven, values efficiency and evidence-based approaches",
        "motivation_stage": "Highly motivated and ready to act, but time-constrained. Needs clear, concise action plans and data-driven insights",
        "support_network": "Wife is supportive; has 2 young kids; employs a cook at home which helps with nutrition management",
        "mental_health": "No formal mental health history; manages work-related stress through exercise"
    },
    "tech_stack": {
        "wearables": "Garmin watch (used for runs), considering Oura ring",
        "apps": "Health apps/platforms (Trainerize, MyFitnessPal, Whoop)",
        "data_sharing": "Willing to enable full data sharing from Garmin and any new wearables for comprehensive integration and analysis",
        "reporting": "Monthly consolidated health report focusing on key trends and actionable insights; quarterly deep-dive into specific health areas"
    },
    "communication": {
        "preferences": "WhatsApp for quick updates, detailed reports via WhatsApp documents, important updates and communication via PA (Sarah) for scheduling",
        "response_time": "Expects responses within 24-48 hours for non-urgent inquiries. For urgent health concerns, contact his PA immediately, who will then inform his wife",
        "detail_depth": "Prefers executive summaries with clear recommendations, but appreciates access to granular data upon request to understand the underlying evidence",
        "language": "English, Indian cultural background, no specific religious considerations impacting health services"
    },
    "scheduling": {
        "availability": "Exercises every morning (20 min routine), occasional runs. Often travels at least once every two weeks",
        "travel_calendar": "Travel calendar provided by PA (Sarah) on a monthly basis. Requires flexible scheduling and consideration for time-zone adjustments during frequent travel",
        "appointment_mix": "Prefers virtual appointments due to travel, but open to on-site for initial comprehensive assessments or specific procedures",
        "transport": "Will arrange his own transport"
    }
}

# Initialize enhanced member profile
enhanced_member_profile = EnhancedMemberProfile(rohan_profile_data)
enhanced_member_context = enhanced_member_profile.get_profile_summary() + "\n" + enhanced_member_profile.get_constraints_summary()

# ----- WHATSAPP MESSAGE FORMATTING -----
class WhatsAppFormatter:
    def __init__(self, start_date=None):
        if start_date:
            self.current_time = start_date
        else:
            self.current_time = datetime(2025, 1, 15, 9, 0)  # Start on Jan 15, 2025, 9:00 AM
    
    def format_timestamp(self, dt):
        """Format timestamp as [M/D/YY, H:MM AM/PM] - cross-platform compatible"""
        month = str(dt.month)
        day = str(dt.day)
        year = dt.strftime("%y")
        
        hour = dt.hour
        if hour == 0:
            hour_12 = 12
            am_pm = "AM"
        elif hour < 12:
            hour_12 = hour
            am_pm = "AM"
        elif hour == 12:
            hour_12 = 12
            am_pm = "PM"
        else:
            hour_12 = hour - 12
            am_pm = "PM"
        
        minute = dt.strftime("%M")
        
        return f"[{month}/{day}/{year}, {hour_12}:{minute} {am_pm}]"
    
    def format_message(self, sender, message, add_minutes=None):
        if add_minutes:
            self.current_time += timedelta(minutes=add_minutes)
        
        timestamp = self.format_timestamp(self.current_time)
        
        # Handle attachments
        if "attached:" in message:
            lines = message.split('\n')
            main_message = []
            attachments = []
            
            for line in lines:
                if "attached:" in line:
                    attachments.append(line.strip())
                else:
                    main_message.append(line)
            
            formatted_msg = f"{timestamp} {sender}: {' '.join(main_message).strip()}"
            
            if attachments:
                for attachment in attachments:
                    self.current_time += timedelta(seconds=30)
                    att_timestamp = self.format_timestamp(self.current_time)
                    formatted_msg += f"\n{att_timestamp} {sender}: {attachment}"
            
            return formatted_msg
        else:
            return f"{timestamp} {sender}: {message}"
    
    def add_time_gap(self, hours=None, days=None, minutes=None):
        """Add specific time gaps for realistic conversation spacing"""
        if days:
            self.current_time += timedelta(days=days)
        if hours:
            self.current_time += timedelta(hours=hours)
        if minutes:
            self.current_time += timedelta(minutes=minutes)
    
    def add_random_delay(self, min_minutes=1, max_minutes=120):
        """Add random delay between messages"""
        delay = random.randint(min_minutes, max_minutes)
        self.current_time += timedelta(minutes=delay)

# ----- ROLES & TITLES -----
role_titles = {
    "Ruby": "Elyx Concierge",
    "Dr. Warren": "Elyx Medical Strategist", 
    "Advik": "Elyx Performance Scientist",
    "Carla": "Elyx Nutritionist",
    "Rachel": "Elyx PT/Physiotherapist",
    "Neel": "Elyx Concierge Lead",
    enhanced_member_profile.snapshot.get('name', 'Member'): "Member"
}

# ----- AI-POWERED CONVERSATION GENERATOR -----
class AIConversationGenerator:
    def __init__(self, formatter, member_name, use_ai=True):
        self.formatter = formatter
        self.member_name = member_name
        self.conversation_log = []
        self.use_ai = use_ai
        
        # Initialize AI model if enabled
        if self.use_ai:
            try:
                self.model = genai.GenerativeModel(MODEL_NAME)
                print("✅ AI model initialized successfully")
            except Exception as e:
                print(f"❌ AI model initialization failed: {e}")
                print("📝 Falling back to template-based generation")
                self.use_ai = False
    
    def get_week_phase_intensity(self, week_number):
        """Get the intensity and focus for a given week"""
        if week_number <= 4:
            return weekly_intensity["weeks_1_4"]
        elif week_number <= 8:
            return weekly_intensity["weeks_5_8"]
        elif week_number <= 12:
            return weekly_intensity["weeks_9_12"]
        elif week_number <= 16:
            return weekly_intensity["weeks_13_16"]
        elif week_number <= 20:
            return weekly_intensity["weeks_17_20"]
        elif week_number <= 24:
            return weekly_intensity["weeks_21_24"]
        elif week_number <= 28:
            return weekly_intensity["weeks_25_28"]
        else:
            return weekly_intensity["weeks_29_32"]
    
    def get_complaint_for_week(self, week_number):
        """Get specific complaint scenario for certain weeks"""
        complaint_mapping = {
            11: complaint_scenarios["week_11_meal_complexity"],
            15: complaint_scenarios["week_15_slow_results"],
            17: complaint_scenarios["week_17_travel_disruption"],
            22: complaint_scenarios["week_22_perfectionism_stress"],
            26: complaint_scenarios["week_26_ending_anxiety"]
        }
        return complaint_mapping.get(week_number)

    def generate_week_prompt_with_constraints(self, week_number, week_description, member_context):
        """Generate AI prompt for a specific week with enhanced constraints"""
        
        # Determine if this is an exercise update week (every 2 weeks)
        is_exercise_update_week = week_number % 2 == 0
        exercise_context = "\n- THIS WEEK: Include exercise plan updates/modifications based on progress" if is_exercise_update_week else ""
        
        # Get week intensity and complaint scenarios
        week_intensity = self.get_week_phase_intensity(week_number)
        week_complaint = self.get_complaint_for_week(week_number)
        
        complaint_context = ""
        if week_complaint:
            complaint_context = f"""
SPECIFIC COMPLAINT SCENARIO FOR WEEK {week_number}:
Member Complaint: "{week_complaint['complaint']}"
Team Response Strategy: {week_complaint['team_response']}
*** IMPORTANT: Include this complaint scenario naturally in the conversation ***
"""
        
        return f"""
You are generating realistic WhatsApp conversations for Elyx Health, a premium concierge healthcare service.

CONTEXT:
{member_context}

32-WEEK PROGRAM OVERVIEW:
{THIRTY_TWO_WEEK_PLAN}

CURRENT WEEK: Week {week_number}
FOCUS: {week_description}

WEEK INTENSITY & CHARACTERISTICS: {week_intensity}
{complaint_context}

ENHANCED CONSTRAINTS FOR THIS WEEK:
- Target: Generate exactly 18-22 messages for this week
- Member initiates: 2-3 conversations (curious questions, research-based inquiries, progress updates)
- Member time commitment: References to ~5 hours/week spent on health activities
- Exercise updates: {"Every 2 weeks - include exercise plan modifications" if is_exercise_update_week else "Maintain current exercise routine"}
- Conversation flow: Mix of scheduled check-ins, member-initiated questions, and team responses
{exercise_context}

ELYX TEAM MEMBERS:
- Ruby (Elyx Concierge): Primary contact, scheduling, coordination
- Dr. Warren (Elyx Medical Strategist): Medical decisions, test interpretation  
- Advik (Elyx Performance Scientist): Data analysis, wearables, performance optimization
- Carla (Elyx Nutritionist): Diet, nutrition plans, meal guidance
- Rachel (Elyx PT/Physiotherapist): Exercise, physical therapy, movement
- Neel (Elyx Concierge Lead): Senior oversight, complex coordination

MEMBER BEHAVIOR PATTERNS TO INCLUDE:
- Shows curiosity about health topics (asks follow-up questions)
- Mentions time spent on health activities (~1 hour daily average)
- References research they've done or articles they've read
- Asks for clarifications on recommendations
- Reports on adherence and challenges with the 5-hour weekly commitment

MESSAGE TYPES TO INCLUDE:
1. Scheduled check-ins from Elyx team (6-8 messages)
2. Member-initiated questions/research discussions (2-3 messages)  
3. Team responses and follow-ups (8-10 messages)
4. Progress updates and data sharing (2-4 messages)
{"5. Exercise plan updates and modifications (2-3 messages)" if is_exercise_update_week else ""}

INSTRUCTIONS:
Generate 18-22 realistic WhatsApp messages for Week {week_number} that:
1. Follow the 32-week program structure and week intensity
2. Show member initiating 2-3 conversations based on curiosity/research
3. Include references to the member's 5-hour weekly time commitment
4. {"Include exercise plan updates and progress discussions" if is_exercise_update_week else "Focus on current intervention adherence"}
5. Incorporate the specific complaint scenario if provided above
6. Maintain professional but friendly tone with realistic conversation flow
7. Show progression appropriate to week {week_number} of 32

FORMAT YOUR RESPONSE AS:
Message 1: [SENDER NAME] - [MESSAGE CONTENT]
Message 2: [SENDER NAME] - [MESSAGE CONTENT]
...

Generate messages for Week {week_number} now:
"""

    def parse_ai_response(self, response_text):
        """Parse AI response into (sender, message) tuples"""
        messages = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for pattern: "Message X: [SENDER] - [MESSAGE]"
            if re.match(r'^Message \d+:', line):
                # Extract everything after "Message X: "
                content = re.sub(r'^Message \d+:\s*', '', line)
                
                # Split on first " - " to separate sender and message
                if ' - ' in content:
                    sender_part, message_part = content.split(' - ', 1)
                    
                    # Clean up sender name (remove brackets if present)
                    sender = sender_part.strip()
                    if sender.startswith('[') and sender.endswith(']'):
                        sender = sender[1:-1]
                    
                    # Clean up message
                    message = message_part.strip()
                    
                    # Skip empty messages
                    if sender and message:
                        messages.append((sender, message))
            
            # Alternative pattern: direct "[SENDER] - [MESSAGE]" format
            elif ' - ' in line and not line.startswith('Week ') and not line.startswith('-'):
                sender_part, message_part = line.split(' - ', 1)
                sender = sender_part.strip()
                message = message_part.strip()
                
                # Remove brackets from sender if present
                if sender.startswith('[') and sender.endswith(']'):
                    sender = sender[1:-1]
                
                if sender and message:
                    messages.append((sender, message))
        
        return messages
    
    def generate_ai_messages(self, week_number, week_description):
        """Generate messages using AI for a specific week with enhanced constraints"""
        if not self.use_ai:
            return self.generate_fallback_messages_with_constraints(week_number, week_description)
        
        try:
            prompt = self.generate_week_prompt_with_constraints(week_number, week_description, enhanced_member_context)
            response = self.model.generate_content(prompt)
            
            if not hasattr(response, 'text') or not response.text:
                print(f"⚠️ Empty AI response for Week {week_number}, using fallback")
                return self.generate_fallback_messages_with_constraints(week_number, week_description)
            
            # Parse AI response
            messages = self.parse_ai_response(response.text)
            
            if not messages:
                print(f"⚠️ No valid messages parsed for Week {week_number}, using fallback")
                return self.generate_fallback_messages_with_constraints(week_number, week_description)
            
            print(f"✅ Generated {len(messages)} AI messages for Week {week_number}")
            return messages
            
        except Exception as e:
            print(f"❌ AI generation failed for Week {week_number}: {e}")
            return self.generate_fallback_messages_with_constraints(week_number, week_description)
    
    def generate_fallback_messages_with_constraints(self, week_number, week_description):
        """Generate fallback messages with all constraints implemented for 32-week plan"""
        
        messages = []
        is_exercise_update_week = week_number % 2 == 0
        
        # Base messages with week-specific content
        if week_number <= 4:
            # Onboarding phase
            base_messages = [
                ("Ruby (Elyx Concierge)", f"Week {week_number}: {week_description}"),
                (self.member_name, "Thanks Ruby. I'm excited to start this comprehensive health journey."),
                ("Dr. Warren (Elyx Medical Strategist)", f"Welcome {self.member_name}! Looking forward to working together on your health goals.")
            ]
        elif week_number == 11:
            # Meal complexity complaint week
            base_messages = [
                ("Ruby (Elyx Concierge)", f"Week {week_number} check-in: How are the meal plans working out?"),
                (self.member_name, "Honestly, these meal plans are taking 2+ hours daily to prep. As a busy executive, this isn't sustainable."),
                ("Carla (Elyx Nutritionist)", "I understand completely. Let me simplify this with meal delivery service integration and 15-minute meal options."),
                ("Ruby (Elyx Concierge)", "We'll have a streamlined plan ready by tomorrow that fits your schedule better.")
            ]
        elif week_number == 15:
            # Slow results complaint week
            base_messages = [
                ("Dr. Warren (Elyx Medical Strategist)", f"Week {week_number}: Time for our progress review!"),
                (self.member_name, "I've been following everything perfectly for 3+ months but my weight is only down 8 pounds. Expected more dramatic results."),
                ("Dr. Warren (Elyx Medical Strategist)", "Let's focus on the non-scale victories - your blood pressure is down, energy is up, and body composition has improved significantly."),
                ("Advik (Elyx Performance Scientist)", "Your biomarker improvements are actually quite impressive for this timeframe. The body changes we can't see are often more important.")
            ]
        elif week_number == 17:
            # Travel disruption week
            base_messages = [
                (self.member_name, "Just back from 10-day Asia trip. Completely fell off plan - feel like I've ruined all progress."),
                ("Ruby (Elyx Concierge)", "Don't worry! Travel disruptions are normal. Let's assess and get you back on track."),
                ("Dr. Warren (Elyx Medical Strategist)", "One week doesn't undo months of progress. Your body is more resilient than you think."),
                ("Rachel (Elyx PT/Physiotherapist)", "Let's develop practical travel protocols for future trips so this doesn't happen again.")
            ]
        elif week_number == 22:
            # Perfectionism stress week
            base_messages = [
                (self.member_name, "I'm getting anxious if I miss a workout or eat something not on plan. This is becoming obsessive."),
                ("Dr. Warren (Elyx Medical Strategist)", "This is actually common and important to address. Health shouldn't create stress."),
                ("Ruby (Elyx Concierge)", "Let's schedule a mental health check-in and work on flexibility training."),
                ("Neel (Elyx Concierge Lead)", "The 80/20 principle applies here - sustainable progress over perfection.")
            ]
        elif week_number == 26:
            # Program ending anxiety week
            base_messages = [
                (self.member_name, "Worried about maintaining progress without weekly check-ins. What if I regain everything?"),
                ("Dr. Warren (Elyx Medical Strategist)", "Your concern is valid and shows how much you've invested in this journey."),
                ("Ruby (Elyx Concierge)", "We're planning a comprehensive transition with quarterly check-ins and ongoing support."),
                ("Neel (Elyx Concierge Lead)", "You've built incredible knowledge and habits - you're more prepared than you realize.")
            ]
        else:
            # Standard weekly messages
            base_messages = [
                ("Ruby (Elyx Concierge)", f"Week {week_number}: {week_description}"),
                (self.member_name, f"Thanks for the check-in. I'm maintaining my 5-hour weekly commitment and seeing good progress."),
                ("Dr. Warren (Elyx Medical Strategist)", "Excellent consistency! That's exactly what drives long-term results.")
            ]
        
        messages.extend(base_messages)
        
        # Add member-initiated curiosity messages (2 per week)
        member_questions = [
            f"I've been reading about heart health and wondering - how do our current interventions specifically target cardiovascular risk factors?",
            f"Quick question about my test results - I saw an article about inflammatory markers. Can you explain how mine compare to optimal ranges?",
            f"I'm curious about the connection between sleep and the biomarkers we're tracking. Are there any studies you'd recommend?",
            f"Been researching nutrition and found conflicting info about intermittent fasting. What's your take for my specific situation?",
            f"I spent about 6 hours this week on the health plan activities. Is this typical? Should I be doing more or less?"
        ]
        
        selected_questions = random.sample(member_questions, min(2, len(member_questions)))
        for question in selected_questions:
            messages.append((self.member_name, question))
        
        # Add exercise update messages if it's an exercise week
        if is_exercise_update_week:
            exercise_update_msgs = [
                ("Rachel (Elyx PT/Physiotherapist)", f"Week {week_number} exercise update: Based on your progress, I'm modifying your routine. Increasing intensity by 10%.\nattached: Updated_Exercise_Plan_Week_{week_number}.pdf"),
                (self.member_name, "Great! I've been feeling stronger. The current routine has been taking about 45 minutes, 4 times a week."),
                ("Rachel (Elyx PT/Physiotherapist)", "Perfect adherence! The new plan will maintain that time commitment but improve efficiency.")
            ]
            messages.extend(exercise_update_msgs)
        
        # Add filler messages to reach target of ~20 messages
        current_count = len(messages)
        target_messages = random.randint(18, 22)
        
        filler_senders = ["Ruby (Elyx Concierge)", "Dr. Warren (Elyx Medical Strategist)", "Advik (Elyx Performance Scientist)"]
        filler_messages = [
            "How are you feeling with the current interventions?",
            "Your data shows positive trends this week.",
            "Any questions about your current health plan?",
            "Remember to stay consistent with the daily activities.",
            "Great progress on your health journey!"
        ]
        
        while current_count < target_messages:
            sender = random.choice(filler_senders)
            message = random.choice(filler_messages)
            messages.append((sender, message))
            current_count += 1
        
        return messages[:target_messages]  # Ensure exact target

    def add_message(self, sender, message, delay_minutes=None):
        """Add a formatted message to conversation log"""
        if delay_minutes:
            self.formatter.add_random_delay(delay_minutes, delay_minutes + 30)
        else:
            self.formatter.add_random_delay(5, 120)
        
        formatted_message = self.formatter.format_message(sender, message)
        self.conversation_log.append(formatted_message)
        return formatted_message
    
    def generate_thirty_two_week_conversation(self):
        """Generate a realistic 32-week conversation with all constraints"""
        print("🚀 Generating 32-Week Elyx Health WhatsApp Conversation")
        print("📅 Duration: January 15 - August 20, 2025 (32 weeks)")
        print("💬 Target: ~20 messages weekly, 640+ messages total")
        print("📋 Constraints: 2 member-initiated/week, 5hrs/week commitment, exercise updates every 2 weeks")
        print("🎯 Includes: Realistic complaints, progression challenges, and breakthrough moments")
        print("=" * 80)
        
        message_count = 0
        weekly_message_counts = []
        
        # Define 32-week phases with realistic complaints
        week_phases = {
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
        
        self.formatter.current_time = datetime(2025, 1, 15, 9, 15)
        
        for week in range(1, 33):  # 32 weeks instead of 20
            print(f"\n📅 Week {week}: {week_phases.get(week, 'Program continuation')}")
            week_start_count = message_count
            
            # Generate messages for this week with constraints
            week_messages = self.generate_ai_messages(week, week_phases.get(week, 'Program continuation'))
            
            # Add messages with realistic timing
            for i, (sender, message) in enumerate(week_messages):
                delay = 15 if i == 0 else random.randint(10, 120)
                msg = self.add_message(sender, message, delay)
                print(msg)
                message_count += 1
            
            # Track weekly message count
            week_msg_count = message_count - week_start_count
            weekly_message_counts.append(week_msg_count)
            print(f"   📊 Week {week} messages: {week_msg_count}")
            
            # Add gap between weeks
            if week < 32:
                self.formatter.add_time_gap(days=random.randint(5, 7), hours=random.randint(1, 12))
        
        # Summary statistics
        avg_weekly_messages = sum(weekly_message_counts) / len(weekly_message_counts)
        print(f"\n✅ Generated {message_count} messages over 32 weeks")
        print(f"📊 Average messages per week: {avg_weekly_messages:.1f}")
        print(f"📈 Weekly message distribution: {min(weekly_message_counts)}-{max(weekly_message_counts)} messages")
        
        if self.conversation_log:
            # Extract the timestamp part (up to and including the first ']')
            first_timestamp = self.conversation_log[0].split(']')[0] + ']'
            last_timestamp = self.conversation_log[0].split(']')[0] + ']'
            print(f"📊 Conversation span: {first_timestamp} to {last_timestamp}")
        
        return self.conversation_log

def save_conversation_to_file(conversation_log, filename="elyx_whatsapp_conversation_32weeks.txt"):
    """Save the conversation to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ELYX HEALTH - 32-WEEK WHATSAPP CONVERSATION LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Member: {enhanced_member_profile.snapshot.get('name')}\n")
            f.write(f"Period: January 15 - August 20, 2025 (32 weeks)\n")
            f.write(f"Total Messages: {len(conversation_log)}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PROGRAM STRUCTURE:\n")
            f.write(THIRTY_TWO_WEEK_PLAN)
            f.write("\n" + "=" * 60 + "\n\n")
            
            for message in conversation_log:
                f.write(message + "\n\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF 32-WEEK CONVERSATION LOG\n")
        
        print(f"💾 Conversation saved to: {filename}")
        print(f"📁 File location: {os.path.abspath(filename)}")
        return filename
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

# ----- MAIN EXECUTION -----
if __name__ == "__main__":
    # Initialize formatter and generator
    formatter = WhatsAppFormatter()
    member_name = enhanced_member_profile.snapshot.get('name')
    
    # Create AI-powered generator (set use_ai=True if you have API key configured)
    generator = AIConversationGenerator(formatter, member_name, use_ai=True)  # Set to True to use AI
    
    # Generate the 32-week conversation
    conversation_log = generator.generate_thirty_two_week_conversation()
    
    # Save to file
    filename = save_conversation_to_file(conversation_log)
    
    print(f"\n📋 Summary:")
    print(f"   • Total messages: {len(conversation_log)}")
    print(f"   • Time span: 32 weeks (Jan 15 - August 20, 2025)")
    print(f"   • Participants: {len(role_titles)} (Rohan + 5 Elyx experts)")
    print(f"   • Program phases: 8 distinct phases with realistic challenges")
    print(f"   • Complaint scenarios: 5 key complaint moments integrated")
    print(f"   • File saved: {filename}")
    print(f"\n🔧 Note: Set use_ai=True in AIConversationGenerator to enable AI-powered message generation")
