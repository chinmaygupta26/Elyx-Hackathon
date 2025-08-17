import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import google.generativeai as genai

# ====== CONFIG ======
MODEL_NAME = "gemini-2.5-pro"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDxJOx6DS-WUBhXiiskYos3HkTyqUEpkZw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class ConversationParser:
    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent parsing
            convert_system_message_to_human=True,
        )
    
    def parse_conversation_to_messages(self, conversation_text: str) -> List[Dict[str, Any]]:
        """Parse raw conversation text into structured messages using LLM"""
        
        system_prompt = """
You are an expert conversation parser. Your task is to parse health conversation text and extract structured message data.

INSTRUCTIONS:
1. Parse each message from the conversation text
2. Extract: timestamp, speaker name, role, and message content
3. Identify roles based on context and speaker patterns:
   - "Medical Strategist" for doctors (Dr. prefix or medical advice)
   - "Concierge" for coordinators/Ruby/concierge staff
   - "Coach" for fitness/wellness coaches  
   - "Nutritionist" for nutrition experts
   - "Member" for patients/clients
   - "PT/Physiotherapist" for physical therapists
   - "Performance Scientist" for data/metrics specialists

4. Return ONLY a valid JSON array with this exact structure:
[
  {
    "speaker": "speaker_name",
    "role": "role_name", 
    "message": "message_content",
    "timestamp": "ISO_timestamp_string",
    "message_index": 0
  }
]

5. Convert timestamps to ISO format (YYYY-MM-DDTHH:MM:SS)
6. Clean up message content (remove extra whitespace, formatting artifacts)
7. Assign sequential message_index starting from 0

CRITICAL: Return ONLY the JSON array, no other text or explanations.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Parse this conversation:\n\n{conversation_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Clean up response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            parsed_messages = json.loads(response_text)
            return parsed_messages
        except Exception as e:
            print(f"Error parsing messages: {e}")
            return []
    
    def extract_key_events(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key healthcare decisions and events using LLM"""
        
        system_prompt = """
You are a healthcare data analyst. Extract key healthcare events, decisions, and recommendations from conversation messages.

LOOK FOR:
- Medical test orders/recommendations
- Medication changes or prescriptions  
- Therapy recommendations
- Exercise plans or modifications
- Specialist referrals or consultations
- Appointment scheduling
- Treatment plan changes
- Important health decisions
- Care coordination activities

For each key event, extract:
- Date (from message timestamp)
- Decision/event description 
- Reason/context
- Related message index
- Event type (test, medication, therapy, exercise, consult, referral, appointment, plan)

Return ONLY a JSON array:
[
  {
    "date": "YYYY-MM-DD",
    "decision": "description_of_decision",
    "reason": "context_or_reason",
    "related_message_index": 0,
    "event_type": "test|medication|therapy|exercise|consult|referral|appointment|plan"
  }
]

CRITICAL: Return ONLY the JSON array, no other text.
"""
        
        messages_text = json.dumps(messages, indent=2)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract key events from these messages:\n\n{messages_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            events = json.loads(response_text)
            return events
        except Exception as e:
            print(f"Error extracting events: {e}")
            return []
    
    def extract_health_metrics(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract health metrics and progress data using LLM"""
        
        system_prompt = """
You are a health data extraction specialist. Analyze conversation messages to extract health metrics, symptoms, and progress data.

EXTRACT:
- Vital signs (blood pressure, heart rate, weight, temperature)
- Symptoms mentioned (dizziness, shortness of breath, pain, fatigue, etc.)
- Emotional state indicators (worried, anxious, happy, improved, frustrated)
- Medication adherence mentions
- Exercise/activity levels
- Sleep quality mentions
- Any measurable health data

Group by date and extract:
- Date (YYYY-MM-DD)
- Specific metrics with values where available
- Symptoms present (boolean indicators)
- Emotional state assessment
- Notable observations

Return ONLY a JSON array:
[
  {
    "date": "YYYY-MM-DD",
    "vital_signs": {
      "weight": 75.5,
      "bp_systolic": 140,
      "bp_diastolic": 90,
      "heart_rate": 85
    },
    "symptoms": {
      "dizziness": true,
      "shortness_of_breath": false,
      "fatigue": true,
      "chest_pain": false
    },
    "emotional_state": "Concerned|Positive|Neutral|Anxious",
    "notes": "Additional observations",
    "medication_adherence": "Good|Poor|Unknown",
    "activity_level": "Low|Moderate|High|Unknown"
  }
]

Use null for missing data. CRITICAL: Return ONLY the JSON array.
"""
        
        messages_text = json.dumps(messages, indent=2)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract health metrics from these messages:\n\n{messages_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            metrics = json.loads(response_text)
            return metrics
        except Exception as e:
            print(f"Error extracting health metrics: {e}")
            return []
    
    def extract_member_profile(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract member profile information using LLM"""
        
        system_prompt = """
You are a patient profile analyst. Extract member/patient information from conversation messages.

EXTRACT:
- Name (first name or full name mentioned)
- Age (if mentioned)
- Location (city, state, country if mentioned)
- Health conditions (diagnosed conditions, chronic issues)
- Health goals (what they want to achieve)
- Key demographics or lifestyle factors

Return ONLY a JSON object:
{
  "name": "member_name",
  "age": 40,
  "location": "City, State, Country",
  "conditions": ["condition1", "condition2"],
  "goals": ["goal1", "goal2"],
  "demographics": {
    "occupation": "if_mentioned",
    "lifestyle_factors": ["factor1", "factor2"]
  }
}

Use null for unknown fields. CRITICAL: Return ONLY the JSON object.
"""
        
        messages_text = json.dumps(messages, indent=2)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract member profile from these messages:\n\n{messages_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            profile = json.loads(response_text)
            return profile
        except Exception as e:
            print(f"Error extracting member profile: {e}")
            return {}

def parse_conversation_file(file_path: str, output_path: str = "member_journey.json"):
    """Main function to parse conversation file and generate structured JSON"""
    
    print("🚀 Starting LLM-powered conversation parsing...")
    
    # Read conversation file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            conversation_text = f.read()
        print(f"✅ Read conversation file: {file_path}")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Initialize parser
    parser = ConversationParser()
    
    # Parse messages
    print("🔄 Parsing messages...")
    parsed_messages = parser.parse_conversation_to_messages(conversation_text)
    print(f"✅ Parsed {len(parsed_messages)} messages")
    
    if not parsed_messages:
        print("❌ No messages could be parsed from the conversation")
        return
    
    # Group by date
    print("🔄 Grouping messages by date...")
    journey_dict = {}
    for msg in parsed_messages:
        date = msg['timestamp'].split("T")[0] if 'T' in msg['timestamp'] else msg['timestamp'][:10]
        if date not in journey_dict:
            journey_dict[date] = []
        journey_dict[date].append(msg)
    
    journey_list = [{"date": date, "messages": msgs} for date, msgs in journey_dict.items()]
    print(f"✅ Grouped into {len(journey_list)} date groups")
    
    # Extract key events
    print("🔄 Extracting key healthcare events...")
    events = parser.extract_key_events(parsed_messages)
    print(f"✅ Extracted {len(events)} key events")
    
    # Extract health metrics
    print("🔄 Extracting health metrics and progress...")
    progress = parser.extract_health_metrics(parsed_messages)
    print(f"✅ Extracted {len(progress)} progress entries")
    
    # Extract member profile
    print("🔄 Extracting member profile...")
    member = parser.extract_member_profile(parsed_messages)
    print(f"✅ Extracted member profile")
    
    # Calculate engagement metrics
    role_counts = {}
    for msg in parsed_messages:
        role = msg.get('role', 'Unknown')
        role_counts[role] = role_counts.get(role, 0) + 1
    
    metrics = {
        "total_messages": len(parsed_messages),
        "total_days": len(journey_list),
        "role_engagement": role_counts,
        "key_events_count": len(events),
        "progress_entries": len(progress)
    }
    
    # Combine everything
    structured_data = {
        "member": member,
        "journey": journey_list,
        "events": events,
        "metrics": metrics,
        "progress": progress,
        "parsing_metadata": {
            "parsed_at": datetime.now().isoformat(),
            "parser_version": "LLM-powered",
            "source_file": file_path
        }
    }
    
    # Save JSON
    print(f"🔄 Saving structured data to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved structured data to {output_path}")
        
        # Print summary
        print("\n📊 PARSING SUMMARY:")
        print(f"Messages parsed: {len(parsed_messages)}")
        print(f"Date range: {len(journey_list)} days")
        print(f"Key events: {len(events)}")
        print(f"Progress entries: {len(progress)}")
        print(f"Member: {member.get('name', 'Unknown')}")
        print(f"Conditions: {', '.join(member.get('conditions', []))}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")


# ====== SAMPLE USAGE ======
if __name__ == "__main__":
    print("🏥 LLM-Powered Health Conversation Parser")
    print("=" * 50)
    
    # Fixed input/output files (no user input)
    input_file = "conversation.txt"          # default input file
    output_file = "member_journey.json"      # default output file
    
    parse_conversation_file(input_file, output_file)
    
    print("\n🎉 Parsing complete! Check the output file for structured data.")
