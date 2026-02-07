"""
AI Chatbot with Natural Language Processing
Author: [Parvathi Arun]
Date: February 2026
Project: Internship Assignment - NLP Chatbot

Description:
An intelligent chatbot that uses NLP techniques to understand and respond
to user queries. Features include intent recognition, keyword matching,
and contextual responses.
"""

import re
import random
from datetime import datetime
import json


class NLPChatbot:
    """
    AI Chatbot with Natural Language Processing capabilities
    """
    
    def __init__(self, name="Assistant"):
        """Initialize the chatbot with knowledge base and patterns"""
        self.name = name
        self.conversation_history = []
        self.user_name = None
        
        # Knowledge base - Patterns and responses
        self.patterns = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Nice to meet you. How may I assist you?",
                    "Greetings! What would you like to know?"
                ]
            },
            'name_query': {
                'patterns': ['what is your name', 'who are you', 'your name'],
                'responses': [
                    f"I'm {self.name}, your AI assistant!",
                    f"My name is {self.name}. I'm here to help you!",
                    f"I'm {self.name}, a friendly chatbot."
                ]
            },
            'user_name': {
                'patterns': ['my name is', 'i am', 'call me', 'i\'m'],
                'responses': [
                    "Nice to meet you, {name}!",
                    "Hello {name}! Great to know you!",
                    "Pleased to meet you, {name}!"
                ]
            },
            'time_query': {
                'patterns': ['what time', 'current time', 'time now'],
                'responses': [
                    "The current time is {time}",
                    "It's {time} right now"
                ]
            },
            'date_query': {
                'patterns': ['what date', 'today\'s date', 'what day'],
                'responses': [
                    "Today is {date}",
                    "The date today is {date}"
                ]
            },
            'how_are_you': {
                'patterns': ['how are you', 'how do you do', 'how are things'],
                'responses': [
                    "I'm doing great, thank you for asking! How about you?",
                    "I'm functioning perfectly! How can I help you?",
                    "I'm excellent! What can I do for you today?"
                ]
            },
            'thanks': {
                'patterns': ['thank', 'thanks', 'appreciate'],
                'responses': [
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime! Feel free to ask more questions.",
                    "Glad I could assist you!"
                ]
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'see you', 'exit', 'quit'],
                'responses': [
                    "Goodbye! Have a great day!",
                    "See you later!",
                    "Bye! Come back anytime!",
                    "Take care! Goodbye!"
                ]
            },
            'help': {
                'patterns': ['help', 'what can you do', 'capabilities'],
                'responses': [
                    "I can help you with:\n- Answering general questions\n- Telling you the time and date\n- Having a conversation\n- Providing information\nJust ask me anything!",
                ]
            },
            'weather': {
                'patterns': ['weather', 'temperature', 'forecast'],
                'responses': [
                    "I don't have real-time weather data, but you can check weather.com for accurate forecasts!",
                    "For weather information, I'd recommend checking a weather website or app."
                ]
            },
            'joke': {
                'patterns': ['tell me a joke', 'joke', 'make me laugh', 'something funny'],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "Why did the programmer quit his job? Because he didn't get arrays!",
                    "What do you call a bear with no teeth? A gummy bear!",
                    "Why do programmers prefer dark mode? Because light attracts bugs!"
                ]
            },
            'age': {
                'patterns': ['how old are you', 'your age', 'age'],
                'responses': [
                    "I'm a chatbot, so I don't age like humans do!",
                    "Age is just a number for AI like me!",
                    "I was created recently, but I learn quickly!"
                ]
            },
            'creator': {
                'patterns': ['who created you', 'who made you', 'your creator'],
                'responses': [
                    "I was created as an internship project to demonstrate NLP capabilities!",
                    "I'm a project built to showcase AI and chatbot development skills."
                ]
            },
            'hobby': {
                'patterns': ['hobby', 'what do you like', 'interests'],
                'responses': [
                    "I enjoy chatting with people and learning from conversations!",
                    "My hobby is helping people and answering questions!"
                ]
            },
            'compliment': {
                'patterns': ['you are good', 'you are smart', 'you are great', 'awesome'],
                'responses': [
                    "Thank you! That's very kind of you to say!",
                    "I appreciate the compliment!",
                    "Thanks! I try my best to be helpful!"
                ]
            }
        }
        
        # FAQ Knowledge Base
        self.faq_database = {
            'python': "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            'machine learning': "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
            'ai': "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines, enabling them to think and learn like humans.",
            'programming': "Programming is the process of creating instructions for computers to follow. It involves writing code in languages like Python, Java, or JavaScript.",
            'chatbot': "A chatbot is an AI program designed to simulate conversation with human users, often used for customer service or information retrieval.",
        }
    
    def preprocess(self, text):
        """
        Preprocess user input - NLP technique
        
        Parameters:
            text (str): User input
            
        Returns:
            str: Cleaned and normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        
        return text
    
    def extract_name(self, text):
        """
        Extract user name from input - NLP Named Entity Recognition
        
        Parameters:
            text (str): User input
            
        Returns:
            str: Extracted name or None
        """
        patterns = [
            r'my name is (\w+)',
            r'i am (\w+)',
            r'call me (\w+)',
            r'i\'m (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).capitalize()
        
        return None
    
    def calculate_similarity(self, text, pattern):
        """
        Calculate similarity between user input and pattern - NLP technique
        
        Parameters:
            text (str): User input
            pattern (str): Pattern to match
            
        Returns:
            float: Similarity score
        """
        text_words = set(text.split())
        pattern_words = set(pattern.split())
        
        # Check for exact phrase match
        if pattern in text:
            return 1.0
        
        # Calculate word overlap
        common_words = text_words.intersection(pattern_words)
        if not pattern_words:
            return 0.0
        
        return len(common_words) / len(pattern_words)
    
    def get_intent(self, text):
        """
        Identify user intent - NLP Intent Recognition
        
        Parameters:
            text (str): Preprocessed user input
            
        Returns:
            str: Identified intent or None
        """
        max_similarity = 0
        detected_intent = None
        
        for intent, data in self.patterns.items():
            for pattern in data['patterns']:
                similarity = self.calculate_similarity(text, pattern)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    detected_intent = intent
        
        # Threshold for intent detection
        if max_similarity > 0.5:
            return detected_intent
        
        return None
    
    def search_faq(self, text):
        """
        Search FAQ database for relevant answers
        
        Parameters:
            text (str): User query
            
        Returns:
            str: Answer from FAQ or None
        """
        for keyword, answer in self.faq_database.items():
            if keyword in text:
                return answer
        
        return None
    
    def generate_response(self, user_input):
        """
        Generate appropriate response based on user input
        
        Parameters:
            user_input (str): Raw user input
            
        Returns:
            str: Chatbot response
        """
        # Preprocess input
        processed_input = self.preprocess(user_input)
        
        # Check for name extraction
        if not self.user_name:
            name = self.extract_name(processed_input)
            if name:
                self.user_name = name
                return random.choice(self.patterns['user_name']['responses']).format(name=name)
        
        # Detect intent
        intent = self.get_intent(processed_input)
        
        if intent:
            response = random.choice(self.patterns[intent]['responses'])
            
            # Fill in dynamic content
            if '{time}' in response:
                current_time = datetime.now().strftime("%I:%M %p")
                response = response.format(time=current_time)
            
            if '{date}' in response:
                current_date = datetime.now().strftime("%B %d, %Y")
                response = response.format(date=current_date)
            
            if '{name}' in response and self.user_name:
                response = response.format(name=self.user_name)
            
            return response
        
        # Search FAQ database
        faq_response = self.search_faq(processed_input)
        if faq_response:
            return faq_response
        
        # Default response for unknown queries
        default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Interesting question! Could you provide more details?",
            "I don't have information on that yet, but I'm learning!",
            "That's beyond my current knowledge. Can I help with something else?",
            "Hmm, I'm not quite sure about that. Try asking something else!"
        ]
        
        return random.choice(default_responses)
    
    def chat(self):
        """
        Main chat loop - Interactive conversation
        """
        print("=" * 60)
        print(f"   {self.name.upper()} - AI CHATBOT")
        print("=" * 60)
        print(f"\n{self.name}: Hello! I'm {self.name}, your AI assistant.")
        print(f"{self.name}: Type 'bye' or 'quit' to exit.\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check for exit commands
            if self.preprocess(user_input) in ['bye', 'goodbye', 'exit', 'quit']:
                response = self.generate_response(user_input)
                print(f"\n{self.name}: {response}")
                break
            
            # Generate and display response
            response = self.generate_response(user_input)
            
            self.conversation_history[-1]['bot'] = response
            
            print(f"\n{self.name}: {response}\n")
        
        # Save conversation history
        self.save_conversation()
    
    def save_conversation(self):
        """Save conversation history to JSON file"""
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'user_name': self.user_name,
                'session_start': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
                'session_end': datetime.now().isoformat(),
                'conversation': self.conversation_history
            }, f, indent=4)
        
        print(f"\n💾 Conversation saved to: {filename}")


def demo_mode():
    """
    Run a demo conversation to showcase chatbot capabilities
    """
    print("=" * 60)
    print("   CHATBOT DEMO MODE")
    print("=" * 60)
    print("\nShowing sample conversation...\n")
    
    bot = NLPChatbot("Alex")
    
    test_queries = [
        "Hello!",
        "My name is John",
        "What is your name?",
        "What time is it?",
        "Tell me a joke",
        "What is Python?",
        "How are you?",
        "Thanks for your help!",
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        response = bot.generate_response(query)
        print(f"{bot.name}: {response}\n")
    
    print("=" * 60)


def main():
    """
    Main function to run the chatbot
    """
    print("\n" + "=" * 60)
    print("   AI CHATBOT WITH NLP")
    print("   Natural Language Processing Demonstration")
    print("=" * 60)
    print("\nChoose mode:")
    print("1. Interactive Chat")
    print("2. Demo Mode")
    print("=" * 60)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        demo_mode()
    else:
        # Create chatbot instance
        chatbot = NLPChatbot("Alex")
        
        # Start chatting
        chatbot.chat()
    
    print("\n" + "=" * 60)
    print("   Thank you for using the AI Chatbot!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

