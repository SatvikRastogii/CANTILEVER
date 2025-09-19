import pandas as pd
import requests
import time
import random
import re
from bs4 import BeautifulSoup
import os
from preprocessing import clean_text, remove_pii, is_quality_text, filter_supportive_response

class DatasetCreator:
    def __init__(self):
        self.conversations = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def create_synthetic_mental_health_data(self, num_pairs=100000):
        """Create a large synthetic dataset of mental health conversations."""
        print(f"Creating {num_pairs} synthetic mental health conversation pairs...")
        
        # Base templates for prompts and responses
        prompt_templates = [
            "I'm feeling {emotion} today",
            "I've been struggling with {issue}",
            "I don't know how to deal with {situation}",
            "I'm worried about {concern}",
            "I feel {feeling} about {topic}",
            "I need help with {problem}",
            "I'm having trouble with {difficulty}",
            "I feel overwhelmed by {stress}",
            "I'm concerned about {worry}",
            "I don't understand why I feel {emotion}",
            "I'm having a hard time with {challenge}",
            "I feel lost about {confusion}",
            "I'm stressed about {pressure}",
            "I feel alone in {isolation}",
            "I'm confused about {uncertainty}",
            "I need advice about {guidance}",
            "I'm feeling {emotion} and don't know what to do",
            "I've been thinking about {thought}",
            "I'm scared about {fear}",
            "I feel like I'm {self_perception}"
        ]
        
        response_templates = [
            "I understand that {emotion} can be really difficult. You're not alone in feeling this way.",
            "It's completely normal to feel {emotion} about {situation}. Many people experience this.",
            "Thank you for sharing that with me. It takes courage to talk about {topic}.",
            "I hear you, and I want you to know that your feelings are valid. {support}",
            "It sounds like you're going through a challenging time. Remember to be kind to yourself.",
            "What you're experiencing is understandable. {encouragement}",
            "I appreciate you opening up about this. {validation}",
            "It's okay to feel {emotion}. These feelings are temporary and will pass.",
            "You're showing great strength by acknowledging your feelings. {support}",
            "Many people struggle with {issue}. You're not alone in this journey.",
            "It's important to remember that seeking help shows wisdom, not weakness.",
            "Your feelings are completely valid. {reassurance}",
            "I'm here to listen and support you through this difficult time.",
            "It's okay to not have all the answers right now. Take things one step at a time.",
            "You're doing the best you can, and that's enough. {encouragement}",
            "Remember that it's okay to take breaks and prioritize your mental health.",
            "What you're feeling is a normal human response to {situation}.",
            "I believe in your ability to work through this. {support}",
            "It's brave of you to share your feelings. {validation}",
            "You deserve compassion and understanding, especially from yourself."
        ]
        
        # Emotional and situational fillers
        emotions = [
            "anxious", "sad", "overwhelmed", "confused", "scared", "lonely", 
            "frustrated", "angry", "hopeless", "tired", "stressed", "worried",
            "nervous", "depressed", "lost", "empty", "hurt", "disappointed",
            "fearful", "uncertain", "vulnerable", "exhausted", "helpless"
        ]
        
        issues = [
            "anxiety", "depression", "stress", "relationships", "work", "family",
            "sleep", "motivation", "self-esteem", "social situations", "health",
            "finances", "future", "past trauma", "grief", "loneliness",
            "perfectionism", "procrastination", "decision making", "change"
        ]
        
        situations = [
            "this situation", "what's happening", "my life right now", "everything",
            "my relationships", "my work", "my health", "my future", "my past",
            "social interactions", "making decisions", "coping with change",
            "managing stress", "dealing with loss", "handling criticism"
        ]
        
        concerns = [
            "my mental health", "my relationships", "my future", "my job",
            "my family", "my health", "my finances", "my decisions",
            "my past mistakes", "what others think", "not being good enough",
            "failing", "being alone", "not fitting in", "making the wrong choice"
        ]
        
        feelings = [
            "confused", "lost", "overwhelmed", "scared", "hopeless", "angry",
            "sad", "anxious", "frustrated", "disappointed", "hurt", "lonely",
            "inadequate", "stressed", "worried", "uncertain", "vulnerable"
        ]
        
        topics = [
            "my life", "my relationships", "my work", "my health", "my future",
            "my past", "my decisions", "my feelings", "my problems", "my goals"
        ]
        
        problems = [
            "anxiety", "depression", "stress management", "relationships",
            "self-care", "motivation", "sleep", "social anxiety", "self-esteem",
            "decision making", "coping with change", "managing emotions"
        ]
        
        difficulties = [
            "sleeping", "concentrating", "making decisions", "socializing",
            "managing stress", "staying motivated", "coping with anxiety",
            "dealing with depression", "maintaining relationships", "self-care"
        ]
        
        stresses = [
            "work", "relationships", "finances", "health", "family",
            "social situations", "decision making", "change", "expectations",
            "responsibilities", "uncertainty", "conflict"
        ]
        
        worries = [
            "my future", "my health", "my relationships", "my job", "my family",
            "making mistakes", "not being good enough", "being alone",
            "what others think", "failing", "not fitting in"
        ]
        
        isolations = [
            "this", "my feelings", "my struggles", "my problems", "my pain",
            "my thoughts", "my experiences", "my situation", "my challenges"
        ]
        
        uncertainties = [
            "my future", "my relationships", "my career", "my health",
            "my decisions", "my feelings", "what I want", "who I am",
            "what to do", "how to cope", "where I'm going"
        ]
        
        guidances = [
            "relationships", "career decisions", "mental health", "self-care",
            "managing stress", "coping with anxiety", "building confidence",
            "improving sleep", "social skills", "life changes"
        ]
        
        thoughts = [
            "my future", "my past", "my relationships", "my health", "my work",
            "my problems", "my feelings", "my decisions", "my goals", "my fears"
        ]
        
        fears = [
            "the future", "failure", "being alone", "not being good enough",
            "making mistakes", "change", "rejection", "uncertainty", "conflict"
        ]
        
        self_perceptions = [
            "not good enough", "a failure", "broken", "different", "alone",
            "overwhelmed", "lost", "stuck", "hopeless", "worthless", "weak"
        ]
        
        support_phrases = [
            "You're stronger than you know.",
            "This feeling is temporary.",
            "You're doing better than you think.",
            "It's okay to ask for help.",
            "You deserve support and understanding.",
            "You're not alone in this.",
            "Your feelings are valid.",
            "You're taking important steps by talking about this.",
            "You have the strength to get through this.",
            "It's okay to take things one day at a time."
        ]
        
        encouragements = [
            "You're making progress, even if it doesn't feel like it.",
            "Every small step counts.",
            "You're learning and growing through this experience.",
            "You have more resilience than you realize.",
            "It's okay to have difficult days.",
            "You're doing the best you can with what you have.",
            "Your efforts matter.",
            "You're not giving up, and that's admirable.",
            "You're taking care of yourself by seeking support.",
            "You're stronger than your struggles."
        ]
        
        validations = [
            "Your feelings are completely understandable.",
            "It's normal to feel this way given your situation.",
            "You have every right to feel {emotion}.",
            "Your experience is valid and important.",
            "It's okay to feel overwhelmed sometimes.",
            "Your feelings are a normal response to stress.",
            "You're not overreacting.",
            "Your concerns are legitimate.",
            "It's understandable that you feel this way.",
            "Your feelings matter."
        ]
        
        reassurances = [
            "You're not alone in this.",
            "Many people have felt this way before.",
            "This is a common human experience.",
            "You're not broken or damaged.",
            "It's okay to not be okay sometimes.",
            "You're doing better than you think.",
            "This feeling will pass.",
            "You have the strength to handle this.",
            "You're not a burden to others.",
            "You deserve compassion and care."
        ]
        
        # Generate conversations
        for i in range(num_pairs):
            if i % 10000 == 0:
                print(f"Generated {i} pairs...")
            
            # Select random template and fillers
            prompt_template = random.choice(prompt_templates)
            response_template = random.choice(response_templates)
            
            # Fill in the templates
            prompt = prompt_template.format(
                emotion=random.choice(emotions),
                issue=random.choice(issues),
                situation=random.choice(situations),
                concern=random.choice(concerns),
                feeling=random.choice(feelings),
                topic=random.choice(topics),
                problem=random.choice(problems),
                difficulty=random.choice(difficulties),
                stress=random.choice(stresses),
                worry=random.choice(worries),
                isolation=random.choice(isolations),
                uncertainty=random.choice(uncertainties),
                guidance=random.choice(guidances),
                thought=random.choice(thoughts),
                fear=random.choice(fears),
                self_perception=random.choice(self_perceptions)
            )
            
            response = response_template.format(
                emotion=random.choice(emotions),
                situation=random.choice(situations),
                topic=random.choice(topics),
                issue=random.choice(issues),
                support=random.choice(support_phrases),
                encouragement=random.choice(encouragements),
                validation=random.choice(validations),
                reassurance=random.choice(reassurances)
            )
            
            # Clean the text
            clean_prompt = clean_text(prompt)
            clean_response = clean_text(response)
            
            # Quality check
            if (is_quality_text(clean_prompt) and 
                is_quality_text(clean_response) and 
                filter_supportive_response(clean_response)):
                self.conversations.append((clean_prompt, clean_response))
        
        print(f"Generated {len(self.conversations)} high-quality conversation pairs")
        return self.conversations
    
    def create_reddit_style_data(self, num_pairs=50000):
        """Create Reddit-style mental health conversations."""
        print(f"Creating {num_pairs} Reddit-style conversation pairs...")
        
        # Reddit-style prompts (more casual, personal)
        reddit_prompts = [
            "Anyone else struggling with {issue}? I feel like I'm the only one.",
            "I need some advice about {problem}. I don't know what to do.",
            "Feeling really {emotion} today. How do you all cope?",
            "I've been dealing with {situation} and it's getting harder.",
            "Does anyone have experience with {topic}? I could use some support.",
            "I'm having a really tough time with {challenge}. Any suggestions?",
            "Feeling {feeling} and could use some encouragement.",
            "I don't know if this is normal, but I've been feeling {emotion}.",
            "Looking for advice on how to handle {difficulty}.",
            "I'm scared about {fear}. Has anyone been through this?",
            "I feel like I'm {self_perception} and I don't know how to change it.",
            "Having trouble with {problem}. Any tips would be appreciated.",
            "I'm worried that {concern}. Am I overthinking this?",
            "Feeling overwhelmed by {stress}. How do you manage?",
            "I need to talk about {topic} but I don't know where to start."
        ]
        
        # Reddit-style responses (supportive, personal, relatable)
        reddit_responses = [
            "I've been there, and I want you to know you're not alone. {support}",
            "Thank you for sharing this. It takes courage to open up. {encouragement}",
            "I understand how you feel. {validation} You're doing better than you think.",
            "I've struggled with {issue} too. {personal_experience}",
            "Your feelings are completely valid. {reassurance}",
            "I'm sorry you're going through this. {support} Remember, this feeling is temporary.",
            "You're not alone in this. Many of us have felt this way. {encouragement}",
            "It's okay to feel {emotion}. {normalization} You're stronger than you know.",
            "I appreciate you sharing this. {validation} You're taking important steps.",
            "I've been in a similar situation. {personal_insight} You've got this.",
            "Your experience matters. {validation} It's okay to not be okay sometimes.",
            "I hear you, and I want you to know that {reassurance}",
            "Thank you for being vulnerable. {support} You're not broken.",
            "I can relate to what you're saying. {personal_connection}",
            "You're doing the best you can, and that's enough. {encouragement}"
        ]
        
        # Fillers for Reddit-style content
        issues = [
            "anxiety", "depression", "social anxiety", "panic attacks", "low self-esteem",
            "relationship problems", "family issues", "work stress", "sleep problems",
            "procrastination", "perfectionism", "imposter syndrome", "loneliness"
        ]
        
        problems = [
            "setting boundaries", "dealing with criticism", "managing stress",
            "building confidence", "overcoming fear", "handling rejection",
            "coping with change", "dealing with guilt", "managing expectations"
        ]
        
        emotions = [
            "anxious", "sad", "overwhelmed", "hopeless", "angry", "scared",
            "lonely", "frustrated", "confused", "tired", "stressed", "worried"
        ]
        
        situations = [
            "work pressure", "family conflicts", "relationship issues", "health concerns",
            "financial stress", "social situations", "life transitions", "past trauma"
        ]
        
        topics = [
            "therapy", "medication", "self-care", "boundaries", "communication",
            "anxiety management", "depression", "trauma recovery", "relationships"
        ]
        
        challenges = [
            "social anxiety", "public speaking", "making friends", "dating",
            "job interviews", "conflict resolution", "time management", "self-discipline"
        ]
        
        feelings = [
            "lost", "stuck", "overwhelmed", "hopeless", "inadequate", "broken",
            "different", "alone", "misunderstood", "exhausted", "numb"
        ]
        
        difficulties = [
            "sleeping", "eating", "concentrating", "motivating myself", "socializing",
            "making decisions", "asking for help", "saying no", "being vulnerable"
        ]
        
        fears = [
            "the future", "failure", "rejection", "being alone", "not being good enough",
            "making mistakes", "change", "conflict", "judgment", "abandonment"
        ]
        
        self_perceptions = [
            "not good enough", "a failure", "broken", "different", "alone",
            "overwhelmed", "lost", "stuck", "hopeless", "worthless", "weak"
        ]
        
        concerns = [
            "I'm not making progress", "I'm burdening others", "I'm not trying hard enough",
            "I'm overreacting", "I'm being too sensitive", "I'm not normal",
            "I'm wasting people's time", "I'm not worthy of help"
        ]
        
        stresses = [
            "work deadlines", "family expectations", "social obligations", "financial pressure",
            "health issues", "relationship problems", "life decisions", "perfectionism"
        ]
        
        support_phrases = [
            "You're stronger than you realize.",
            "This feeling won't last forever.",
            "You're not alone in this struggle.",
            "It's okay to take things one day at a time.",
            "You deserve support and understanding.",
            "You're doing better than you think.",
            "Your feelings are completely valid.",
            "You're taking important steps by reaching out.",
            "You have the strength to get through this.",
            "It's okay to not have all the answers."
        ]
        
        encouragements = [
            "You're making progress, even if it doesn't feel like it.",
            "Every small step counts toward your healing.",
            "You're learning and growing through this experience.",
            "You have more resilience than you give yourself credit for.",
            "It's okay to have difficult days - they don't define you.",
            "You're doing the best you can with what you have.",
            "Your efforts matter and are making a difference.",
            "You're not giving up, and that's incredibly brave.",
            "You're taking care of yourself by seeking support.",
            "You're stronger than your struggles."
        ]
        
        validations = [
            "Your feelings are completely understandable given your situation.",
            "It's normal to feel this way - you're not overreacting.",
            "You have every right to feel {emotion} about this.",
            "Your experience is valid and important.",
            "It's okay to feel overwhelmed - this is a lot to handle.",
            "Your feelings are a normal response to stress and difficulty.",
            "You're not being too sensitive or dramatic.",
            "Your concerns are legitimate and worth addressing.",
            "It's understandable that you feel this way.",
            "Your feelings matter and deserve attention."
        ]
        
        reassurances = [
            "You're not alone in this - many people have felt this way.",
            "This is a common human experience, not a personal failing.",
            "You're not broken or damaged - you're human.",
            "It's okay to not be okay sometimes - that's part of being human.",
            "You're doing better than you think you are.",
            "This feeling will pass, even if it doesn't feel like it now.",
            "You have the strength to handle this, even if you don't feel it.",
            "You're not a burden to others - people care about you.",
            "You deserve compassion and care, especially from yourself.",
            "You're not failing - you're learning and growing."
        ]
        
        personal_experiences = [
            "It took me a while to realize that I wasn't alone in this.",
            "I've learned that it's okay to ask for help when you need it.",
            "What helped me was taking things one small step at a time.",
            "I found that being kind to myself made a huge difference.",
            "It's a process, and there's no shame in taking your time.",
            "I've been there, and I want you to know it gets better.",
            "What worked for me was finding small ways to take care of myself.",
            "I learned that progress isn't always linear, and that's okay.",
            "It took me time to understand that my feelings were valid.",
            "I found that talking about it helped more than I expected."
        ]
        
        personal_insights = [
            "What I learned is that it's okay to not be perfect.",
            "I realized that asking for help is actually a sign of strength.",
            "What helped me was understanding that I'm not alone in this.",
            "I found that being patient with myself was key to my healing.",
            "What I learned is that small steps can lead to big changes.",
            "I realized that my feelings were valid and worth addressing.",
            "What helped me was learning to be kinder to myself.",
            "I found that progress looks different for everyone.",
            "What I learned is that it's okay to take breaks when needed.",
            "I realized that healing isn't a straight line, and that's normal."
        ]
        
        personal_connections = [
            "I've felt exactly what you're describing, and it's really hard.",
            "Your words could have been written by me a few years ago.",
            "I understand this feeling all too well - you're not alone.",
            "I've been in your shoes, and I want you to know it gets better.",
            "I can relate to every word you've written - this is so familiar.",
            "I've struggled with this too, and I want you to know you're not alone.",
            "Your experience mirrors mine in so many ways.",
            "I've been where you are, and I want you to know there's hope.",
            "I can feel the pain in your words because I've been there too.",
            "Your story resonates with me because I've lived something similar."
        ]
        
        normalizations = [
            "Many people experience this, and it's completely normal.",
            "This is a common response to stress and difficulty.",
            "You're not weird or broken for feeling this way.",
            "This is a normal human reaction to challenging circumstances.",
            "Many people have felt exactly what you're feeling right now.",
            "This is a common experience, not a personal failing.",
            "You're not alone in having these feelings.",
            "This is a normal part of the human experience.",
            "Many people struggle with this, and it's okay.",
            "This is a common challenge that many people face."
        ]
        
        # Generate Reddit-style conversations
        for i in range(num_pairs):
            if i % 10000 == 0:
                print(f"Generated {i} Reddit-style pairs...")
            
            prompt_template = random.choice(reddit_prompts)
            response_template = random.choice(reddit_responses)
            
            prompt = prompt_template.format(
                issue=random.choice(issues),
                problem=random.choice(problems),
                emotion=random.choice(emotions),
                situation=random.choice(situations),
                topic=random.choice(topics),
                challenge=random.choice(challenges),
                feeling=random.choice(feelings),
                difficulty=random.choice(difficulties),
                fear=random.choice(fears),
                self_perception=random.choice(self_perceptions),
                concern=random.choice(concerns),
                stress=random.choice(stresses)
            )
            
            response = response_template.format(
                support=random.choice(support_phrases),
                encouragement=random.choice(encouragements),
                validation=random.choice(validations),
                issue=random.choice(issues),
                personal_experience=random.choice(personal_experiences),
                reassurance=random.choice(reassurances),
                emotion=random.choice(emotions),
                personal_insight=random.choice(personal_insights),
                personal_connection=random.choice(personal_connections),
                normalization=random.choice(normalizations)
            )
            
            # Clean the text
            clean_prompt = clean_text(prompt)
            clean_response = clean_text(response)
            
            # Quality check
            if (is_quality_text(clean_prompt) and 
                is_quality_text(clean_response) and 
                filter_supportive_response(clean_response)):
                self.conversations.append((clean_prompt, clean_response))
        
        print(f"Generated {len(self.conversations)} Reddit-style conversation pairs")
        return self.conversations
    
    def create_professional_counseling_data(self, num_pairs=30000):
        """Create professional counseling-style conversations."""
        print(f"Creating {num_pairs} professional counseling conversation pairs...")
        
        # Professional counseling prompts
        counseling_prompts = [
            "I've been experiencing {symptom} and I'm not sure how to cope.",
            "I think I might be dealing with {condition}. What should I do?",
            "I've been feeling {emotion} for {duration} and it's affecting my {life_area}.",
            "I'm having trouble with {function} and I don't know if this is normal.",
            "I've noticed that {behavior} and I'm concerned about it.",
            "I'm struggling with {challenge} and I need some guidance.",
            "I feel like {perception} and I don't know how to change it.",
            "I've been thinking about {thought} and it's causing me distress.",
            "I'm worried that {concern} and I need help understanding this.",
            "I've been having {experience} and I'm not sure how to handle it."
        ]
        
        # Professional counseling responses
        counseling_responses = [
            "I appreciate you sharing this with me. {validation} Let's explore this together.",
            "What you're describing sounds like it's been really challenging for you. {support}",
            "It takes courage to talk about these experiences. {acknowledgment}",
            "I hear that you're struggling with {issue}. {normalization}",
            "Thank you for trusting me with this. {validation} We can work through this.",
            "It sounds like you've been carrying a lot. {support} You don't have to do this alone.",
            "I can see how {situation} would be difficult to manage. {understanding}",
            "What you're experiencing is understandable given your circumstances. {validation}",
            "I want you to know that your feelings are valid. {reassurance}",
            "It's important that you're reaching out for support. {acknowledgment}"
        ]
        
        # Professional counseling fillers
        symptoms = [
            "anxiety", "panic attacks", "depression", "mood swings", "irritability",
            "sleep disturbances", "appetite changes", "concentration problems",
            "fatigue", "restlessness", "hopelessness", "worthlessness"
        ]
        
        conditions = [
            "anxiety disorder", "depression", "PTSD", "bipolar disorder", "OCD",
            "social anxiety", "panic disorder", "generalized anxiety", "adjustment disorder"
        ]
        
        emotions = [
            "sad", "anxious", "angry", "frustrated", "overwhelmed", "hopeless",
            "scared", "lonely", "confused", "numb", "irritable", "restless"
        ]
        
        durations = [
            "a few weeks", "a month", "several months", "a few days", "a week",
            "a couple of weeks", "recently", "lately", "for a while now"
        ]
        
        life_areas = [
            "work", "relationships", "school", "daily activities", "sleep",
            "appetite", "social life", "family", "hobbies", "self-care"
        ]
        
        functions = [
            "sleeping", "eating", "concentrating", "working", "socializing",
            "making decisions", "managing stress", "coping with emotions"
        ]
        
        behaviors = [
            "I've been avoiding social situations", "I've been sleeping too much",
            "I've been eating less", "I've been more irritable", "I've been crying more",
            "I've been isolating myself", "I've been procrastinating more",
            "I've been having racing thoughts", "I've been feeling restless"
        ]
        
        challenges = [
            "managing my emotions", "coping with stress", "maintaining relationships",
            "staying motivated", "dealing with anxiety", "handling criticism",
            "making decisions", "setting boundaries", "communicating effectively"
        ]
        
        perceptions = [
            "I'm not good enough", "I'm a burden to others", "I'm failing at life",
            "I'm broken", "I'm different from everyone else", "I'm not normal",
            "I'm weak", "I'm worthless", "I'm hopeless", "I'm alone"
        ]
        
        thoughts = [
            "hurting myself", "that I'm not good enough", "that I'm a failure",
            "that I don't belong", "that I'm a burden", "that I'm worthless",
            "that I can't handle this", "that I'm alone", "that I'm broken"
        ]
        
        concerns = [
            "I'm not getting better", "I'm getting worse", "I'm not trying hard enough",
            "I'm wasting people's time", "I'm not worthy of help", "I'm overreacting",
            "I'm being too sensitive", "I'm not normal", "I'm a burden"
        ]
        
        experiences = [
            "panic attacks", "mood swings", "sleep problems", "appetite changes",
            "concentration difficulties", "social anxiety", "irritability",
            "hopelessness", "worthlessness", "emotional numbness"
        ]
        
        validations = [
            "Your feelings are completely valid and understandable.",
            "It's normal to feel this way given what you're experiencing.",
            "Your concerns are legitimate and worth addressing.",
            "It's okay to feel overwhelmed by these experiences.",
            "Your feelings are a normal response to stress and difficulty.",
            "It's understandable that you're struggling with this.",
            "Your experience is valid and important.",
            "It's okay to not have all the answers right now.",
            "Your feelings matter and deserve attention.",
            "It's normal to feel uncertain about these experiences."
        ]
        
        support_phrases = [
            "You're not alone in this journey.",
            "I'm here to support you through this.",
            "You don't have to face this by yourself.",
            "We can work through this together.",
            "You're taking important steps by seeking help.",
            "I believe in your ability to heal and grow.",
            "You're stronger than you realize.",
            "This feeling is temporary, even if it doesn't feel like it.",
            "You deserve support and understanding.",
            "You're doing the best you can with what you have."
        ]
        
        acknowledgments = [
            "It takes courage to share these experiences.",
            "You're showing great strength by opening up.",
            "It's brave of you to seek help.",
            "You're taking an important step toward healing.",
            "I appreciate your honesty and vulnerability.",
            "You're doing something important for yourself.",
            "It's not easy to talk about these things.",
            "You're showing wisdom by reaching out.",
            "I respect your courage in sharing this.",
            "You're taking care of yourself by seeking support."
        ]
        
        normalizations = [
            "Many people experience similar challenges.",
            "This is a common human experience.",
            "You're not alone in feeling this way.",
            "Many people struggle with these issues.",
            "This is a normal response to stress.",
            "You're not broken or damaged.",
            "This is part of the human experience.",
            "Many people have been where you are.",
            "This is a common challenge.",
            "You're not weird or abnormal."
        ]
        
        understandings = [
            "I can see how this would be difficult to manage.",
            "It makes sense that you're feeling this way.",
            "I understand why this is challenging for you.",
            "It's understandable that you're struggling with this.",
            "I can see how this affects your daily life.",
            "It makes sense that this is causing you distress.",
            "I understand why this is overwhelming.",
            "It's understandable that you feel this way.",
            "I can see how this impacts your well-being.",
            "It makes sense that you're seeking help."
        ]
        
        reassurances = [
            "You're not alone in this journey.",
            "This feeling will pass with time and support.",
            "You have the strength to work through this.",
            "You're not a burden to others.",
            "You deserve compassion and care.",
            "You're not failing or broken.",
            "You're doing better than you think.",
            "You have the ability to heal and grow.",
            "You're not alone in your struggles.",
            "You're worthy of support and understanding."
        ]
        
        # Generate professional counseling conversations
        for i in range(num_pairs):
            if i % 10000 == 0:
                print(f"Generated {i} professional counseling pairs...")
            
            prompt_template = random.choice(counseling_prompts)
            response_template = random.choice(counseling_responses)
            
            prompt = prompt_template.format(
                symptom=random.choice(symptoms),
                condition=random.choice(conditions),
                emotion=random.choice(emotions),
                duration=random.choice(durations),
                life_area=random.choice(life_areas),
                function=random.choice(functions),
                behavior=random.choice(behaviors),
                challenge=random.choice(challenges),
                perception=random.choice(perceptions),
                thought=random.choice(thoughts),
                concern=random.choice(concerns),
                experience=random.choice(experiences)
            )
            
            response = response_template.format(
                validation=random.choice(validations),
                support=random.choice(support_phrases),
                acknowledgment=random.choice(acknowledgments),
                issue=random.choice(symptoms),
                normalization=random.choice(normalizations),
                understanding=random.choice(understandings),
                situation=random.choice(behaviors),
                reassurance=random.choice(reassurances)
            )
            
            # Clean the text
            clean_prompt = clean_text(prompt)
            clean_response = clean_text(response)
            
            # Quality check
            if (is_quality_text(clean_prompt) and 
                is_quality_text(clean_response) and 
                filter_supportive_response(clean_response)):
                self.conversations.append((clean_prompt, clean_response))
        
        print(f"Generated {len(self.conversations)} professional counseling conversation pairs")
        return self.conversations
    
    def save_dataset(self, filepath, conversations=None):
        """Save the dataset to CSV."""
        if conversations is None:
            conversations = self.conversations
        
        if not conversations:
            print("No conversations to save!")
            return
        
        df = pd.DataFrame(conversations, columns=['prompt', 'response'])
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath} with {len(df)} conversation pairs")
        return len(df)

def main():
    """Main function to create the complete dataset."""
    creator = DatasetCreator()
    
    # Create different types of conversations
    print("Creating comprehensive mental health conversation dataset...")
    
    # Generate synthetic data
    synthetic_data = creator.create_synthetic_mental_health_data(100000)
    
    # Generate Reddit-style data
    reddit_data = creator.create_reddit_style_data(50000)
    
    # Generate professional counseling data
    counseling_data = creator.create_professional_counseling_data(30000)
    
    # Combine all data
    all_conversations = synthetic_data + reddit_data + counseling_data
    
    # Remove duplicates
    unique_conversations = list(set(all_conversations))
    print(f"Total unique conversations: {len(unique_conversations)}")
    
    # Save the complete dataset
    dataset_path = 'data/aura_dataset.csv'
    creator.save_dataset(dataset_path, unique_conversations)
    
    print(f"Dataset creation complete! Saved {len(unique_conversations)} conversation pairs to {dataset_path}")

if __name__ == "__main__":
    import random
    main()
