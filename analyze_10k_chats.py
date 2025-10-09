#!/usr/bin/env python3
"""
Script to generate 10,000 random chat inputs using data from mplist.csv and merchant_frame,
then process them through Presidio analyzer to detect PII.
"""

import pandas as pd
import csv
import random
from presidio_analyzer import AnalyzerEngine

def generate_and_analyze_chat_inputs():
    # Initialize Presidio analyzer
    analyzer = AnalyzerEngine()
    
    # Read the merchant frame data to get location information
    merchant_file = '/Users/davidlock/Downloads/soccer data python/testing poe/merchant_frame for ATM.csv'
    df = pd.read_csv(merchant_file)
    
    # Get unique locations from the formatted_address column
    locations = df['formatted_address'].dropna().tolist()
    
    # Read the MP list data to get forename, surname, and email information
    mp_file = '/Users/davidlock/Downloads/soccer data python/testing poe/mplist.csv'
    mp_df = pd.read_csv(mp_file)
    
    # Get the personal data from MP list
    forenames = mp_df['Forename'].dropna().tolist()
    surnames = mp_df['Surname'].dropna().tolist()
    emails = mp_df['Email'].dropna().tolist()
    
    # Chat templates (same 55 templates from before)
    templates = [
        "Hi there! My name is &&forename&& &&surname&& and my email address is &&email&&. I live at &&location&&.",
        "Hello, I'm &&forename&& &&surname&&. You can reach me at &&email&& and I'm located at &&location&&.",
        "Good morning! I'm &&forename&& &&surname&&, my email is &&email&&, and my address is &&location&&.",
        "Hi, my name is &&forename&& &&surname&&. My contact email is &&email&& and I reside at &&location&&.",
        "Greetings! I am &&forename&& &&surname&&, email: &&email&&, address: &&location&&.",
        "Hey! I'm &&forename&& &&surname&&. Feel free to email me at &&email&&. I live near &&location&&.",
        "Hello there! The name's &&forename&& &&surname&&. Drop me a line at &&email&&. I'm based at &&location&&.",
        "Hi everyone! I'm &&forename&& &&surname&&. You can contact me via &&email&&. My home is at &&location&&.",
        "Good day! My full name is &&forename&& &&surname&&, my email address is &&email&&, and I call &&location&& home.",
        "Nice to meet you! I'm &&forename&& &&surname&&. My email is &&email&& and I'm currently living at &&location&&.",
        "Dear colleagues, I am &&forename&& &&surname&&. Please contact me at &&email&&. My office address is &&location&&.",
        "Good afternoon. My name is &&forename&& &&surname&&, email: &&email&&, located at &&location&&.",
        "Hello, I would like to introduce myself as &&forename&& &&surname&&. My email is &&email&& and my business address is &&location&&.",
        "Greetings. I am &&forename&& &&surname&&. For correspondence, please use &&email&&. I am stationed at &&location&&.",
        "Dear team, I'm &&forename&& &&surname&&. My contact details are: email &&email&&, address &&location&&.",
        "What's up! I'm &&forename&& &&surname&&. Hit me up at &&email&&. I'm chilling at &&location&&.",
        "Hey everyone! &&forename&& &&surname&& here. Email me at &&email&&. I'm hanging out at &&location&&.",
        "Yo! Name's &&forename&& &&surname&&. Shoot me an email at &&email&&. I'm posted up at &&location&&.",
        "Sup! I'm &&forename&& &&surname&&. My email's &&email&& and I'm crashing at &&location&&.",
        "Hey there! &&forename&& &&surname&& is the name. Email: &&email&&. Currently at &&location&&.",
        "Who am I? I'm &&forename&& &&surname&&. How to reach me? &&email&&. Where do I live? &&location&&.",
        "Name? &&forename&& &&surname&&. Email? &&email&&. Address? &&location&&. Nice to meet you!",
        "You asked for my details: Name: &&forename&& &&surname&&, Email: &&email&&, Location: &&location&&.",
        "Here's my info: I'm &&forename&& &&surname&&, you can email me at &&email&&, and I live at &&location&&.",
        "My details are as follows: &&forename&& &&surname&&, &&email&&, &&location&&.",
        "Let me tell you about myself. I'm &&forename&& &&surname&&, you can reach me at &&email&&, and I've been living at &&location&& for a while now.",
        "So here's the thing - I'm &&forename&& &&surname&&. If you need to email me, it's &&email&&. Oh, and I live at &&location&&.",
        "I should probably introduce myself properly. I'm &&forename&& &&surname&&, my email address is &&email&&, and I make my home at &&location&&.",
        "Just to give you some background, I'm &&forename&& &&surname&&. My email contact is &&email&& and my residence is at &&location&&.",
        "Here's a bit about me: name's &&forename&& &&surname&&, email is &&email&&, and I call &&location&& my home base.",
        "I would like to formally introduce myself as Mr./Ms. &&forename&& &&surname&&. My electronic mail address is &&email&& and my residential address is &&location&&.",
        "Allow me to present myself: &&forename&& &&surname&&. My email correspondence address is &&email&& and I am domiciled at &&location&&.",
        "I hereby provide my contact information: Name: &&forename&& &&surname&&, Email: &&email&&, Residence: &&location&&.",
        "For the record, I am &&forename&& &&surname&&. My email address for communication is &&email&& and my address of residence is &&location&&.",
        "I am pleased to introduce myself as &&forename&& &&surname&&. My email contact is &&email&& and my address is &&location&&.",
        "&&forename&& &&surname&&. Email: &&email&&. Address: &&location&&.",
        "I'm &&forename&& &&surname&&. Email me: &&email&&. Live at: &&location&&.",
        "&&forename&& &&surname&& here! Email: &&email&&, Location: &&location&&.",
        "Name: &&forename&& &&surname&&. Contact: &&email&&. Home: &&location&&.",
        "&&forename&& &&surname&&, &&email&&, &&location&&. That's me!",
        "Picture this: a person named &&forename&& &&surname&&, contactable at &&email&&, living the life at &&location&&.",
        "Once upon a time, there was someone called &&forename&& &&surname&& who could be reached at &&email&& and lived at &&location&&. That someone is me!",
        "Breaking news: &&forename&& &&surname&& has joined the chat! Email updates available at &&email&&. Live reporting from &&location&&.",
        "New player entered the game: &&forename&& &&surname&&. Contact info: &&email&&. Current location: &&location&&.",
        "Alert! &&forename&& &&surname&& is now online. Email notifications to &&email&&. Broadcasting from &&location&&.",
        "They call me &&forename&& &&surname&&. For business inquiries, email &&email&&. You'll find me at &&location&&.",
        "I go by &&forename&& &&surname&&. Drop me a note at &&email&& whenever you're around &&location&&.",
        "Friends know me as &&forename&& &&surname&&. Professional contact: &&email&&. Home base: &&location&&.",
        "Most people call me &&forename&& &&surname&&. Best way to reach me is &&email&&. I'm usually around &&location&&.",
        "I answer to &&forename&& &&surname&&. Electronic correspondence: &&email&&. Physical location: &&location&&.",
        "Just so you know, I'm &&forename&& &&surname&&. My inbox is &&email&& and my doorstep is at &&location&&.",
        "For future reference: &&forename&& &&surname&&, reachable at &&email&&, residing at &&location&&.",
        "Quick intro: &&forename&& &&surname&& is my name, &&email&& is my email, &&location&& is my place.",
        "Personal details: I'm &&forename&& &&surname&&, contact me at &&email&&, visit me at &&location&&.",
        "Final answer: &&forename&& &&surname&&, email &&email&&, location &&location&&. Questions?"
    ]
    
    print("Generating 10,000 random chat inputs and analyzing with Presidio...")
    print(f"Data sources: {len(forenames)} forenames, {len(surnames)} surnames, {len(emails)} emails, {len(locations)} locations")
    
    results = []
    
    # Generate 10,000 random chat inputs
    for i in range(10000):
        # Randomly select values
        forename = random.choice(forenames)
        surname = random.choice(surnames)
        email = random.choice(emails)
        location = random.choice(locations)
        template = random.choice(templates)
        
        # Create populated chat input
        chat_input = template.replace('&&forename&&', forename)
        chat_input = chat_input.replace('&&surname&&', surname)
        chat_input = chat_input.replace('&&email&&', email)
        chat_input = chat_input.replace('&&location&&', location)
        
        # Analyze with Presidio
        analysis_results = analyzer.analyze(text=chat_input, language='en')
        
        # Extract entity information
        entity_types = []
        confidence_scores = []
        detected_entities = []
        
        for result in analysis_results:
            entity_types.append(result.entity_type)
            confidence_scores.append(f"{result.score:.3f}")
            # Extract the actual detected text
            detected_text = chat_input[result.start:result.end]
            detected_entities.append(f"{result.entity_type}:{detected_text}")
        
        # Store results
        result_row = {
            'input_id': i + 1,
            'chat_input': chat_input,
            'forename_used': forename,
            'surname_used': surname,
            'email_used': email,
            'location_used': location,
            'detected_entities': '; '.join(detected_entities) if detected_entities else 'No PII detected',
            'entity_types': '; '.join(entity_types) if entity_types else 'None',
            'confidence_scores': '; '.join(confidence_scores) if confidence_scores else 'N/A',
            'total_entities_found': len(analysis_results)
        }
        
        results.append(result_row)
        
        # Print progress every 1000 rows
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} chat inputs...")
    
    # Save results to CSV
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/chat_analysis_10k.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total chat inputs analyzed: 10,000")
    print(f"Results saved to: {output_file}")
    
    # Calculate statistics
    total_entities = sum([row['total_entities_found'] for row in results])
    inputs_with_pii = len([row for row in results if row['total_entities_found'] > 0])
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Inputs with PII detected: {inputs_with_pii:,} ({inputs_with_pii/100:.1f}%)")
    print(f"Inputs without PII: {10000-inputs_with_pii:,} ({(10000-inputs_with_pii)/100:.1f}%)")
    print(f"Total entities detected: {total_entities:,}")
    print(f"Average entities per input: {total_entities/10000:.2f}")
    
    # Count entity types
    all_entity_types = []
    for result in results:
        if result['entity_types'] != 'None':
            all_entity_types.extend(result['entity_types'].split('; '))
    
    if all_entity_types:
        entity_counts = {}
        for entity in all_entity_types:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        print(f"\n=== ENTITY TYPE BREAKDOWN ===")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_entities) * 100
            print(f"{entity_type}: {count:,} occurrences ({percentage:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    results = generate_and_analyze_chat_inputs()
