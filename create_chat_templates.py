#!/usr/bin/env python3
"""
Script to generate 50 chat input templates with placeholders for personal information
including name, surname, address, email, and location from merchant_frame data.
"""

import pandas as pd
import csv
import random

def create_chat_templates():
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
    
    # Create 50 different chat templates
    templates = [
        # Basic introductions
        "Hi there! My name is &&forename&& &&surname&& and my email address is &&email&&. I live at &&location&&.",
        
        "Hello, I'm &&forename&& &&surname&&. You can reach me at &&email&& and I'm located at &&location&&.",
        
        "Good morning! I'm &&forename&& &&surname&&, my email is &&email&&, and my address is &&location&&.",
        
        "Hi, my name is &&forename&& &&surname&&. My contact email is &&email&& and I reside at &&location&&.",
        
        "Greetings! I am &&forename&& &&surname&&, email: &&email&&, address: &&location&&.",
        
        # More conversational styles
        "Hey! I'm &&forename&& &&surname&&. Feel free to email me at &&email&&. I live near &&location&&.",
        
        "Hello there! The name's &&forename&& &&surname&&. Drop me a line at &&email&&. I'm based at &&location&&.",
        
        "Hi everyone! I'm &&forename&& &&surname&&. You can contact me via &&email&&. My home is at &&location&&.",
        
        "Good day! My full name is &&forename&& &&surname&&, my email address is &&email&&, and I call &&location&& home.",
        
        "Nice to meet you! I'm &&forename&& &&surname&&. My email is &&email&& and I'm currently living at &&location&&.",
        
        # Professional introductions
        "Dear colleagues, I am &&forename&& &&surname&&. Please contact me at &&email&&. My office address is &&location&&.",
        
        "Good afternoon. My name is &&forename&& &&surname&&, email: &&email&&, located at &&location&&.",
        
        "Hello, I would like to introduce myself as &&forename&& &&surname&&. My email is &&email&& and my business address is &&location&&.",
        
        "Greetings. I am &&forename&& &&surname&&. For correspondence, please use &&email&&. I am stationed at &&location&&.",
        
        "Dear team, I'm &&forename&& &&surname&&. My contact details are: email &&email&&, address &&location&&.",
        
        # Casual variations
        "What's up! I'm &&forename&& &&surname&&. Hit me up at &&email&&. I'm chilling at &&location&&.",
        
        "Hey everyone! &&forename&& &&surname&& here. Email me at &&email&&. I'm hanging out at &&location&&.",
        
        "Yo! Name's &&forename&& &&surname&&. Shoot me an email at &&email&&. I'm posted up at &&location&&.",
        
        "Sup! I'm &&forename&& &&surname&&. My email's &&email&& and I'm crashing at &&location&&.",
        
        "Hey there! &&forename&& &&surname&& is the name. Email: &&email&&. Currently at &&location&&.",
        
        # Question-response format
        "Who am I? I'm &&forename&& &&surname&&. How to reach me? &&email&&. Where do I live? &&location&&.",
        
        "Name? &&forename&& &&surname&&. Email? &&email&&. Address? &&location&&. Nice to meet you!",
        
        "You asked for my details: Name: &&forename&& &&surname&&, Email: &&email&&, Location: &&location&&.",
        
        "Here's my info: I'm &&forename&& &&surname&&, you can email me at &&email&&, and I live at &&location&&.",
        
        "My details are as follows: &&forename&& &&surname&&, &&email&&, &&location&&.",
        
        # Story-telling style
        "Let me tell you about myself. I'm &&forename&& &&surname&&, you can reach me at &&email&&, and I've been living at &&location&& for a while now.",
        
        "So here's the thing - I'm &&forename&& &&surname&&. If you need to email me, it's &&email&&. Oh, and I live at &&location&&.",
        
        "I should probably introduce myself properly. I'm &&forename&& &&surname&&, my email address is &&email&&, and I make my home at &&location&&.",
        
        "Just to give you some background, I'm &&forename&& &&surname&&. My email contact is &&email&& and my residence is at &&location&&.",
        
        "Here's a bit about me: name's &&forename&& &&surname&&, email is &&email&&, and I call &&location&& my home base.",
        
        # Formal variations
        "I would like to formally introduce myself as Mr./Ms. &&forename&& &&surname&&. My electronic mail address is &&email&& and my residential address is &&location&&.",
        
        "Allow me to present myself: &&forename&& &&surname&&. My email correspondence address is &&email&& and I am domiciled at &&location&&.",
        
        "I hereby provide my contact information: Name: &&forename&& &&surname&&, Email: &&email&&, Residence: &&location&&.",
        
        "For the record, I am &&forename&& &&surname&&. My email address for communication is &&email&& and my address of residence is &&location&&.",
        
        "I am pleased to introduce myself as &&forename&& &&surname&&. My email contact is &&email&& and my address is &&location&&.",
        
        # Short and sweet
        "&&forename&& &&surname&&. Email: &&email&&. Address: &&location&&.",
        
        "I'm &&forename&& &&surname&&. Email me: &&email&&. Live at: &&location&&.",
        
        "&&forename&& &&surname&& here! Email: &&email&&, Location: &&location&&.",
        
        "Name: &&forename&& &&surname&&. Contact: &&email&&. Home: &&location&&.",
        
        "&&forename&& &&surname&&, &&email&&, &&location&&. That's me!",
        
        # Creative variations
        "Picture this: a person named &&forename&& &&surname&&, contactable at &&email&&, living the life at &&location&&.",
        
        "Once upon a time, there was someone called &&forename&& &&surname&& who could be reached at &&email&& and lived at &&location&&. That someone is me!",
        
        "Breaking news: &&forename&& &&surname&& has joined the chat! Email updates available at &&email&&. Live reporting from &&location&&.",
        
        "New player entered the game: &&forename&& &&surname&&. Contact info: &&email&&. Current location: &&location&&.",
        
        "Alert! &&forename&& &&surname&& is now online. Email notifications to &&email&&. Broadcasting from &&location&&.",
        
        # Mixed formats
        "They call me &&forename&& &&surname&&. For business inquiries, email &&email&&. You'll find me at &&location&&.",
        
        "I go by &&forename&& &&surname&&. Drop me a note at &&email&& whenever you're around &&location&&.",
        
        "Friends know me as &&forename&& &&surname&&. Professional contact: &&email&&. Home base: &&location&&.",
        
        "Most people call me &&forename&& &&surname&&. Best way to reach me is &&email&&. I'm usually around &&location&&.",
        
        "I answer to &&forename&& &&surname&&. Electronic correspondence: &&email&&. Physical location: &&location&&.",
        
        # Final variations
        "Just so you know, I'm &&forename&& &&surname&&. My inbox is &&email&& and my doorstep is at &&location&&.",
        
        "For future reference: &&forename&& &&surname&&, reachable at &&email&&, residing at &&location&&.",
        
        "Quick intro: &&forename&& &&surname&& is my name, &&email&& is my email, &&location&& is my place.",
        
        "Personal details: I'm &&forename&& &&surname&&, contact me at &&email&&, visit me at &&location&&.",
        
        "Final answer: &&forename&& &&surname&&, email &&email&&, location &&location&&. Questions?"
    ]
    
    # Create output with templates and sample data
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/chat_templates.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['template_id', 'template_text', 'sample_location', 'sample_forename', 'sample_surname', 'sample_email', 'populated_template']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i, template in enumerate(templates, 1):
            # Pick random values from the data
            sample_location = random.choice(locations) if locations else "Sample Location"
            sample_forename = random.choice(forenames) if forenames else "John"
            sample_surname = random.choice(surnames) if surnames else "Doe"
            sample_email = random.choice(emails) if emails else "john.doe@example.com"
            
            # Create a populated version of the template
            populated_template = template.replace('&&forename&&', sample_forename)
            populated_template = populated_template.replace('&&surname&&', sample_surname)
            populated_template = populated_template.replace('&&email&&', sample_email)
            populated_template = populated_template.replace('&&location&&', sample_location)
            
            writer.writerow({
                'template_id': i,
                'template_text': template,
                'sample_location': sample_location,
                'sample_forename': sample_forename,
                'sample_surname': sample_surname,
                'sample_email': sample_email,
                'populated_template': populated_template
            })
    
    print(f"Created {len(templates)} chat templates!")
    print(f"Templates saved to: {output_file}")
    print(f"\nData sources:")
    print(f"- Forenames: {len(forenames)} options from mplist.csv")
    print(f"- Surnames: {len(surnames)} options from mplist.csv") 
    print(f"- Emails: {len(emails)} options from mplist.csv")
    print(f"- Locations: {len(locations)} options from merchant_frame.csv")
    print("\nTemplate placeholders used:")
    print("- &&forename&& : First name (randomly selected from MP list)")
    print("- &&surname&& : Last name (randomly selected from MP list)")
    print("- &&email&& : Email address (randomly selected from MP list)")
    print("- &&location&& : Address/Location (randomly selected from merchant_frame data)")
    
    # Show first 3 populated templates as examples
    print("\nFirst 3 populated templates:")
    
    # Re-read the file to show populated examples
    df_output = pd.read_csv(output_file)
    for i in range(min(3, len(df_output))):
        print(f"\n{i+1}. Template: {df_output.iloc[i]['template_text']}")
        print(f"   Populated: {df_output.iloc[i]['populated_template']}")
    
    return templates

if __name__ == "__main__":
    templates = create_chat_templates()
