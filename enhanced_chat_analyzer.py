#!/usr/bin/env python3
"""
Enhanced script to generate varied chat inputs with partial information,
analyze with Presidio, create masked versions, and compare for data exposure.
"""

import pandas as pd
import csv
import random
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def extract_address_parts(address):
    """Extract different parts of an address"""
    if pd.isna(address) or not address:
        return "", "", ""
    
    # Split address by commas
    parts = [part.strip() for part in address.split(',')]
    
    # First portion (usually street address)
    first_portion = parts[0] if parts else ""
    
    # Last portion (usually country)
    last_portion = parts[-1] if parts else ""
    
    # Extract UK postcode pattern (letters and numbers at the end)
    postcode_match = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', address)
    postcode = postcode_match.group() if postcode_match else ""
    
    return first_portion, postcode, last_portion

def generate_enhanced_chat_inputs():
    # Initialize Presidio engines
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Read data sources
    merchant_file = '/Users/davidlock/Downloads/soccer data python/testing poe/merchant_frame for ATM.csv'
    df = pd.read_csv(merchant_file)
    locations = df['formatted_address'].dropna().tolist()
    
    mp_file = '/Users/davidlock/Downloads/soccer data python/testing poe/mplist.csv'
    mp_df = pd.read_csv(mp_file)
    forenames = mp_df['Forename'].dropna().tolist()
    surnames = mp_df['Surname'].dropna().tolist()
    emails = mp_df['Email'].dropna().tolist()
    
    # Titles for prefixing names
    titles = ["Mr", "Ms", "Mrs", "Dr", "Prof", "Sir", "Dame", "Lord", "Lady"]
    
    # Enhanced templates with categories and partial information
    template_categories = {
        "full_info": [
            "Hi there! My name is &&forename&& &&surname&& and my email address is &&email&&. I live at &&location&&.",
            "Hello, I'm &&forename&& &&surname&&. You can reach me at &&email&& and I'm located at &&location&&.",
            "Good morning! I'm &&forename&& &&surname&&, my email is &&email&&, and my address is &&location&&.",
            "Nice to meet you! I'm &&forename&& &&surname&&. My email is &&email&& and I'm currently living at &&location&&.",
            "Greetings! I am &&forename&& &&surname&&, email: &&email&&, address: &&location&&."
        ],
        "first_name_only": [
            "Hey! I'm &&forename&&. Nice to meet you!",
            "Hi everyone, &&forename&& here!",
            "Hello, my name is &&forename&&. How are you?",
            "What's up! &&forename&& speaking.",
            "Good day! I'm &&forename&&, pleasure to meet you.",
            "Hi there! I'm &&forename&& and nice to meet you.",
            "Hello! &&forename&& checking in.",
            "What's going on? &&forename&& here."
        ],
        "titled_surname": [
            "Greetings, I am &&title&& &&surname&&.",
            "Hello, &&title&& &&surname&& at your service.",
            "Good afternoon, this is &&title&& &&surname&&.",
            "You can call me &&title&& &&surname&&.",
            "I'm &&title&& &&surname&&, pleased to meet you.",
            "&&title&& &&surname&& speaking.",
            "This is &&title&& &&surname&&.",
            "Good morning, &&title&& &&surname&& here."
        ],
        "email_only": [
            "You can reach me at &&email&&.",
            "My email is &&email&& if you need to contact me.",
            "Drop me a line at &&email&&.",
            "Send me an email: &&email&&.",
            "Contact me via &&email&&.",
            "Email me at &&email&&.",
            "My contact is &&email&&.",
            "Reach out to &&email&&."
        ],
        "address_first_only": [
            "I live on &&address_first&&.",
            "My address is &&address_first&&.",
            "You can find me at &&address_first&&.",
            "I'm located on &&address_first&&.",
            "My place is &&address_first&&.",
            "I'm based at &&address_first&&.",
            "You'll find me on &&address_first&&.",
            "I'm situated on &&address_first&&."
        ],
        "postcode_only": [
            "My postcode is &&postcode&&.",
            "I'm in the &&postcode&& area.",
            "You can find me in &&postcode&&.",
            "My area code is &&postcode&&.",
            "I live in &&postcode&&.",
            "I'm located in &&postcode&&.",
            "My postal area is &&postcode&&.",
            "I'm based in &&postcode&&."
        ],
        "mixed_name_address": [
            "I'm &&forename&& and I live on &&address_first&&.",
            "&&forename&& &&surname&& speaking, I'm located on &&address_first&&.",
            "Hello, &&forename&& &&surname&& here. I live on &&address_first&&.",
            "Hi! I'm &&forename&& from &&address_first&&.",
            "&&forename&& here, located on &&address_first&&.",
            "I'm &&forename&& &&surname&& from &&address_first&&."
        ],
        "mixed_title_email": [
            "&&title&& &&surname&& here, email me at &&email&&.",
            "&&title&& &&surname&& speaking. You can email me at &&email&&.",
            "This is &&title&& &&surname&&, contact: &&email&&.",
            "Professional contact: &&title&& &&surname&&, email &&email&&.",
            "&&title&& &&surname&&, reach me at &&email&&."
        ],
        "mixed_name_postcode": [
            "Hello! I'm &&forename&& from the &&postcode&& area.",
            "You can call me &&title&& &&surname&&, I'm in &&postcode&&.",
            "I'm &&forename&& and my postcode is &&postcode&&.",
            "&&forename&& here from the &&postcode&& area.",
            "Hi there! I'm &&forename&& and I'm around &&postcode&&."
        ],
        "mixed_name_email": [
            "Hi, I'm &&forename&&. My email is &&email&&.",
            "Hey, I'm &&forename&&! Hit me up at &&email&&.",
            "I'm &&forename&&, contact me at &&email&&.",
            "&&forename&& here, email: &&email&&.",
            "Hello! &&forename&& speaking, email &&email&&."
        ],
        "formal_partial": [
            "I am &&title&& &&surname&&, residing in the &&postcode&& postal area.",
            "My name is &&forename&& and I can be contacted at &&email&&.",
            "This is &&title&& &&surname&& from &&address_first&&.",
            "I am &&forename&& &&surname&&, postal code &&postcode&&.",
            "&&title&& &&surname&& here, located on &&address_first&&."
        ],
        "casual_partial": [
            "What's up! &&forename&& from &&postcode&& here.",
            "Yo! &&title&& &&surname&& checking in from &&address_first&&.",
            "Hello! &&forename&& &&surname&& living on &&address_first&&.",
            "Hey there! I'm &&forename&& and I'm around &&postcode&&.",
            "What's good! &&title&& &&surname&& from &&address_first&&."
        ],
        "question_style": [
            "Who am I? I'm &&forename&&. Where am I? &&postcode&&.",
            "Name? &&title&& &&surname&&. Email? &&email&&.",
            "What's my name? &&forename&&. Where do I live? &&address_first&&.",
            "Who's speaking? &&forename&& &&surname&&. Where from? &&postcode&&.",
            "My name? &&title&& &&surname&&. My location? &&address_first&&."
        ],
        "business_style": [
            "This is &&title&& &&surname&& calling from &&address_first&&.",
            "&&forename&& &&surname&& here, office located on &&address_first&&.",
            "Business inquiry from &&forename&& in the &&postcode&& area.",
            "&&title&& &&surname&& speaking, based on &&address_first&&.",
            "Professional call from &&title&& &&surname&&, email &&email&&."
        ]
    }
    
    # Flatten templates with category tracking
    templates = []
    template_to_category = {}
    for category, template_list in template_categories.items():
        for template in template_list:
            templates.append(template)
            template_to_category[template] = category
    
    print("Generating 10,000 enhanced chat inputs with partial information...")
    print(f"Data sources: {len(forenames)} forenames, {len(surnames)} surnames, {len(emails)} emails, {len(locations)} locations")
    print(f"Template categories: {len(template_categories)} categories with {len(templates)} total templates")
    
    results = []
    
    # Generate 10,000 random chat inputs with varied information
    for i in range(10000):
        # Randomly select base data
        forename = random.choice(forenames)
        surname = random.choice(surnames)
        email = random.choice(emails)
        location = random.choice(locations)
        title = random.choice(titles)
        template = random.choice(templates)
        
        # Extract address parts
        address_first, postcode, _ = extract_address_parts(location)
        
        # Get template category
        template_category = template_to_category[template]
        
        # Create populated chat input
        chat_input = template.replace('&&forename&&', forename)
        chat_input = chat_input.replace('&&surname&&', surname)
        chat_input = chat_input.replace('&&email&&', email)
        chat_input = chat_input.replace('&&location&&', location)
        chat_input = chat_input.replace('&&title&&', title)
        chat_input = chat_input.replace('&&address_first&&', address_first)
        chat_input = chat_input.replace('&&postcode&&', postcode)
        
        # Analyze with Presidio
        analysis_results = analyzer.analyze(text=chat_input, language='en')
        
        # Create masked version using anonymizer
        if analysis_results:
            anonymized_result = anonymizer.anonymize(
                text=chat_input,
                analyzer_results=analysis_results
            )
            masked_input = anonymized_result.text
        else:
            masked_input = chat_input
        
        # Extract entity information
        entity_types = []
        confidence_scores = []
        detected_entities = []
        
        for result in analysis_results:
            entity_types.append(result.entity_type)
            confidence_scores.append(f"{result.score:.3f}")
            detected_text = chat_input[result.start:result.end]
            detected_entities.append(f"{result.entity_type}:{detected_text}")
        
        # Check for data exposure by comparing used data with masked result
        data_still_exposed = []
        
        # Check if original data elements are still visible in masked text
        if forename.lower() in masked_input.lower() and forename.lower() not in ["x", ""]:
            data_still_exposed.append(f"forename:{forename}")
        if surname.lower() in masked_input.lower() and surname.lower() not in ["x", ""]:
            data_still_exposed.append(f"surname:{surname}")
        if email.lower() in masked_input.lower():
            data_still_exposed.append(f"email:{email}")
        if address_first and len(address_first) > 3 and address_first.lower() in masked_input.lower():
            data_still_exposed.append(f"address_first:{address_first}")
        if postcode and len(postcode) > 2 and postcode.lower() in masked_input.lower():
            data_still_exposed.append(f"postcode:{postcode}")
        
        # Store results
        result_row = {
            'input_id': i + 1,
            'template_category': template_category,
            'original_chat_input': chat_input,
            'masked_chat_input': masked_input,
            'forename_used': forename,
            'surname_used': surname,
            'email_used': email,
            'location_used': location,
            'title_used': title,
            'address_first_used': address_first,
            'postcode_used': postcode,
            'detected_entities': '; '.join(detected_entities) if detected_entities else 'No PII detected',
            'entity_types': '; '.join(entity_types) if entity_types else 'None',
            'confidence_scores': '; '.join(confidence_scores) if confidence_scores else 'N/A',
            'total_entities_found': len(analysis_results),
            'data_still_exposed': '; '.join(data_still_exposed) if data_still_exposed else 'None',
            'masking_effective': 'No' if data_still_exposed else 'Yes',
            'exposure_category': template_category if data_still_exposed else 'None'
        }
        
        results.append(result_row)
        
        # Print progress every 1000 rows
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} chat inputs...")
    
    # Save results to CSV
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/enhanced_chat_analysis_10k.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Generate comprehensive statistics
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total chat inputs analyzed: 10,000")
    print(f"Results saved to: {output_file}")
    
    # Calculate statistics
    total_entities = sum([row['total_entities_found'] for row in results])
    inputs_with_pii = len([row for row in results if row['total_entities_found'] > 0])
    inputs_with_exposure = len([row for row in results if row['data_still_exposed'] != 'None'])
    effective_masking = len([row for row in results if row['masking_effective'] == 'Yes'])
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Inputs with PII detected: {inputs_with_pii:,} ({inputs_with_pii/100:.1f}%)")
    print(f"Inputs without PII: {10000-inputs_with_pii:,} ({(10000-inputs_with_pii)/100:.1f}%)")
    print(f"Total entities detected: {total_entities:,}")
    print(f"Average entities per input: {total_entities/10000:.2f}")
    
    print(f"\n=== MASKING EFFECTIVENESS ===")
    print(f"Inputs with effective masking: {effective_masking:,} ({effective_masking/100:.1f}%)")
    print(f"Inputs with data still exposed: {inputs_with_exposure:,} ({inputs_with_exposure/100:.1f}%)")
    
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
    
    # Analyze data exposure patterns
    if inputs_with_exposure > 0:
        exposure_types = {}
        for result in results:
            if result['data_still_exposed'] != 'None':
                exposures = result['data_still_exposed'].split('; ')
                for exposure in exposures:
                    exposure_type = exposure.split(':')[0]
                    exposure_types[exposure_type] = exposure_types.get(exposure_type, 0) + 1
        
        print(f"\n=== DATA EXPOSURE BREAKDOWN ===")
        for exposure_type, count in sorted(exposure_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / inputs_with_exposure) * 100
            print(f"{exposure_type}: {count:,} instances ({percentage:.1f}% of exposed inputs)")
    
    # Analyze exposure by template category
    category_exposure = {}
    category_totals = {}
    
    for result in results:
        category = result['template_category']
        category_totals[category] = category_totals.get(category, 0) + 1
        
        if result['data_still_exposed'] != 'None':
            category_exposure[category] = category_exposure.get(category, 0) + 1
    
    print(f"\n=== EXPOSURE BY TEMPLATE CATEGORY ===")
    for category in sorted(category_totals.keys()):
        total = category_totals[category]
        exposed = category_exposure.get(category, 0)
        percentage = (exposed / total) * 100 if total > 0 else 0
        print(f"{category}: {exposed}/{total} exposed ({percentage:.1f}%)")
    
    # Show template category distribution
    print(f"\n=== TEMPLATE CATEGORY DISTRIBUTION ===")
    for category, count in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / 10000) * 100
        print(f"{category}: {count:,} uses ({percentage:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    results = generate_enhanced_chat_inputs()
