#!/usr/bin/env python3
"""
Extract address components (first line, town/city, postcode) from address field
and create both clean and dirty versions of these new fields
"""

import pandas as pd
import re
import random

# Set random seed for reproducibility
random.seed(42)

def extract_postcode(address):
    """Extract UK postcode from address"""
    if not address or pd.isna(address):
        return ''
    
    # UK postcode pattern - more flexible to handle various formats
    postcode_patterns = [
        r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b',  # Standard UK postcode
        r'\b([A-Z]{1,2}\d[A-Z]?\s*\d[A-Z]{2})\b',       # Alternative format
    ]
    
    for pattern in postcode_patterns:
        match = re.search(pattern, address.upper())
        if match:
            postcode = match.group(1)
            # Normalize spacing
            postcode = re.sub(r'([A-Z0-9]+)(\d[A-Z]{2})$', r'\1 \2', postcode)
            return postcode
    
    return ''

def extract_town_city(address):
    """Extract town/city from address - typically the part before the postcode"""
    if not address or pd.isna(address):
        return ''
    
    # Remove postcode first
    address_without_postcode = re.sub(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b.*$', '', address, flags=re.IGNORECASE)
    
    # Split by comma and take the last meaningful part before postcode
    parts = [part.strip() for part in address_without_postcode.split(',') if part.strip()]
    
    if parts:
        # Remove common country indicators
        last_part = parts[-1]
        if last_part.upper() in ['UK', 'UNITED KINGDOM', 'ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']:
            if len(parts) > 1:
                last_part = parts[-2]
        
        # Clean up common suffixes
        last_part = re.sub(r'\s+(UK|United Kingdom|England|Scotland|Wales)$', '', last_part, flags=re.IGNORECASE)
        
        return last_part.strip()
    
    return ''

def extract_first_line(address):
    """Extract first line of address (street address)"""
    if not address or pd.isna(address):
        return ''
    
    # Take the first part before the first comma, or first meaningful chunk
    parts = [part.strip() for part in address.split(',') if part.strip()]
    
    if parts:
        first_line = parts[0]
        
        # If the first part looks like just a number, combine with the next part
        if len(parts) > 1 and re.match(r'^\d+$', first_line.strip()):
            first_line = f"{first_line}, {parts[1]}"
        
        return first_line.strip()
    
    return address.strip()

def corrupt_address_component(component, component_type):
    """Apply corruptions specific to address components"""
    if not component or pd.isna(component) or component == '':
        return component
    
    if component_type == 'postcode':
        corruptions = [
            # Space variations
            lambda x: x.replace(' ', ''),
            lambda x: x.replace(' ', '-'),
            lambda x: f" {x} ",
            
            # Case variations
            lambda x: x.lower(),
            lambda x: x.upper(),
            
            # Character substitutions
            lambda x: x.replace('O', '0') if 'O' in x else x,
            lambda x: x.replace('0', 'O') if '0' in x else x,
            lambda x: x.replace('I', '1') if 'I' in x else x,
            lambda x: x.replace('1', 'I') if '1' in x else x,
            
            # Missing characters
            lambda x: x[:-1] if len(x) > 3 else x,
            
            # No corruption (keep some clean)
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ]
    
    elif component_type == 'town_city':
        corruptions = [
            # Case variations
            lambda x: x.upper(),
            lambda x: x.lower(),
            lambda x: x.title(),
            
            # Extra spaces
            lambda x: f" {x} ",
            lambda x: f"  {x}",
            lambda x: f"{x}  ",
            
            # Character corruptions
            lambda x: x.replace('a', '@') if 'a' in x else x,
            lambda x: x.replace('e', '3') if 'e' in x else x,
            
            # Missing characters
            lambda x: x[1:] if len(x) > 3 else x,
            lambda x: x[:-1] if len(x) > 3 else x,
            
            # No corruption (keep many clean)
            lambda x: x,
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ]
    
    else:  # first_line
        corruptions = [
            # Case variations
            lambda x: x.upper(),
            lambda x: x.lower(),
            
            # Extra spaces
            lambda x: f" {x} ",
            lambda x: x.replace(',', ' , '),
            lambda x: x.replace(',', ''),
            
            # Number corruptions
            lambda x: re.sub(r'\b(\d+)\b', lambda m: str(int(m.group(1)) + random.randint(-2, 2)) if int(m.group(1)) > 2 else m.group(1), x),
            
            # Abbreviations
            lambda x: x.replace('Street', 'St') if 'Street' in x else x,
            lambda x: x.replace('Road', 'Rd') if 'Road' in x else x,
            lambda x: x.replace('Avenue', 'Ave') if 'Avenue' in x else x,
            lambda x: x.replace(' St', ' Street') if ' St' in x and ' Street' not in x else x,
            
            # No corruption (keep some clean)
            lambda x: x,
            lambda x: x,
            lambda x: x,
        ]
    
    return random.choice(corruptions)(component)

# Files to process
files_to_process = [
    'standardized_650_clean_and_dirty.csv',
    'standardized_9056_clean_and_dirty.csv',
    'unified_all_clean_and_dirty_pii_data.csv'
]

print("🏠 EXTRACTING ADDRESS COMPONENTS AND CREATING DIRTY VERSIONS")
print("=" * 60)

for filename in files_to_process:
    print(f"\n📂 Processing {filename}...")
    
    try:
        # Load the dataset
        df = pd.read_csv(filename)
        print(f"   Loaded: {len(df):,} records with {len(df.columns)} fields")
        
        # Extract address components
        print("   Extracting address components...")
        df['address_first_line'] = df['address'].apply(extract_first_line)
        df['address_town_city'] = df['address'].apply(extract_town_city)
        df['address_postcode'] = df['address'].apply(extract_postcode)
        
        # Create dirty versions of address components
        print("   Creating dirty versions of address components...")
        
        # For records that are already marked as 'dirty', corrupt the new address fields
        # For 'clean' records, keep the address components clean
        df['address_first_line_dirty'] = df.apply(
            lambda row: corrupt_address_component(row['address_first_line'], 'first_line') 
            if row.get('data_version') == 'dirty' else row['address_first_line'], 
            axis=1
        )
        
        df['address_town_city_dirty'] = df.apply(
            lambda row: corrupt_address_component(row['address_town_city'], 'town_city') 
            if row.get('data_version') == 'dirty' else row['address_town_city'], 
            axis=1
        )
        
        df['address_postcode_dirty'] = df.apply(
            lambda row: corrupt_address_component(row['address_postcode'], 'postcode') 
            if row.get('data_version') == 'dirty' else row['address_postcode'], 
            axis=1
        )
        
        # Count extraction success
        first_line_success = (df['address_first_line'] != '').sum()
        town_city_success = (df['address_town_city'] != '').sum()
        postcode_success = (df['address_postcode'] != '').sum()
        
        print(f"   Address component extraction results:")
        print(f"     First line: {first_line_success:,}/{len(df):,} ({first_line_success/len(df)*100:.1f}%)")
        print(f"     Town/city: {town_city_success:,}/{len(df):,} ({town_city_success/len(df)*100:.1f}%)")
        print(f"     Postcode: {postcode_success:,}/{len(df):,} ({postcode_success/len(df)*100:.1f}%)")
        
        # Save the enhanced dataset
        df.to_csv(filename, index=False)
        print(f"   ✅ Updated {filename}")
        print(f"   New structure: {len(df):,} records × {len(df.columns)} fields")
        
    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")

print(f"\n📊 EXTRACTION AND CORRUPTION SUMMARY")
print(f"New fields added to each dataset:")
print(f"  • address_first_line (clean version)")
print(f"  • address_town_city (clean version)")  
print(f"  • address_postcode (clean version)")
print(f"  • address_first_line_dirty (corrupted for dirty records)")
print(f"  • address_town_city_dirty (corrupted for dirty records)")
print(f"  • address_postcode_dirty (corrupted for dirty records)")

# Show some examples
print(f"\n📋 EXTRACTION EXAMPLES:")
try:
    df_sample = pd.read_csv(files_to_process[0])
    print(f"\nSample extractions from {files_to_process[0]}:")
    
    # Show a few examples
    sample_records = df_sample.head(5)
    for idx, row in sample_records.iterrows():
        print(f"\nExample {idx + 1}:")
        print(f"  Full Address: {row['address']}")
        print(f"  First Line: '{row['address_first_line']}'")
        print(f"  Town/City: '{row['address_town_city']}'")
        print(f"  Postcode: '{row['address_postcode']}'")
        
        if row.get('data_version') == 'dirty':
            print(f"  Dirty First Line: '{row['address_first_line_dirty']}'")
            print(f"  Dirty Town/City: '{row['address_town_city_dirty']}'")
            print(f"  Dirty Postcode: '{row['address_postcode_dirty']}'")
        
        if idx >= 2:  # Show first 3 examples
            break

except Exception as e:
    print(f"Could not show examples: {e}")

print(f"\n✅ ADDRESS COMPONENT EXTRACTION AND CORRUPTION COMPLETE")
print(f"🎯 All datasets now include granular address fields with clean and dirty versions")
print(f"📊 Ready for component-level PII detection testing")
