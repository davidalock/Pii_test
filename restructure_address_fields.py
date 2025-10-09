#!/usr/bin/env python3
"""
Remove separate dirty address fields and instead create dirty addresses
from which the components are derived. Track the corruption changes.
"""

import pandas as pd
import re
import random

# Set random seed for reproducibility
random.seed(42)

def corrupt_full_address(address):
    """Apply corruptions to the full address, then derive components from the corrupted version"""
    if not address or pd.isna(address):
        return address, 'no_change'
    
    corruptions = [
        # Case variations
        (lambda x: x.upper(), 'uppercase'),
        (lambda x: x.lower(), 'lowercase'),
        (lambda x: x.title(), 'title_case'),
        
        # Extra spaces and punctuation
        (lambda x: f" {x} ", 'extra_spaces'),
        (lambda x: x.replace(",", " , "), 'spaced_commas'),
        (lambda x: x.replace(",", ""), 'removed_commas'),
        (lambda x: x.replace("  ", " "), 'normalized_spaces'),
        
        # Number corruptions
        (lambda x: re.sub(r'\b(\d+)\b', lambda m: str(int(m.group(1)) + random.randint(-3, 3)) if int(m.group(1)) > 3 else m.group(1), x), 'number_change'),
        
        # Abbreviations
        (lambda x: x.replace("Street", "St") if "Street" in x else x, 'street_abbreviation'),
        (lambda x: x.replace("Road", "Rd") if "Road" in x else x, 'road_abbreviation'),
        (lambda x: x.replace("Avenue", "Ave") if "Avenue" in x else x, 'avenue_abbreviation'),
        (lambda x: x.replace(" St ", " Street ") if " St " in x else x, 'street_expansion'),
        (lambda x: x.replace(" Rd ", " Road ") if " Rd " in x else x, 'road_expansion'),
        
        # Postcode corruptions
        (lambda x: re.sub(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?) (\d[A-Z]{2})\b', r'\1\2', x), 'postcode_no_space'),
        (lambda x: re.sub(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)(\d[A-Z]{2})\b', r'\1 \2', x), 'postcode_add_space'),
        
        # Character substitutions
        (lambda x: x.replace("O", "0") if "O" in x else x, 'O_to_0'),
        (lambda x: x.replace("0", "O") if "0" in x else x, '0_to_O'),
        (lambda x: x.replace("I", "1") if "I" in x else x, 'I_to_1'),
        (lambda x: x.replace("1", "I") if "1" in x else x, '1_to_I'),
        
        # Missing parts
        (lambda x: x.rsplit(",", 1)[0] if x.count(",") > 2 else x, 'remove_last_part'),
        
        # Extra country indicators
        (lambda x: f"{x}, England" if "UK" in x and "England" not in x else x, 'add_england'),
        (lambda x: x.replace("UK", "United Kingdom") if "UK" in x and "United Kingdom" not in x else x, 'expand_uk'),
        (lambda x: x.replace("United Kingdom", "UK") if "United Kingdom" in x else x, 'abbreviate_uk'),
        
        # No corruption (keep some addresses clean)
        (lambda x: x, 'no_change'),
        (lambda x: x, 'no_change'),
        (lambda x: x, 'no_change'),
        (lambda x: x, 'no_change'),
    ]
    
    corruption_func, change_type = random.choice(corruptions)
    return corruption_func(address), change_type

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

# Files to process
files_to_process = [
    'standardized_650_clean_and_dirty.csv',
    'standardized_9056_clean_and_dirty.csv',
    'unified_all_clean_and_dirty_pii_data.csv'
]

print("🔧 RESTRUCTURING ADDRESS FIELDS - REMOVING DIRTY COMPONENT FIELDS")
print("=" * 70)

for filename in files_to_process:
    print(f"\n📂 Processing {filename}...")
    
    try:
        # Load the dataset
        df = pd.read_csv(filename)
        print(f"   Loaded: {len(df):,} records with {len(df.columns)} fields")
        
        # Remove the dirty address component fields
        dirty_fields_to_remove = [
            'address_first_line_dirty',
            'address_town_city_dirty',
            'address_postcode_dirty'
        ]
        
        fields_removed = []
        for field in dirty_fields_to_remove:
            if field in df.columns:
                df = df.drop(columns=[field])
                fields_removed.append(field)
        
        print(f"   Removed fields: {fields_removed}")
        
        # For dirty records, corrupt the address and derive components from corrupted address
        print("   Creating corrupted addresses for dirty records...")
        
        # Initialize corruption tracking
        df['address_corruption_type'] = 'no_change'
        
        # Process dirty records
        dirty_mask = df['data_version'] == 'dirty'
        dirty_count = dirty_mask.sum()
        
        if dirty_count > 0:
            print(f"   Processing {dirty_count} dirty records...")
            
            # Apply corruption to addresses for dirty records
            corrupted_addresses = []
            corruption_types = []
            
            for idx, row in df[dirty_mask].iterrows():
                corrupted_addr, corruption_type = corrupt_full_address(row['address'])
                corrupted_addresses.append(corrupted_addr)
                corruption_types.append(corruption_type)
            
            # Update addresses for dirty records
            df.loc[dirty_mask, 'address'] = corrupted_addresses
            df.loc[dirty_mask, 'address_corruption_type'] = corruption_types
            
            # Count corruption types
            corruption_counts = pd.Series(corruption_types).value_counts()
            print(f"   Corruption type distribution:")
            for corruption_type, count in corruption_counts.head(10).items():
                print(f"     {corruption_type}: {count} ({count/len(corruption_types)*100:.1f}%)")
        
        # Re-derive address components from the (potentially corrupted) addresses
        print("   Re-deriving address components from addresses...")
        df['address_first_line'] = df['address'].apply(extract_first_line)
        df['address_town_city'] = df['address'].apply(extract_town_city)
        df['address_postcode'] = df['address'].apply(extract_postcode)
        
        # Count extraction success
        first_line_success = (df['address_first_line'] != '').sum()
        town_city_success = (df['address_town_city'] != '').sum()
        postcode_success = (df['address_postcode'] != '').sum()
        
        print(f"   Address component extraction results:")
        print(f"     First line: {first_line_success:,}/{len(df):,} ({first_line_success/len(df)*100:.1f}%)")
        print(f"     Town/city: {town_city_success:,}/{len(df):,} ({town_city_success/len(df)*100:.1f}%)")
        print(f"     Postcode: {postcode_success:,}/{len(df):,} ({postcode_success/len(df)*100:.1f}%)")
        
        # Save the restructured dataset
        df.to_csv(filename, index=False)
        print(f"   ✅ Updated {filename}")
        print(f"   New structure: {len(df):,} records × {len(df.columns)} fields")
        
    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")

print(f"\n📊 RESTRUCTURING SUMMARY")
print(f"Changes made:")
print(f"  ❌ Removed: address_first_line_dirty, address_town_city_dirty, address_postcode_dirty")
print(f"  ✅ Added: address_corruption_type (tracks changes to address)")
print(f"  🔄 Modified: address field (corrupted for dirty records)")
print(f"  🔄 Re-derived: address_first_line, address_town_city, address_postcode (from corrupted addresses)")

# Show the final structure
try:
    df_sample = pd.read_csv(files_to_process[0])
    print(f"\n📋 FINAL FIELD STRUCTURE ({len(df_sample.columns)} fields):")
    for i, col in enumerate(df_sample.columns, 1):
        marker = '🆕' if col == 'address_corruption_type' else '🔄' if 'address' in col else '📄'
        print(f"  {i:2d}. {marker} {col}")
    
    # Show examples of corruptions
    print(f"\n📋 CORRUPTION EXAMPLES:")
    print("=" * 40)
    
    dirty_sample = df_sample[df_sample['data_version'] == 'dirty'].head(5)
    for i, (_, row) in enumerate(dirty_sample.iterrows()):
        print(f"\nExample {i+1} (Corruption: {row['address_corruption_type']}):")
        print(f"  Address: {row['address']}")
        print(f"  First Line: '{row['address_first_line']}'")
        print(f"  Town/City: '{row['address_town_city']}'")
        print(f"  Postcode: '{row['address_postcode']}'")
        
        if i >= 2:  # Show first 3 examples
            break

except Exception as e:
    print(f"Could not show examples: {e}")

print(f"\n✅ ADDRESS FIELD RESTRUCTURING COMPLETE")
print(f"🎯 Simpler structure with corruption tracking")
print(f"📊 Address components derived from single corrupted address per record")
