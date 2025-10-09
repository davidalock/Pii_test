#!/usr/bin/env python3
"""
Script to split address field into separate components:
- street_address (house number, street name)
- city_town (city or town name)
- postcode (UK postal code)
"""

import pandas as pd
import re
import csv

def extract_uk_postcode(address):
    """Extract UK postcode from address"""
    # UK postcode patterns (handle various formats including corrupted ones)
    patterns = [
        r'\b[A-Z]{1,2}\s?[0-9][A-Z0-9]?\s+[0-9][A-Z]{2}\b',  # Standard: "M1 1AA"
        r'\b[A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}\b',        # No space: "M11AA"
        r'\b[a-z]{1,2}\s?[0-9][a-z0-9]?\s+[0-9][a-z]{2}\b',  # Lowercase: "m1 1aa"
        r'\b[a-z]{1,2}[0-9][a-z0-9]?[0-9][a-z]{2}\b',        # Lowercase no space: "m11aa"
        r'\b[A-Z]\s[0-9]\s[0-9][A-Z]{2}\b',                   # Spaced: "M 1 1AA"
        r'\b[A-Z]{1,2}[0-9][A-Z0-9]?-[0-9][A-Z]{2}\b',       # Hyphenated: "M1-1AA"
        r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s{2,}[0-9][A-Z]{2}\b',  # Multiple spaces: "M1  1AA"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, address)
        if match:
            postcode = match.group().strip()
            # Clean up the address by removing the postcode
            cleaned_address = address.replace(match.group(), '').strip()
            # Remove trailing comma if present
            cleaned_address = re.sub(r',\s*$', '', cleaned_address)
            return postcode, cleaned_address
    
    # If no postcode found, return empty postcode and original address
    return '', address

def extract_city_town(address_without_postcode):
    """Extract city/town from address (usually the last component before postcode)"""
    if not address_without_postcode:
        return '', ''
    
    # Remove "UK" or "United Kingdom" if present
    address = re.sub(r',?\s*(UK|United Kingdom)\s*$', '', address_without_postcode).strip()
    address = re.sub(r',?\s*$', '', address).strip()  # Remove trailing comma
    
    # Split by comma and take the last meaningful part as city/town
    parts = [part.strip() for part in address.split(',') if part.strip()]
    
    if not parts:
        return '', address_without_postcode
    
    if len(parts) == 1:
        # If only one part, it's likely just a street address
        return '', address_without_postcode
    
    # The last part is usually the city/town
    city_town = parts[-1]
    
    # The rest is the street address
    street_address = ', '.join(parts[:-1]) if len(parts) > 1 else ''
    
    return city_town, street_address

def split_addresses(df):
    """Split addresses in the dataframe into components"""
    
    print("🔍 Splitting addresses into components...")
    
    # Add new columns
    df['postcode'] = ''
    df['city_town'] = ''
    df['street_address'] = ''
    
    successful_splits = 0
    postcode_found = 0
    city_found = 0
    
    for index, row in df.iterrows():
        original_address = str(row['address'])
        
        # Extract postcode first
        postcode, address_without_postcode = extract_uk_postcode(original_address)
        
        # Extract city/town and street address
        city_town, street_address = extract_city_town(address_without_postcode)
        
        # Update the dataframe
        df.at[index, 'postcode'] = postcode
        df.at[index, 'city_town'] = city_town
        df.at[index, 'street_address'] = street_address
        
        # Count successful extractions
        if postcode:
            postcode_found += 1
        if city_town:
            city_found += 1
        if postcode or city_town:
            successful_splits += 1
        
        # Progress indicator
        if (index + 1) % 200 == 0:
            print(f"   Processed {index + 1:,} records...")
    
    print(f"✅ Address splitting complete:")
    print(f"   Postcodes extracted: {postcode_found:,} ({postcode_found/len(df)*100:.1f}%)")
    print(f"   Cities/towns extracted: {city_found:,} ({city_found/len(df)*100:.1f}%)")
    print(f"   Successful splits: {successful_splits:,} ({successful_splits/len(df)*100:.1f}%)")
    
    return df

def analyze_address_components(df):
    """Analyze the extracted address components"""
    
    print(f"\n📊 ADDRESS COMPONENT ANALYSIS:")
    print("=" * 50)
    
    # Postcode analysis
    postcodes_with_content = df[df['postcode'] != '']['postcode']
    print(f"Postcode Statistics:")
    print(f"   Total postcodes extracted: {len(postcodes_with_content):,}")
    
    if len(postcodes_with_content) > 0:
        # Show postcode format distribution
        format_counts = {}
        for pc in postcodes_with_content:
            if ' ' in pc:
                format_counts['spaced'] = format_counts.get('spaced', 0) + 1
            elif pc.islower():
                format_counts['lowercase'] = format_counts.get('lowercase', 0) + 1
            elif '-' in pc:
                format_counts['hyphenated'] = format_counts.get('hyphenated', 0) + 1
            else:
                format_counts['standard'] = format_counts.get('standard', 0) + 1
        
        print(f"   Postcode formats:")
        for fmt, count in format_counts.items():
            print(f"      {fmt}: {count} ({count/len(postcodes_with_content)*100:.1f}%)")
    
    # City/Town analysis
    cities_with_content = df[df['city_town'] != '']['city_town']
    print(f"\nCity/Town Statistics:")
    print(f"   Total cities/towns extracted: {len(cities_with_content):,}")
    
    if len(cities_with_content) > 0:
        # Show top cities/towns
        city_counts = cities_with_content.value_counts()
        print(f"   Top 10 cities/towns:")
        for city, count in city_counts.head(10).items():
            print(f"      {city}: {count}")
    
    # Street address analysis
    streets_with_content = df[df['street_address'] != '']['street_address']
    print(f"\nStreet Address Statistics:")
    print(f"   Total street addresses extracted: {len(streets_with_content):,}")
    
    # Data quality by clean vs dirty
    if 'data_quality' in df.columns:
        clean_df = df[df['data_quality'] == 'clean']
        dirty_df = df[df['data_quality'] == 'dirty']
        
        print(f"\nExtraction Success by Data Quality:")
        print(f"   Clean data:")
        clean_pc_success = len(clean_df[clean_df['postcode'] != '']) / len(clean_df) * 100
        clean_city_success = len(clean_df[clean_df['city_town'] != '']) / len(clean_df) * 100
        print(f"      Postcode extraction: {clean_pc_success:.1f}%")
        print(f"      City/town extraction: {clean_city_success:.1f}%")
        
        print(f"   Dirty data:")
        dirty_pc_success = len(dirty_df[dirty_df['postcode'] != '']) / len(dirty_df) * 100
        dirty_city_success = len(dirty_df[dirty_df['city_town'] != '']) / len(dirty_df) * 100
        print(f"      Postcode extraction: {dirty_pc_success:.1f}%")
        print(f"      City/town extraction: {dirty_city_success:.1f}%")

def show_address_examples(df):
    """Show examples of address splitting"""
    
    print(f"\n📋 ADDRESS SPLITTING EXAMPLES:")
    print("=" * 80)
    
    # Show successful splits
    successful_examples = df[(df['postcode'] != '') & (df['city_town'] != '')].head(5)
    
    print("🟢 SUCCESSFUL SPLITS:")
    for i, (_, row) in enumerate(successful_examples.iterrows(), 1):
        print(f"\nExample {i}:")
        print(f"   Original: {row['address']}")
        print(f"   Street: {row['street_address']}")
        print(f"   City/Town: {row['city_town']}")
        print(f"   Postcode: {row['postcode']}")
        if 'data_quality' in row:
            print(f"   Quality: {row['data_quality']}")
    
    # Show challenging cases
    challenging_examples = df[df['postcode'] == ''].head(3)
    if len(challenging_examples) > 0:
        print(f"\n🔴 CHALLENGING CASES (no postcode extracted):")
        for i, (_, row) in enumerate(challenging_examples.iterrows(), 1):
            print(f"\nChallenge {i}:")
            print(f"   Original: {row['address']}")
            print(f"   Street: {row['street_address']}")
            print(f"   City/Town: {row['city_town']}")
            if 'data_quality' in row:
                print(f"   Quality: {row['data_quality']}")

def save_split_addresses(df):
    """Save the dataframe with split addresses"""
    
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/split_address_data.csv'
    
    # Reorder columns to put address components together
    if 'data_quality' in df.columns:
        column_order = [
            'record_id', 'forename', 'surname', 'full_name', 'email', 
            'address', 'street_address', 'city_town', 'postcode',
            'mobile_phone', 'national_insurance', 'pan_number', 'sort_code', 
            'account_number', 'iban', 'data_quality', 'corruptions_applied'
        ]
    else:
        column_order = [
            'record_id', 'forename', 'surname', 'full_name', 'email', 
            'address', 'street_address', 'city_town', 'postcode',
            'mobile_phone', 'national_insurance', 'pan_number', 'sort_code', 
            'account_number', 'iban'
        ]
    
    # Filter to only existing columns
    existing_columns = [col for col in column_order if col in df.columns]
    df_reordered = df[existing_columns]
    
    df_reordered.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Split address data saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    print("🏗️  SPLITTING ADDRESS COMPONENTS")
    print("=" * 60)
    
    # Load the clean and dirty data
    input_file = '/Users/davidlock/Downloads/soccer data python/testing poe/clean_and_dirty_test_data.csv'
    print(f"📂 Loading data from: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df):,} records")
    
    # Split addresses
    df_split = split_addresses(df)
    
    # Analyze results
    analyze_address_components(df_split)
    
    # Show examples
    show_address_examples(df_split)
    
    # Save results
    output_file = save_split_addresses(df_split)
    
    print(f"\n✅ ADDRESS SPLITTING COMPLETE")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   Input records: {len(df):,}")
    print(f"   Output file: split_address_data.csv")
    print(f"   New fields added: street_address, city_town, postcode")
    print(f"\n🎯 Benefits:")
    print(f"   • More granular address testing")
    print(f"   • Separate PII detection for address components")
    print(f"   • Better analysis of geographic data")
    print(f"   • Improved data structure for ML models")
