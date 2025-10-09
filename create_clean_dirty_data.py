#!/usr/bin/env python3
"""
Script to create a "dirty" version of the comprehensive test data with formatting issues,
then combine clean and dirty data for testing PII detection robustness.
"""

import pandas as pd
import csv
import random
import re

def corrupt_address(address):
    """Introduce formatting issues in addresses"""
    corruptions = [
        lambda addr: addr.replace(',', ''),  # Remove commas
        lambda addr: addr.replace(', ', ' '),  # Remove comma-space combinations
        lambda addr: addr.replace(' ', ''),  # Remove all spaces
        lambda addr: addr.lower(),  # Make lowercase
        lambda addr: addr.upper(),  # Make uppercase
        lambda addr: addr.replace(' UK', ''),  # Remove country
        lambda addr: addr.replace(', United Kingdom', ''),  # Remove full country
        lambda addr: ' '.join(addr.split()[::-1]),  # Reverse word order
        lambda addr: addr.replace('St', 'Street').replace('Rd', 'Road').replace('Ave', 'Avenue'),  # Expand abbreviations
        lambda addr: addr.replace('Street', 'St').replace('Road', 'Rd').replace('Avenue', 'Ave'),  # Contract words
    ]
    
    corruption_type = random.choice([
        "no_commas", "no_comma_space", "no_spaces", "lowercase", "uppercase", 
        "no_uk", "no_country", "reversed", "expanded_abbrev", "contracted"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(address)
    
    return corrupted, corruption_type

def corrupt_postcode(address):
    """Introduce formatting issues in postcodes within addresses"""
    # Extract UK postcode pattern and corrupt it
    postcode_pattern = r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b'
    postcodes = re.findall(postcode_pattern, address)
    
    if not postcodes:
        return address, "no_postcode_found"
    
    postcode = postcodes[0]
    corruptions = [
        lambda pc: pc.replace(' ', ''),  # Remove spaces: "M1 1AA" -> "M11AA"
        lambda pc: pc.lower(),  # Lowercase: "M1 1AA" -> "m1 1aa"
        lambda pc: ' '.join(list(pc.replace(' ', ''))),  # Add spaces between all: "M1 1AA" -> "M 1 1 A A"
        lambda pc: pc.replace(' ', '  '),  # Double spaces: "M1 1AA" -> "M1  1AA"
        lambda pc: pc[:3] + ' ' + pc[3:],  # Wrong spacing: "M1 1AA" -> "M1  1AA"
        lambda pc: pc.replace(' ', '-'),  # Hyphen instead of space: "M1 1AA" -> "M1-1AA"
        lambda pc: pc + ' UK',  # Add country code
        lambda pc: 'UK ' + pc,  # Prefix with country
    ]
    
    corruption_type = random.choice([
        "no_space", "lowercase", "char_spaced", "double_space", 
        "wrong_space", "hyphenated", "suffix_uk", "prefix_uk"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted_postcode = corruption_func(postcode)
    corrupted_address = address.replace(postcode, corrupted_postcode)
    
    return corrupted_address, f"postcode_{corruption_type}"

def corrupt_email(email):
    """Introduce formatting issues in email addresses"""
    corruptions = [
        lambda em: em.replace('@', ' @ '),  # Add spaces around @: "user@domain.com" -> "user @ domain.com"
        lambda em: em.replace('.', ','),  # Comma instead of dot: "user@domain.com" -> "user@domain,com"
        lambda em: em.replace('.', ' . '),  # Spaces around dots: "user@domain.com" -> "user@domain . com"
        lambda em: em.upper(),  # Uppercase: "user@domain.com" -> "USER@DOMAIN.COM"
        lambda em: em.replace('@', ' at '),  # Text instead of @: "user@domain.com" -> "user at domain.com"
        lambda em: ' ' + em + ' ',  # Add leading/trailing spaces
        lambda em: em.replace('@', '@@'),  # Double @: "user@domain.com" -> "user@@domain.com"
        lambda em: em.split('@')[0] + ' @ ' + em.split('@')[1],  # Spaces around @
        lambda em: em.replace('-', ' '),  # Replace hyphens with spaces in domain
        lambda em: em.replace('.com', ' com').replace('.co.uk', ' co uk').replace('.org', ' org').replace('.net', ' net'),  # Separate TLD
    ]
    
    corruption_type = random.choice([
        "spaced_at", "comma_dot", "spaced_dots", "uppercase", "text_at",
        "padded_spaces", "double_at", "at_spaces", "no_hyphens", "separated_tld"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(email)
    
    return corrupted, corruption_type

def corrupt_phone(phone):
    """Introduce formatting issues in phone numbers"""
    corruptions = [
        lambda ph: ph.replace(' ', ''),  # Remove all spaces: "+44 7123456789" -> "+447123456789"
        lambda ph: ph.replace('+44', '0044'),  # Replace + with 00: "+44 7123456789" -> "0044 7123456789"
        lambda ph: ph.replace('+', ''),  # Remove +: "+44 7123456789" -> "44 7123456789"
        lambda ph: ' '.join(list(ph.replace(' ', ''))),  # Space between every digit
        lambda ph: ph.replace(' ', '-'),  # Hyphens instead of spaces: "+44 7123456789" -> "+44-7123456789"
        lambda ph: ph.replace(' ', '.'),  # Dots instead of spaces: "+44 7123456789" -> "+44.7123456789"
        lambda ph: ph + ' UK',  # Add country suffix
        lambda ph: 'UK ' + ph,  # Add country prefix
        lambda ph: ph.replace('+44 ', ''),  # Remove country code: "+44 7123456789" -> "7123456789"
        lambda ph: ph.replace(' ', '  '),  # Double spaces
    ]
    
    corruption_type = random.choice([
        "no_spaces", "double_zero", "no_plus", "digit_spaced", "hyphenated",
        "dotted", "suffix_uk", "prefix_uk", "no_country", "double_spaces"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(phone)
    
    return corrupted, corruption_type

def corrupt_nino(nino):
    """Introduce formatting issues in National Insurance numbers"""
    corruptions = [
        lambda ni: ni.replace(' ', ''),  # Remove all spaces: "AB 12 34 56 C" -> "AB123456C"
        lambda ni: ni.replace(' ', '-'),  # Hyphens instead of spaces: "AB 12 34 56 C" -> "AB-12-34-56-C"
        lambda ni: ni.lower(),  # Lowercase: "AB 12 34 56 C" -> "ab 12 34 56 c"
        lambda ni: ' '.join(list(ni.replace(' ', ''))),  # Space between every char
        lambda ni: ni.replace(' ', '  '),  # Double spaces
        lambda ni: ni.replace(' ', '.'),  # Dots instead of spaces: "AB 12 34 56 C" -> "AB.12.34.56.C"
        lambda ni: ni + ' UK',  # Add country suffix
        lambda ni: 'NI: ' + ni,  # Add prefix
        lambda ni: ni.replace(' ', '/'),  # Slashes: "AB 12 34 56 C" -> "AB/12/34/56/C"
        lambda ni: ni[:2] + ' ' + ''.join(ni[3:].split()),  # Compress middle: "AB 12 34 56 C" -> "AB 123456C"
    ]
    
    corruption_type = random.choice([
        "no_spaces", "hyphenated", "lowercase", "char_spaced", "double_spaces",
        "dotted", "suffix_uk", "prefixed", "slashed", "compressed"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(nino)
    
    return corrupted, corruption_type

def corrupt_pan(pan):
    """Introduce formatting issues in PAN numbers"""
    # Convert to string if it's a number
    pan = str(pan)
    
    corruptions = [
        lambda p: ' '.join([p[i:i+4] for i in range(0, len(p), 4)]),  # Add spaces every 4 digits
        lambda p: '-'.join([p[i:i+4] for i in range(0, len(p), 4)]),  # Add hyphens every 4 digits
        lambda p: ' '.join(list(p)),  # Space between every digit
        lambda p: p[:4] + ' ' + p[4:8] + ' ' + p[8:12] + ' ' + p[12:],  # Credit card format
        lambda p: p.lower() if any(c.isalpha() for c in p) else p,  # Lowercase if contains letters
        lambda p: p + ' VISA' if p.startswith('4') else p + ' AMEX' if p.startswith('34') else p + ' MASTER',  # Add card type
        lambda p: 'CARD: ' + p,  # Add prefix
        lambda p: p + ' UK',  # Add country
        lambda p: p[:8] + ' ' + p[8:],  # Split in middle
        lambda p: '.'.join([p[i:i+4] for i in range(0, len(p), 4)]),  # Dots every 4 digits
    ]
    
    corruption_type = random.choice([
        "spaced_groups", "hyphen_groups", "digit_spaced", "cc_format", "lowercase",
        "card_suffix", "prefixed", "suffix_uk", "middle_split", "dotted_groups"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(pan)
    
    return corrupted, corruption_type

def corrupt_sort_code(sort_code):
    """Introduce formatting issues in sort codes"""
    corruptions = [
        lambda sc: sc.replace('-', ''),  # Remove hyphens: "12-34-56" -> "123456"
        lambda sc: sc.replace('-', ' '),  # Spaces instead of hyphens: "12-34-56" -> "12 34 56"
        lambda sc: sc.replace('-', '.'),  # Dots instead of hyphens: "12-34-56" -> "12.34.56"
        lambda sc: sc.replace('-', '/'),  # Slashes: "12-34-56" -> "12/34/56"
        lambda sc: ' '.join(list(sc.replace('-', ''))),  # Space every digit
        lambda sc: sc + ' UK',  # Add country
        lambda sc: 'SC: ' + sc,  # Add prefix
        lambda sc: sc.replace('-', '--'),  # Double hyphens
        lambda sc: sc[:2] + sc[3:5] + sc[6:],  # Remove all separators differently
        lambda sc: sc.lower(),  # Lowercase (shouldn't matter for numbers but for consistency)
    ]
    
    corruption_type = random.choice([
        "no_hyphens", "spaced", "dotted", "slashed", "digit_spaced",
        "suffix_uk", "prefixed", "double_hyphens", "no_separators", "lowercase"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(sort_code)
    
    return corrupted, corruption_type

def corrupt_iban(iban):
    """Introduce formatting issues in IBAN"""
    corruptions = [
        lambda ib: ' '.join([ib[i:i+4] for i in range(0, len(ib), 4)]),  # Add spaces every 4 chars
        lambda ib: ib.lower(),  # Lowercase
        lambda ib: ib[:2] + ' ' + ib[2:],  # Space after country code
        lambda ib: ib + ' UK',  # Add country suffix
        lambda ib: 'IBAN: ' + ib,  # Add prefix
        lambda ib: '-'.join([ib[i:i+4] for i in range(0, len(ib), 4)]),  # Hyphens every 4
        lambda ib: '.'.join([ib[i:i+4] for i in range(0, len(ib), 4)]),  # Dots every 4
        lambda ib: ' ' + ib + ' ',  # Add padding spaces
        lambda ib: ib.replace('GB', 'UK'),  # Change country code
        lambda ib: ib[2:] + ib[:2],  # Move country code to end
    ]
    
    corruption_type = random.choice([
        "spaced_groups", "lowercase", "country_spaced", "suffix_uk", "prefixed",
        "hyphen_groups", "dot_groups", "padded", "changed_country", "moved_country"
    ])
    
    corruption_func = random.choice(corruptions)
    corrupted = corruption_func(iban)
    
    return corrupted, corruption_type

def create_dirty_data():
    """Create dirty version of the comprehensive test data"""
    
    print("🔄 Loading clean test data...")
    
    # Load the clean data
    clean_file = '/Users/davidlock/Downloads/soccer data python/testing poe/comprehensive_test_data.csv'
    clean_df = pd.read_csv(clean_file)
    
    print(f"📊 Loaded {len(clean_df)} clean records")
    print("🔧 Creating dirty versions with formatting issues...")
    
    dirty_data = []
    corruption_stats = {}
    
    for index, row in clean_df.iterrows():
        # Randomly decide which fields to corrupt (corrupt 1-3 fields per record)
        fields_to_corrupt = random.sample(['address', 'email', 'mobile_phone', 'national_insurance', 'pan_number', 'sort_code', 'iban'], random.randint(1, 3))
        
        corrupted_record = row.to_dict()
        corruptions_applied = []
        
        for field in fields_to_corrupt:
            if field == 'address':
                # Decide whether to corrupt address or postcode within address
                if random.choice([True, False]):
                    corrupted_record[field], corruption_type = corrupt_address(str(row[field]))
                    corruptions_applied.append(f"address_{corruption_type}")
                else:
                    corrupted_record[field], corruption_type = corrupt_postcode(str(row[field]))
                    corruptions_applied.append(f"address_{corruption_type}")
            
            elif field == 'email':
                corrupted_record[field], corruption_type = corrupt_email(str(row[field]))
                corruptions_applied.append(f"email_{corruption_type}")
            
            elif field == 'mobile_phone':
                corrupted_record[field], corruption_type = corrupt_phone(str(row[field]))
                corruptions_applied.append(f"phone_{corruption_type}")
            
            elif field == 'national_insurance':
                corrupted_record[field], corruption_type = corrupt_nino(str(row[field]))
                corruptions_applied.append(f"nino_{corruption_type}")
            
            elif field == 'pan_number':
                corrupted_record[field], corruption_type = corrupt_pan(str(row[field]))
                corruptions_applied.append(f"pan_{corruption_type}")
            
            elif field == 'sort_code':
                corrupted_record[field], corruption_type = corrupt_sort_code(str(row[field]))
                corruptions_applied.append(f"sort_{corruption_type}")
            
            elif field == 'iban':
                corrupted_record[field], corruption_type = corrupt_iban(str(row[field]))
                corruptions_applied.append(f"iban_{corruption_type}")
        
        # Add corruption tracking
        corrupted_record['data_quality'] = 'dirty'
        corrupted_record['corruptions_applied'] = '; '.join(corruptions_applied)
        
        # Update corruption statistics
        for corruption in corruptions_applied:
            corruption_stats[corruption] = corruption_stats.get(corruption, 0) + 1
        
        dirty_data.append(corrupted_record)
        
        # Progress indicator
        if (index + 1) % 100 == 0:
            print(f"   Corrupted {index + 1} records...")
    
    return dirty_data, corruption_stats

def combine_clean_and_dirty_data(dirty_data):
    """Combine clean and dirty data into final dataset"""
    
    print("🔄 Combining clean and dirty data...")
    
    # Load clean data again and add quality markers
    clean_file = '/Users/davidlock/Downloads/soccer data python/testing poe/comprehensive_test_data.csv'
    clean_df = pd.read_csv(clean_file)
    
    # Add quality tracking to clean data
    clean_data = []
    for index, row in clean_df.iterrows():
        clean_record = row.to_dict()
        clean_record['data_quality'] = 'clean'
        clean_record['corruptions_applied'] = 'none'
        clean_data.append(clean_record)
    
    # Combine datasets
    combined_data = clean_data + dirty_data
    
    # Shuffle the combined data
    random.shuffle(combined_data)
    
    return combined_data

def save_combined_data(combined_data):
    """Save combined clean and dirty data"""
    
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/clean_and_dirty_test_data.csv'
    
    fieldnames = [
        'record_id', 'forename', 'surname', 'full_name', 'email', 'address',
        'mobile_phone', 'national_insurance', 'pan_number', 'sort_code', 
        'account_number', 'iban', 'data_quality', 'corruptions_applied'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in combined_data:
            writer.writerow(record)
    
    print(f"✅ Combined data saved to: {output_file}")
    return output_file

def analyze_corruptions(corruption_stats):
    """Analyze corruption statistics"""
    
    print(f"\n📊 CORRUPTION ANALYSIS:")
    print("=" * 50)
    
    total_corruptions = sum(corruption_stats.values())
    print(f"Total corruptions applied: {total_corruptions}")
    
    # Group by field type
    field_groups = {}
    for corruption, count in corruption_stats.items():
        field = corruption.split('_')[0]
        field_groups[field] = field_groups.get(field, 0) + count
    
    print(f"\nCorruptions by field type:")
    for field, count in sorted(field_groups.items()):
        percentage = count / total_corruptions * 100
        print(f"   {field}: {count} ({percentage:.1f}%)")
    
    # Top 10 specific corruptions
    print(f"\nTop 10 specific corruptions:")
    sorted_corruptions = sorted(corruption_stats.items(), key=lambda x: x[1], reverse=True)
    for corruption, count in sorted_corruptions[:10]:
        percentage = count / total_corruptions * 100
        print(f"   {corruption}: {count} ({percentage:.1f}%)")

def show_samples(combined_data):
    """Show sample clean and dirty records"""
    
    print(f"\n📋 SAMPLE RECORDS:")
    print("=" * 80)
    
    # Find clean and dirty examples
    clean_samples = [r for r in combined_data if r['data_quality'] == 'clean'][:2]
    dirty_samples = [r for r in combined_data if r['data_quality'] == 'dirty'][:3]
    
    print("🟢 CLEAN RECORDS:")
    for i, record in enumerate(clean_samples, 1):
        print(f"\nClean Record {i}:")
        print(f"   Name: {record['full_name']}")
        print(f"   Email: {record['email']}")
        print(f"   Address: {record['address']}")
        print(f"   Mobile: {record['mobile_phone']}")
        print(f"   NI: {record['national_insurance']}")
        print(f"   PAN: {record['pan_number']}")
    
    print(f"\n🔴 DIRTY RECORDS:")
    for i, record in enumerate(dirty_samples, 1):
        print(f"\nDirty Record {i} (Corruptions: {record['corruptions_applied']}):")
        print(f"   Name: {record['full_name']}")
        print(f"   Email: {record['email']}")
        print(f"   Address: {record['address']}")
        print(f"   Mobile: {record['mobile_phone']}")
        print(f"   NI: {record['national_insurance']}")
        print(f"   PAN: {record['pan_number']}")

if __name__ == "__main__":
    print("🏗️  CREATING CLEAN AND DIRTY TEST DATA")
    print("=" * 60)
    
    # Create dirty data
    dirty_data, corruption_stats = create_dirty_data()
    
    # Combine with clean data
    combined_data = combine_clean_and_dirty_data(dirty_data)
    
    # Save combined data
    output_file = save_combined_data(combined_data)
    
    # Analyze corruptions
    analyze_corruptions(corruption_stats)
    
    # Show samples
    show_samples(combined_data)
    
    print(f"\n✅ CLEAN AND DIRTY DATA CREATED")
    print("=" * 60)
    print(f"📊 Total records: {len(combined_data):,}")
    print(f"   Clean records: {len([r for r in combined_data if r['data_quality'] == 'clean'])}")
    print(f"   Dirty records: {len([r for r in combined_data if r['data_quality'] == 'dirty'])}")
    print(f"💾 Output file: clean_and_dirty_test_data.csv")
    print(f"\n🎯 Perfect for testing PII detection robustness!")
    print(f"   • Tests how well systems handle formatting variations")
    print(f"   • Includes tracking of which corruptions were applied")
    print(f"   • Maintains original clean data for comparison")
