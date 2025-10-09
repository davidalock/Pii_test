#!/usr/bin/env python3
"""
Create dirty version of full merchant PII data with expanded corruption types
"""

import pandas as pd
import random
import string
import re
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

def corrupt_name(name):
    """Apply various corruptions to names"""
    if not name or pd.isna(name):
        return name
        
    corruptions = [
        # Case variations
        lambda x: x.upper(),
        lambda x: x.lower(),
        lambda x: x.title() if not x.istitle() else x,
        
        # Extra spaces
        lambda x: f" {x} ",
        lambda x: f"  {x}",
        lambda x: f"{x}  ",
        lambda x: x.replace(" ", "  "),
        
        # Character corruptions
        lambda x: x.replace("a", "@") if "a" in x else x,
        lambda x: x.replace("e", "3") if "e" in x else x,
        lambda x: x.replace("i", "1") if "i" in x else x,
        lambda x: x.replace("o", "0") if "o" in x else x,
        
        # Missing characters
        lambda x: x[1:] if len(x) > 1 else x,
        lambda x: x[:-1] if len(x) > 1 else x,
        
        # Extra characters
        lambda x: f"{x}{random.choice(string.ascii_letters)}",
        lambda x: f"{random.choice(string.ascii_letters)}{x}",
        
        # Hyphenation
        lambda x: x.replace(" ", "-") if " " in x else x,
        
        # Apostrophes
        lambda x: x.replace("'", "") if "'" in x else x,
        lambda x: x.replace("'", "'") if "'" in x else x,
        
        # No corruption (keep some clean)
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(name)

def corrupt_email(email):
    """Apply various corruptions to email addresses"""
    if not email or pd.isna(email):
        return email
        
    corruptions = [
        # Case variations
        lambda x: x.upper(),
        lambda x: x.lower(),
        
        # Extra spaces
        lambda x: f" {x} ",
        lambda x: x.replace("@", " @ "),
        lambda x: x.replace(".", " . "),
        
        # Missing @ or .
        lambda x: x.replace("@", "") if "@" in x else x,
        lambda x: x.replace(".", "") if x.count(".") > 1 else x,
        
        # Multiple @
        lambda x: x.replace("@", "@@", 1) if "@" in x else x,
        
        # Character substitutions
        lambda x: x.replace("@", "at") if "@" in x else x,
        lambda x: x.replace(".", "dot") if "." in x else x,
        
        # Extra characters
        lambda x: f"{x}{random.choice(string.digits)}",
        
        # Domain corruptions
        lambda x: x.replace(".com", ".co") if ".com" in x else x,
        lambda x: x.replace(".co.uk", ".uk") if ".co.uk" in x else x,
        
        # No corruption (keep many clean for realistic detection)
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(email)

def corrupt_address(address):
    """Apply various corruptions to addresses"""
    if not address or pd.isna(address):
        return address
        
    corruptions = [
        # Case variations
        lambda x: x.upper(),
        lambda x: x.lower(),
        lambda x: x.title(),
        
        # Extra spaces and punctuation
        lambda x: f" {x} ",
        lambda x: x.replace(",", " , "),
        lambda x: x.replace(",", ""),
        lambda x: x.replace("  ", " "),
        
        # Number corruptions
        lambda x: re.sub(r'\b(\d+)\b', lambda m: str(int(m.group(1)) + random.randint(-5, 5)) if int(m.group(1)) > 5 else m.group(1), x),
        
        # Abbreviations
        lambda x: x.replace("Street", "St") if "Street" in x else x,
        lambda x: x.replace("Road", "Rd") if "Road" in x else x,
        lambda x: x.replace("Avenue", "Ave") if "Avenue" in x else x,
        lambda x: x.replace("St ", "Street ") if " St " in x else x,
        lambda x: x.replace("Rd ", "Road ") if " Rd " in x else x,
        
        # Missing parts
        lambda x: x.rsplit(",", 1)[0] if x.count(",") > 1 else x,
        lambda x: x.split(",", 1)[1] if "," in x else x,
        
        # Extra country indicators
        lambda x: f"{x}, United Kingdom" if not x.endswith("Kingdom") else x,
        lambda x: f"{x}, England" if "UK" in x else x,
        
        # Postcode variations
        lambda x: re.sub(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?) (\d[A-Z]{2})\b', r'\1\2', x),  # Remove postcode space
        lambda x: re.sub(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)(\d[A-Z]{2})\b', r'\1 \2', x),  # Add postcode space
        
        # No corruption (keep some clean)
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(address)

def corrupt_phone(phone):
    """Apply various corruptions to phone numbers"""
    if not phone or pd.isna(phone):
        return phone
        
    corruptions = [
        # Space variations
        lambda x: x.replace(" ", ""),
        lambda x: x.replace(" ", "-"),
        lambda x: x.replace(" ", "."),
        lambda x: x.replace("-", " "),
        
        # Bracket variations
        lambda x: x.replace("(", "").replace(")", ""),
        lambda x: f"({x[:3]}) {x[3:]}" if len(x.replace(" ", "")) > 3 else x,
        
        # Country code variations
        lambda x: f"+44 {x[1:]}" if x.startswith("0") else x,
        lambda x: f"0{x[4:]}" if x.startswith("+44 ") else x,
        lambda x: f"44 {x[1:]}" if x.startswith("0") else x,
        
        # Extra characters
        lambda x: f"{x} ext.123",
        lambda x: f"{x}x456",
        
        # Digit corruptions
        lambda x: x[:-1] + str(random.randint(0, 9)) if len(x) > 1 else x,
        lambda x: str(random.randint(0, 9)) + x[1:] if len(x) > 1 else x,
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(phone)

def corrupt_date(date_str):
    """Apply various corruptions to dates"""
    if not date_str or pd.isna(date_str):
        return date_str
        
    corruptions = [
        # Separator variations
        lambda x: x.replace("/", "-"),
        lambda x: x.replace("-", "/"),
        lambda x: x.replace("/", "."),
        lambda x: x.replace("-", "."),
        
        # Format variations
        lambda x: x.replace("/", "") if "/" in x else x,
        lambda x: x.replace("-", "") if "-" in x else x,
        
        # Add extra spaces
        lambda x: x.replace("/", " / "),
        lambda x: x.replace("-", " - "),
        
        # Year variations (2-digit vs 4-digit)
        lambda x: re.sub(r'\b(19|20)(\d{2})\b', r'\2', x),  # 4-digit to 2-digit
        lambda x: re.sub(r'\b(\d{2})(?=\D|$)', lambda m: f"19{m.group(1)}" if int(m.group(1)) > 25 else f"20{m.group(1)}", x),  # 2-digit to 4-digit
        
        # Month name corruptions
        lambda x: x.replace("January", "Jan").replace("February", "Feb").replace("March", "Mar"),
        lambda x: x.replace("Jan", "January").replace("Feb", "February").replace("Mar", "March"),
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(date_str)

def corrupt_nino(nino):
    """Apply corruptions to NINO"""
    if not nino or pd.isna(nino):
        return nino
        
    corruptions = [
        # Space variations
        lambda x: f"{x[:2]} {x[2:8]} {x[8:]}" if len(x) == 9 else x,
        lambda x: x.replace(" ", ""),
        
        # Case variations
        lambda x: x.lower(),
        lambda x: x.upper(),
        
        # Character substitutions
        lambda x: x.replace("O", "0") if "O" in x else x,
        lambda x: x.replace("0", "O") if "0" in x else x,
        lambda x: x.replace("I", "1") if "I" in x else x,
        lambda x: x.replace("1", "I") if "1" in x else x,
        
        # Missing characters
        lambda x: x[:-1] if len(x) > 1 else x,
        
        # Extra characters
        lambda x: f"{x}{random.choice(string.ascii_uppercase)}",
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(nino)

def corrupt_pan(pan):
    """Apply corruptions to PAN numbers"""
    if not pan or pd.isna(pan):
        return pan
        
    # Convert to string if it's not already
    pan_str = str(pan)
    
    corruptions = [
        # Space variations
        lambda x: x.replace(" ", ""),
        lambda x: x.replace(" ", "-"),
        lambda x: x.replace(" ", "."),
        
        # Digit corruptions
        lambda x: x[:-1] + str(random.randint(0, 9)) if len(x.replace(" ", "")) > 1 else x,
        lambda x: str(random.randint(0, 9)) + x[1:] if len(x.replace(" ", "")) > 1 else x,
        
        # Extra characters
        lambda x: f"{x}*",
        lambda x: f"*{x}",
        
        # Missing digits
        lambda x: x[1:] if len(x.replace(" ", "")) > 4 else x,
        lambda x: x[:-1] if len(x.replace(" ", "")) > 4 else x,
        
        # Format changes
        lambda x: f"xxxx-xxxx-xxxx-{x.replace(' ', '')[-4:]}" if len(x.replace(" ", "")) >= 4 else x,
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(pan_str)

def corrupt_sort_code(sort_code):
    """Apply corruptions to sort codes"""
    if not sort_code or pd.isna(sort_code):
        return sort_code
        
    corruptions = [
        # Separator variations
        lambda x: x.replace("-", ""),
        lambda x: x.replace("-", " "),
        lambda x: x.replace("-", "."),
        lambda x: x.replace("-", ":"),
        
        # Extra spaces
        lambda x: x.replace("-", " - "),
        
        # Digit corruptions
        lambda x: x[:-1] + str(random.randint(0, 9)) if len(x) > 1 else x,
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(sort_code)

def corrupt_account_number(account_num):
    """Apply corruptions to account numbers"""
    if not account_num or pd.isna(account_num):
        return account_num
        
    # Convert to string if it's not already
    account_str = str(account_num)
    
    corruptions = [
        # Add spaces or separators
        lambda x: f"{x[:2]} {x[2:4]} {x[4:6]} {x[6:]}" if len(x) >= 6 else x,
        lambda x: f"{x[:4]}-{x[4:]}" if len(x) > 4 else x,
        
        # Leading zeros issues
        lambda x: x.lstrip("0") if x.startswith("0") and len(x) > 1 else x,
        lambda x: f"0{x}" if not x.startswith("0") and len(x) < 8 else x,
        
        # Digit corruption
        lambda x: x[:-1] + str(random.randint(0, 9)) if len(x) > 1 else x,
        
        # Extra characters
        lambda x: f"{x}*",
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(account_str)

def corrupt_iban(iban):
    """Apply corruptions to IBAN"""
    if not iban or pd.isna(iban):
        return iban
        
    corruptions = [
        # Space variations
        lambda x: f"{x[:4]} {x[4:8]} {x[8:12]} {x[12:16]} {x[16:20]} {x[20:]}" if len(x) >= 20 else x,
        lambda x: x.replace(" ", ""),
        
        # Case variations
        lambda x: x.lower(),
        lambda x: x.upper() if not x.isupper() else x,
        
        # Character substitutions
        lambda x: x.replace("O", "0") if "O" in x else x,
        lambda x: x.replace("0", "O") if "0" in x else x,
        
        # Country code variations
        lambda x: x.replace("GB", "UK") if x.startswith("GB") else x,
        lambda x: x.replace("UK", "GB") if x.startswith("UK") else x,
        
        # Missing characters
        lambda x: x[:-1] if len(x) > 10 else x,
        
        # No corruption
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    
    return random.choice(corruptions)(iban)

# Load the clean full merchant PII data
print("📂 Loading full merchant PII data...")
df_clean = pd.read_csv('full_merchant_pii_data.csv')
print(f"📊 Loaded {len(df_clean):,} clean records")

# Create dirty version
print("🏗️ Creating dirty version with expanded corruption types...")
df_dirty = df_clean.copy()

# Track corruption statistics
corruption_stats = {}

# Apply corruptions to each field
corruption_functions = {
    'first_name': corrupt_name,
    'surname': corrupt_name,
    'full_name': corrupt_name,
    'email': corrupt_email,
    'address': corrupt_address,
    'phone_number': corrupt_phone,
    'date_of_birth': corrupt_date,
    'uk_nino': corrupt_nino,
    'pan_number': corrupt_pan,
    'uk_sort_code': corrupt_sort_code,
    'uk_account_number': corrupt_account_number,
    'uk_iban': corrupt_iban,
}

for field, corrupt_func in corruption_functions.items():
    if field in df_dirty.columns:
        print(f"   Corrupting {field}...")
        original_values = df_dirty[field].copy()
        df_dirty[field] = df_dirty[field].apply(corrupt_func)
        
        # Count changes
        changed = (original_values != df_dirty[field]).sum()
        corruption_stats[field] = {
            'total': len(df_dirty),
            'changed': changed,
            'unchanged': len(df_dirty) - changed,
            'change_rate': changed / len(df_dirty) * 100
        }

# Add version indicator
df_clean['data_version'] = 'clean'
df_dirty['data_version'] = 'dirty'

# Combine clean and dirty data
print("🔗 Combining clean and dirty datasets...")
df_combined = pd.concat([df_clean, df_dirty], ignore_index=True)

# Update record IDs to be unique
df_combined['original_record_id'] = df_combined['record_id']
df_combined['record_id'] = range(1, len(df_combined) + 1)

# Save combined dataset
filename = 'full_merchant_clean_and_dirty_pii_data.csv'
df_combined.to_csv(filename, index=False)

print(f"\n✅ FULL MERCHANT CLEAN & DIRTY DATA GENERATION COMPLETE")
print(f"📁 Saved to: {filename}")
print(f"📊 Total records: {len(df_combined):,}")
print(f"   Clean records: {len(df_clean):,}")
print(f"   Dirty records: {len(df_dirty):,}")

print(f"\n📈 CORRUPTION STATISTICS:")
print(f"{'Field':<20} {'Total':<8} {'Changed':<8} {'Rate':<10}")
print("-" * 50)
for field, stats in corruption_stats.items():
    print(f"{field:<20} {stats['total']:<8} {stats['changed']:<8} {stats['change_rate']:>6.1f}%")

print(f"\n🔍 CORRUPTION EXAMPLES:")
print("=" * 60)

# Show examples of corruptions
sample_indices = random.sample(range(len(df_clean)), 5)
for i, idx in enumerate(sample_indices):
    print(f"\nExample {i+1} (Record {idx+1}):")
    clean_record = df_clean.iloc[idx]
    dirty_record = df_dirty.iloc[idx]
    
    for field in ['full_name', 'email', 'phone_number', 'uk_nino', 'pan_number']:
        if field in clean_record:
            clean_val = clean_record[field]
            dirty_val = dirty_record[field]
            if clean_val != dirty_val:
                print(f"   {field}:")
                print(f"     Clean: '{clean_val}'")
                print(f"     Dirty: '{dirty_val}'")

print(f"\n🧪 NEW CORRUPTION TYPES ADDED:")
print("• Name corruptions: Case variations, character substitutions, extra spaces")
print("• Email corruptions: Missing @/., multiple @, domain variations")
print("• Address corruptions: Number changes, abbreviation variations, missing parts")
print("• Phone corruptions: Format changes, country code variations, extensions")
print("• Date corruptions: Separator variations, year format changes, spaces")
print("• NINO corruptions: Character substitutions (O/0, I/1), spacing, case")
print("• PAN corruptions: Format changes, masking patterns, digit corruption")
print("• Sort code corruptions: Separator variations, digit changes")
print("• Account number corruptions: Leading zero issues, spacing, separators")
print("• IBAN corruptions: Country code variations, character substitutions, spacing")

print(f"\n📋 DATASET READY FOR COMPREHENSIVE PII ROBUSTNESS TESTING")
