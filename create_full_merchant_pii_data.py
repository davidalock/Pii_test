#!/usr/bin/env python3
"""
Create comprehensive PII test data using ALL records from merchant frame
Based on the same data fields from previous comprehensive test data
"""

import pandas as pd
import random
import string
import uuid
from datetime import datetime, timedelta
import re

# Set random seed for reproducibility
random.seed(42)

def generate_uk_nino():
    """Generate a realistic UK National Insurance Number"""
    # First letter: not D, F, I, Q, U, V
    first_letters = ['A', 'B', 'C', 'E', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z']
    # Second letter: not D, F, I, Q, U, V, O (O also excluded from second position)
    second_letters = ['A', 'B', 'C', 'E', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z']
    
    first = random.choice(first_letters)
    second = random.choice(second_letters)
    numbers = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    suffix = random.choice(['A', 'B', 'C', 'D'])
    
    return f"{first}{second}{numbers}{suffix}"

def generate_pan_number():
    """Generate a realistic PAN (Primary Account Number) for credit/debit cards"""
    # Card type prefixes
    card_types = {
        'visa': ['4'],
        'mastercard': ['5'],
        'amex': ['34', '37']
    }
    
    card_type = random.choice(['visa', 'mastercard', 'amex'])
    prefix = random.choice(card_types[card_type])
    
    if card_type == 'amex':
        # Amex has 15 digits
        remaining_digits = 15 - len(prefix) - 1  # -1 for check digit
    else:
        # Visa and Mastercard have 16 digits
        remaining_digits = 16 - len(prefix) - 1  # -1 for check digit
    
    # Generate remaining digits (except check digit)
    middle_digits = ''.join([str(random.randint(0, 9)) for _ in range(remaining_digits)])
    
    # Simple check digit (not actual Luhn algorithm, just for testing)
    check_digit = str(random.randint(0, 9))
    
    pan = prefix + middle_digits + check_digit
    
    # Format with spaces
    if card_type == 'amex':
        return f"{pan[:4]} {pan[4:10]} {pan[10:]}"
    else:
        return f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:]}"

def generate_uk_sort_code():
    """Generate a realistic UK bank sort code"""
    # Format: XX-XX-XX
    return f"{random.randint(10, 99):02d}-{random.randint(10, 99):02d}-{random.randint(10, 99):02d}"

def generate_uk_account_number():
    """Generate a realistic UK bank account number"""
    return str(random.randint(10000000, 99999999))

def generate_uk_iban():
    """Generate a realistic UK IBAN"""
    # UK IBAN format: GB + 2 check digits + 4 bank code + 6 sort code + 8 account number
    check_digits = f"{random.randint(10, 99):02d}"
    bank_code = ''.join([str(random.randint(0, 9)) for _ in range(4)])
    sort_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    account_num = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    
    return f"GB{check_digits}{bank_code}{sort_code}{account_num}"

def generate_phone_number():
    """Generate realistic UK phone numbers"""
    formats = [
        # Mobile
        lambda: f"07{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}",
        # London landline
        lambda: f"020 {random.randint(1000, 9999)} {random.randint(1000, 9999)}",
        # Other landlines
        lambda: f"01{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)}",
        # Alternative mobile format
        lambda: f"+44 7{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}",
    ]
    return random.choice(formats)()

def generate_date_of_birth():
    """Generate realistic date of birth (18-80 years old)"""
    today = datetime.now()
    start_date = today - timedelta(days=80*365)  # 80 years ago
    end_date = today - timedelta(days=18*365)    # 18 years ago
    
    random_date = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    
    formats = [
        lambda d: d.strftime("%d/%m/%Y"),
        lambda d: d.strftime("%d-%m-%Y"),
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%d %B %Y"),
    ]
    
    return random.choice(formats)(random_date)

# Load names data
print("📂 Loading names data...")
names_df = pd.read_csv('mplist.csv')

# Personal email domains (30 personal + 70 business domains)
personal_domains = [
    'gmail.com', 'outlook.com', 'hotmail.com', 'yahoo.com', 'icloud.com',
    'live.com', 'aol.com', 'protonmail.com', 'tutanota.com', 'zoho.com',
    'yandex.com', 'mail.com', 'gmx.com', 'fastmail.com', 'hushmail.com',
    'rocketmail.com', 'btinternet.com', 'virginmedia.com', 'sky.com', 'talktalk.net',
    'plusnet.com', 'o2.co.uk', 'ee.co.uk', 'three.co.uk', 'vodafone.co.uk',
    'tiscali.co.uk', 'ntlworld.com', 'blueyonder.co.uk', 'freeserve.co.uk', 'lineone.net'
]

business_domains = [
    'accenture.com', 'apple.com', 'amazon.com', 'microsoft.com', 'google.com',
    'facebook.com', 'tesla.com', 'netflix.com', 'salesforce.com', 'oracle.com',
    'ibm.com', 'intel.com', 'cisco.com', 'adobe.com', 'nvidia.com',
    'paypal.com', 'uber.com', 'airbnb.com', 'spotify.com', 'dropbox.com',
    'zoom.us', 'slack.com', 'atlassian.com', 'shopify.com', 'stripe.com',
    'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'docker.com',
    'kubernetes.io', 'mongodb.com', 'redis.com', 'elastic.co', 'splunk.com',
    'tableau.com', 'snowflake.com', 'databricks.com', 'palantir.com', 'twilio.com',
    'okta.com', 'auth0.com', 'cloudflare.com', 'fastly.com', 'akamai.com',
    'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com', 'digitalocean.com', 'linode.com',
    'heroku.com', 'vercel.com', 'netlify.com', 'firebase.google.com', 'supabase.io',
    'planetscale.com', 'railway.app', 'render.com', 'fly.io', 'cyclic.sh',
    'replit.com', 'codesandbox.io', 'stackblitz.com', 'gitpod.io', 'codespaces.github.com',
    'notion.so', 'airtable.com', 'monday.com', 'asana.com', 'trello.com',
    'miro.com', 'figma.com', 'canva.com', 'sketch.com', 'invision.com'
]

all_domains = personal_domains + business_domains

def generate_email(first_name, surname):
    """Generate realistic email addresses"""
    domain = random.choice(all_domains)
    
    formats = [
        f"{first_name.lower()}.{surname.lower()}@{domain}",
        f"{first_name.lower()}{surname.lower()}@{domain}",
        f"{first_name.lower()}_{surname.lower()}@{domain}",
        f"{first_name[0].lower()}.{surname.lower()}@{domain}",
        f"{first_name.lower()}{random.randint(1, 999)}@{domain}",
        f"{surname.lower()}{first_name[0].lower()}@{domain}",
    ]
    
    return random.choice(formats)

# Load merchant frame
print("📂 Loading merchant frame data...")
merchant_df = pd.read_csv('merchant_frame for ATM.csv')
print(f"📊 Loaded {len(merchant_df)} merchant records")

# Prepare to generate comprehensive test data
print("🏗️ Generating comprehensive PII test data...")

test_data = []

for index, row in merchant_df.iterrows():
    if index % 1000 == 0:
        print(f"   Processing record {index + 1}/{len(merchant_df)}")
    
    # Get random name from MP list
    name_record = names_df.sample(n=1).iloc[0]
    first_name = name_record['Forename']
    surname = name_record['Surname']
    
    # Use the merchant address directly
    address = row['formatted_address']
    
    # Generate all PII fields
    record = {
        'record_id': index + 1,
        'first_name': first_name,
        'surname': surname,
        'full_name': f"{first_name} {surname}",
        'email': generate_email(first_name, surname),
        'address': address,
        'phone_number': generate_phone_number(),
        'date_of_birth': generate_date_of_birth(),
        'uk_nino': generate_uk_nino(),
        'pan_number': generate_pan_number(),
        'uk_sort_code': generate_uk_sort_code(),
        'uk_account_number': generate_uk_account_number(),
        'uk_iban': generate_uk_iban(),
        'merchant_name': row['name'] if pd.notna(row['name']) else 'ATM',
        'merchant_types': row['types'] if pd.notna(row['types']) else 'atm|finance',
        'latitude': row['res$results$geometry$location$lat'] if pd.notna(row['res$results$geometry$location$lat']) else '',
        'longitude': row['res$results$geometry$location$lng'] if pd.notna(row['res$results$geometry$location$lng']) else '',
        'is_open': row['res$results$opening_hours$open_now'] if pd.notna(row['res$results$opening_hours$open_now']) else '',
        'price_level': row['price_level'] if pd.notna(row['price_level']) else '',
        'place_id': row['place_id'] if pd.notna(row['place_id']) else ''
    }
    
    test_data.append(record)

# Create DataFrame and save
df = pd.DataFrame(test_data)

# Save the comprehensive dataset
filename = 'full_merchant_pii_data.csv'
df.to_csv(filename, index=False)

print(f"\n✅ FULL MERCHANT PII DATA GENERATION COMPLETE")
print(f"📁 Saved to: {filename}")
print(f"📊 Total records: {len(df):,}")
print(f"🎯 Records per field:")

# Count non-empty values for each field
for col in df.columns:
    if col not in ['record_id', 'merchant_name', 'merchant_types', 'latitude', 'longitude', 'is_open', 'price_level', 'place_id']:
        non_empty = df[col].notna().sum()
        print(f"   {col}: {non_empty:,} ({non_empty/len(df)*100:.1f}%)")

print(f"\n📈 Sample records:")
print(df.head(3).to_string())

print(f"\n🔍 Domain distribution in emails:")
domain_counts = df['email'].str.extract(r'@(.+)$')[0].value_counts().head(10)
for domain, count in domain_counts.items():
    print(f"   {domain}: {count} ({count/len(df)*100:.1f}%)")

print(f"\n📍 Address examples:")
for i in range(3):
    print(f"   {df.iloc[i]['address']}")
