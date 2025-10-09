#!/usr/bin/env python3
"""
Script to create a comprehensive test data file combining MP list names, 
modified email domains, merchant addresses, and generated financial/personal data.
"""

import pandas as pd
import csv
import random
import string

def generate_realistic_domains():
    """Generate a list of realistic domain names including personal and business domains"""
    
    # Personal email domains that individuals commonly use
    personal_domains = [
        'gmail.com', 'outlook.com', 'hotmail.com', 'yahoo.com', 'icloud.com',
        'protonmail.com', 'aol.com', 'live.com', 'msn.com', 'mail.com',
        'zoho.com', 'yandex.com', 'gmx.com', 'fastmail.com', 'tutanota.com',
        'hushmail.com', 'runbox.com', 'mailfence.com', 'posteo.de', 'startmail.com',
        'yahoo.co.uk', 'btinternet.com', 'sky.com', 'virginmedia.com', 'talktalk.net',
        'plusnet.com', 'tiscali.co.uk', 'ntlworld.com', 'lineone.net', 'blueyonder.co.uk'
    ]
    
    # Business/Corporate domains
    business_domains = [
        'meridian-corp.com', 'pinnacle-group.co.uk', 'summit-enterprises.com', 
        'sterling-solutions.net', 'apex-consulting.org', 'vertex-systems.com',
        'nexus-technologies.co.uk', 'catalyst-innovations.com', 'paradigm-corp.net',
        'vanguard-services.org', 'horizon-dynamics.com', 'zenith-partners.co.uk'
    ]
    
    # Technology/Digital domains
    tech_domains = [
        'digitalforce.com', 'cloudstream.net', 'bytevision.co.uk', 'netpulse.org',
        'techbridge.com', 'datafusion.net', 'cyberflow.co.uk', 'infowave.com',
        'smartgrid.org', 'pixelcraft.net', 'codestream.co.uk', 'webdynamo.com'
    ]
    
    # Professional services domains
    professional_domains = [
        'lawfield-associates.com', 'hartwell-consulting.co.uk', 'blackstone-advisory.net',
        'riverside-partners.org', 'cromwell-services.com', 'ashford-group.co.uk',
        'westfield-associates.net', 'eastgate-consulting.com', 'northbridge-advisory.org',
        'southpoint-partners.co.uk', 'millfield-services.com', 'oakwood-group.net'
    ]
    
    # Healthcare/Medical domains
    healthcare_domains = [
        'wellspring-health.com', 'carepoint-medical.co.uk', 'vitality-clinic.org',
        'healthbridge-services.net', 'medcare-solutions.com', 'wellness-partners.co.uk',
        'lifetech-medical.org', 'healthstream-services.net', 'medpoint-care.com',
        'vitacare-group.co.uk'
    ]
    
    # Financial services domains
    finance_domains = [
        'goldstone-finance.com', 'silverbridge-capital.co.uk', 'emerald-investments.net',
        'platinum-advisors.org', 'diamond-wealth.com', 'sapphire-financial.co.uk',
        'crystal-capital.net', 'pearl-investments.com', 'ruby-advisors.org',
        'amber-financial.co.uk'
    ]
    
    # Educational domains
    education_domains = [
        'learningbridge.org', 'knowledge-hub.edu', 'skillsacademy.co.uk',
        'brightfuture-education.org', 'mindbridge-learning.com', 'wisdompath.edu',
        'intellilearn.co.uk', 'studypoint.org', 'edgeucation-services.com',
        'learningcrest.edu'
    ]
    
    # Media/Creative domains
    media_domains = [
        'creativeedge.com', 'designstudio-pro.co.uk', 'mediastream.net',
        'artisanworks.org', 'visualcraft.com', 'storybridge-media.co.uk',
        'pixelworks.net', 'brandforge.com', 'creativehub.org',
        'designwave.co.uk'
    ]
    
    # Manufacturing/Industrial domains
    industrial_domains = [
        'steelworks-ltd.com', 'precision-engineering.co.uk', 'industrial-solutions.net',
        'manufacturing-pro.org', 'techforge-industries.com', 'metalcraft-ltd.co.uk',
        'engineered-systems.net', 'production-solutions.com', 'industrial-dynamics.org',
        'machinery-specialists.co.uk'
    ]
    
    # Retail/Commercial domains
    retail_domains = [
        'marketplace-direct.com', 'retail-solutions.co.uk', 'commerce-hub.net',
        'tradecenter-online.org', 'merchant-services.com', 'shopbridge.co.uk',
        'retailtech-solutions.net', 'commercial-partners.com', 'tradeworks.org',
        'marketplace-pro.co.uk'
    ]
    
    # Combine all domain categories (personal domains get higher weight for realism)
    all_domains = (personal_domains * 2 + business_domains + tech_domains + professional_domains + 
                  healthcare_domains + finance_domains + education_domains +
                  media_domains + industrial_domains + retail_domains)
    
    return all_domains

def generate_uk_mobile():
    """Generate UK mobile number - mix of country code and standard format"""
    if random.choice([True, False]):
        # UK standard format (07...)
        return f"0{random.randint(7000000000, 7999999999)}"
    else:
        # Country code format (+44 7...)
        return f"+44 {random.randint(7000000000, 7999999999)}"

def generate_uk_nino():
    """Generate test UK National Insurance Number"""
    # Format: AA 12 34 56 C
    # First two letters (avoid certain combinations)
    valid_first = ['AA', 'AB', 'AE', 'AH', 'AK', 'AL', 'AM', 'AP', 'AR', 'AS', 'AT', 'AW', 'AX', 'AY', 'AZ',
                   'BA', 'BB', 'BE', 'BH', 'BK', 'BL', 'BM', 'BT', 'CA', 'CB', 'CE', 'CH', 'CK', 'CL', 'CR']
    prefix = random.choice(valid_first)
    
    # Six digits
    digits = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    
    # Final letter (A, B, C, or D)
    suffix = random.choice(['A', 'B', 'C', 'D'])
    
    # Format as XX XX XX XX X
    formatted_nino = f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:6]} {suffix}"
    return formatted_nino

def generate_pan():
    """Generate test PAN (Primary Account Number) starting with 34, 4, or 5"""
    # Choose starting digit
    start = random.choice(['34', '4', '5'])
    
    if start == '34':
        # American Express format (15 digits total)
        remaining = 13
    else:
        # Visa/Mastercard format (16 digits total)
        remaining = 15
    
    # Generate remaining digits
    pan = start + ''.join([str(random.randint(0, 9)) for _ in range(remaining)])
    
    return pan

def generate_sort_code():
    """Generate UK bank sort code (XX-XX-XX format)"""
    # UK sort codes are 6 digits, often formatted as XX-XX-XX
    code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{code[:2]}-{code[2:4]}-{code[4:6]}"

def generate_account_number():
    """Generate UK bank account number (8 digits)"""
    return ''.join([str(random.randint(0, 9)) for _ in range(8)])

def generate_iban(sort_code, account_number):
    """Generate test UK IBAN"""
    # UK IBAN format: GB + 2 check digits + 4 bank code + 6 sort code + 8 account number
    # This is a simplified version for testing - real IBANs have complex check digit calculation
    bank_code = "ABCD"  # Test bank code
    check_digits = f"{random.randint(10, 99)}"
    sort_digits = sort_code.replace('-', '')
    
    iban = f"GB{check_digits}{bank_code}{sort_digits}{account_number}"
    return iban

def create_test_data():
    """Create comprehensive test data file"""
    
    print("🔄 Loading source data...")
    
    # Load MP list data
    mp_file = '/Users/davidlock/Downloads/soccer data python/testing poe/mplist.csv'
    mp_df = pd.read_csv(mp_file)
    
    # Load merchant addresses
    merchant_file = '/Users/davidlock/Downloads/soccer data python/testing poe/merchant_frame for ATM.csv'
    merchant_df = pd.read_csv(merchant_file)
    
    # Get data arrays
    forenames = mp_df['Forename'].dropna().tolist()
    surnames = mp_df['Surname'].dropna().tolist()
    original_emails = mp_df['Email'].dropna().tolist()
    addresses = merchant_df['formatted_address'].dropna().tolist()
    
    # Generate realistic domains
    realistic_domains = generate_realistic_domains()
    
    print(f"📊 Source data loaded:")
    print(f"   Forenames: {len(forenames)}")
    print(f"   Surnames: {len(surnames)}")
    print(f"   Original emails: {len(original_emails)}")
    print(f"   Addresses: {len(addresses)}")
    print(f"   Realistic domains: {len(realistic_domains)}")
    
    # Determine number of records (use minimum available or limit to reasonable number)
    num_records = min(len(forenames), len(surnames), 1000)  # Limit to 1000 for performance
    
    print(f"\n🔧 Generating {num_records} test records...")
    
    test_data = []
    
    for i in range(num_records):
        # Basic personal info
        forename = random.choice(forenames)
        surname = random.choice(surnames)
        full_name = f"{forename} {surname}"
        
        # Generate new email with random domain
        original_email = random.choice(original_emails)
        username = original_email.split('@')[0]  # Keep the username part
        new_domain = random.choice(realistic_domains)
        new_email = f"{username}@{new_domain}"
        
        # Random address
        address = random.choice(addresses)
        
        # Generate mobile phone
        mobile = generate_uk_mobile()
        
        # Generate UK National Insurance Number
        nino = generate_uk_nino()
        
        # Generate PAN
        pan = generate_pan()
        
        # Generate banking details
        sort_code = generate_sort_code()
        account_number = generate_account_number()
        iban = generate_iban(sort_code, account_number)
        
        # Add to test data
        record = {
            'record_id': i + 1,
            'forename': forename,
            'surname': surname,
            'full_name': full_name,
            'email': new_email,
            'address': address,
            'mobile_phone': mobile,
            'national_insurance': nino,
            'pan_number': pan,
            'sort_code': sort_code,
            'account_number': account_number,
            'iban': iban
        }
        
        test_data.append(record)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1} records...")
    
    return test_data

def save_test_data(test_data):
    """Save test data to CSV file"""
    
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/comprehensive_test_data.csv'
    
    fieldnames = [
        'record_id', 'forename', 'surname', 'full_name', 'email', 'address',
        'mobile_phone', 'national_insurance', 'pan_number', 'sort_code', 
        'account_number', 'iban'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in test_data:
            writer.writerow(record)
    
    print(f"✅ Test data saved to: {output_file}")
    return output_file

def show_data_samples(test_data, num_samples=5):
    """Display sample records"""
    
    print(f"\n📋 SAMPLE RECORDS ({num_samples} examples):")
    print("=" * 80)
    
    for i in range(min(num_samples, len(test_data))):
        record = test_data[i]
        print(f"\nRecord {i+1}:")
        print(f"   Name: {record['full_name']}")
        print(f"   Email: {record['email']}")
        print(f"   Address: {record['address']}")
        print(f"   Mobile: {record['mobile_phone']}")
        print(f"   NI Number: {record['national_insurance']}")
        print(f"   PAN: {record['pan_number']}")
        print(f"   Sort Code: {record['sort_code']}")
        print(f"   Account: {record['account_number']}")
        print(f"   IBAN: {record['iban']}")

def analyze_data_distribution(test_data):
    """Analyze the distribution of generated data"""
    
    print(f"\n📊 DATA ANALYSIS:")
    print("=" * 50)
    
    # Mobile phone format distribution
    country_code_count = sum(1 for record in test_data if record['mobile_phone'].startswith('+44'))
    standard_count = len(test_data) - country_code_count
    
    print(f"Mobile Phone Formats:")
    print(f"   Country code (+44): {country_code_count} ({country_code_count/len(test_data)*100:.1f}%)")
    print(f"   Standard (07...): {standard_count} ({standard_count/len(test_data)*100:.1f}%)")
    
    # PAN starting digits distribution
    pan_34_count = sum(1 for record in test_data if record['pan_number'].startswith('34'))
    pan_4_count = sum(1 for record in test_data if record['pan_number'].startswith('4'))
    pan_5_count = sum(1 for record in test_data if record['pan_number'].startswith('5'))
    
    print(f"\nPAN Number Distributions:")
    print(f"   Starting with 34 (Amex): {pan_34_count} ({pan_34_count/len(test_data)*100:.1f}%)")
    print(f"   Starting with 4 (Visa): {pan_4_count} ({pan_4_count/len(test_data)*100:.1f}%)")
    print(f"   Starting with 5 (Mastercard): {pan_5_count} ({pan_5_count/len(test_data)*100:.1f}%)")
    
    # Email domain distribution
    domain_counts = {}
    for record in test_data:
        domain = record['email'].split('@')[1]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop Email Domains:")
    for domain, count in top_domains:
        print(f"   {domain}: {count} ({count/len(test_data)*100:.1f}%)")

def create_presidio_test_script():
    """Create a script to test the generated data with Presidio"""
    
    test_script = '''#!/usr/bin/env python3
"""
Script to test Presidio PII detection on the comprehensive test data.
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern

def create_enhanced_analyzer():
    """Create analyzer with UK-specific recognizers"""
    analyzer = AnalyzerEngine()
    
    # UK Postcode
    uk_postcode = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[Pattern(
            name="uk_postcode",
            regex=r'\\b[A-Z]{1,2}[0-9][A-Z0-9]?\\s?[0-9][A-Z]{2}\\b',
            score=0.9
        )]
    )
    
    # UK National Insurance Number
    uk_nino = PatternRecognizer(
        supported_entity="UK_NINO",
        patterns=[Pattern(
            name="uk_nino",
            regex=r'\\b[A-Z]{2}\\s?\\d{2}\\s?\\d{2}\\s?\\d{2}\\s?[A-Z]\\b',
            score=0.9
        )]
    )
    
    # UK Sort Code
    uk_sort = PatternRecognizer(
        supported_entity="UK_SORT_CODE",
        patterns=[Pattern(
            name="uk_sort",
            regex=r'\\b\\d{2}-\\d{2}-\\d{2}\\b',
            score=0.9
        )]
    )
    
    analyzer.registry.add_recognizer(uk_postcode)
    analyzer.registry.add_recognizer(uk_nino)
    analyzer.registry.add_recognizer(uk_sort)
    
    return analyzer

def test_comprehensive_data():
    """Test PII detection on comprehensive test data"""
    
    # Load test data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/comprehensive_test_data.csv'
    df = pd.read_csv(data_file)
    
    analyzer = create_enhanced_analyzer()
    
    print(f"Testing PII detection on {len(df)} records...")
    
    # Test a sample record
    sample_record = df.iloc[0]
    test_text = f"My name is {sample_record['full_name']}, email: {sample_record['email']}, address: {sample_record['address']}, mobile: {sample_record['mobile_phone']}, NI: {sample_record['national_insurance']}, PAN: {sample_record['pan_number']}, sort code: {sample_record['sort_code']}, account: {sample_record['account_number']}, IBAN: {sample_record['iban']}"
    
    results = analyzer.analyze(text=test_text, language='en')
    
    print(f"\\nTest text: {test_text}")
    print(f"\\nDetected entities:")
    for result in results:
        detected_text = test_text[result.start:result.end]
        print(f"   {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")

if __name__ == "__main__":
    test_comprehensive_data()
'''
    
    script_file = '/Users/davidlock/Downloads/soccer data python/testing poe/test_comprehensive_data.py'
    with open(script_file, 'w') as f:
        f.write(test_script)
    
    print(f"\n📝 Created Presidio test script: test_comprehensive_data.py")

if __name__ == "__main__":
    print("🏗️  CREATING COMPREHENSIVE TEST DATA")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    
    # Save to file
    output_file = save_test_data(test_data)
    
    # Show samples
    show_data_samples(test_data)
    
    # Analyze distribution
    analyze_data_distribution(test_data)
    
    # Create Presidio test script
    create_presidio_test_script()
    
    print(f"\n✅ COMPREHENSIVE TEST DATA CREATED")
    print("=" * 60)
    print(f"📊 Records created: {len(test_data)}")
    print(f"💾 Data file: comprehensive_test_data.csv")
    print(f"🧪 Test script: test_comprehensive_data.py")
    print(f"\nData includes:")
    print(f"   • Names (forename, surname, full name)")
    print(f"   • Email addresses (with random domains)")
    print(f"   • UK addresses (from merchant data)")
    print(f"   • Mobile phones (mixed formats)")
    print(f"   • UK National Insurance numbers")
    print(f"   • PAN numbers (34/4/5 prefixes)")
    print(f"   • UK sort codes and account numbers")
    print(f"   • UK IBAN codes")
    print(f"\n🎯 Perfect for PII detection testing!")
