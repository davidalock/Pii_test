#!/usr/bin/env python3
"""
Enhanced Data Corruption System
Adds sophisticated corruptions including misspelled places, card number issues, 
name variations, and other realistic data quality problems
"""

import pandas as pd
import re
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

def corrupt_place_names(address):
    """Add common misspellings of UK place names"""
    if not address or pd.isna(address):
        return address, 'no_change'
    
    place_misspellings = {
        # Major cities
        'London': ['Londan', 'Londin', 'Londn', 'Londen'],
        'Manchester': ['Machester', 'Manchestor', 'Manchster', 'Manshester'],
        'Birmingham': ['Birmingam', 'Birmigham', 'Birminghan', 'Burmingham'],
        'Liverpool': ['Liverpol', 'Liverpoool', 'Liverpul', 'Liverppol'],
        'Edinburgh': ['Edinburg', 'Edinborough', 'Edinbourgh', 'Edingburgh'],
        'Glasgow': ['Glasgo', 'Glascow', 'Glasglow', 'Glagow'],
        'Cardiff': ['Cardif', 'Carrdiff', 'Cardff', 'Caerdiff'],
        'Belfast': ['Belast', 'Belfest', 'Bellfast', 'Belffast'],
        'Newcastle': ['Newcastel', 'New Castle', 'Newcatle', 'Newcstle'],
        'Leeds': ['Leds', 'Leedes', 'Leeeds', 'Leedz'],
        'Sheffield': ['Shefield', 'Sheffild', 'Sheffeild', 'Sheffied'],
        'Bristol': ['Bristal', 'Bristoll', 'Bristl', 'Bristel'],
        'Coventry': ['Coventary', 'Coventri', 'Coventrry', 'Covantry'],
        'Leicester': ['Liecester', 'Leicster', 'Leciester', 'Leiceter'],
        'Nottingham': ['Notingham', 'Nottingam', 'Nottinham', 'Notthingham'],
        
        # Common areas/districts
        'Westminster': ['Westminser', 'Westminister', 'Westminstor', 'Westminter'],
        'Kensington': ['Kensinton', 'Kensigton', 'Kensingtan', 'Kensingtom'],
        'Chelsea': ['Chelsa', 'Chelse', 'Chelsey', 'Chelsae'],
        'Piccadilly': ['Picadilly', 'Piccadily', 'Piccadilli', 'Picadilly'],
        'Gloucester': ['Glouster', 'Gloucestor', 'Gloucster', 'Glouceter'],
        'Worcester': ['Wocester', 'Worcestor', 'Worcster', 'Worceter'],
        'Leicester': ['Liecester', 'Leicster', 'Leciester', 'Leiceter'],
    }
    
    corrupted_address = address
    corruption_applied = False
    
    for correct_name, misspellings in place_misspellings.items():
        if correct_name in address:
            misspelled = random.choice(misspellings)
            corrupted_address = corrupted_address.replace(correct_name, misspelled)
            corruption_applied = True
            break
    
    if corruption_applied:
        return corrupted_address, 'misspelled_place_name'
    else:
        return address, 'no_change'

def corrupt_card_number(pan_number):
    """Add corruptions to card numbers including alpha characters"""
    if not pan_number or pd.isna(pan_number) or str(pan_number) == 'None':
        return pan_number, 'no_change'
    
    pan_str = str(pan_number)
    
    corruptions = [
        # Add alpha characters
        (lambda x: x[:4] + 'O' + x[5:], 'alpha_in_pan'),  # Replace digit with O
        (lambda x: x[:8] + 'l' + x[9:], 'alpha_in_pan'),  # Replace digit with l (lowercase L)
        (lambda x: x[:12] + 'I' + x[13:], 'alpha_in_pan'), # Replace digit with I
        
        # Missing digits
        (lambda x: x[:15], 'truncated_pan'),
        (lambda x: x[:14], 'truncated_pan'), 
        
        # Extra digits
        (lambda x: x + '9', 'extra_digit_pan'),
        (lambda x: x + '12', 'extra_digits_pan'),
        
        # Spaces in wrong places
        (lambda x: x[:4] + ' ' + x[4:8] + ' ' + x[8:12] + ' ' + x[12:], 'formatted_pan'),
        (lambda x: x[:6] + ' ' + x[6:], 'partial_space_pan'),
        
        # Hyphens instead of spaces
        (lambda x: x[:4] + '-' + x[4:8] + '-' + x[8:12] + '-' + x[12:], 'hyphenated_pan'),
        
        # Mixed case (shouldn't be letters but OCR errors)
        (lambda x: x.replace('0', 'O').replace('1', 'I'), 'ocr_error_pan'),
        
        # Missing leading zeros
        (lambda x: x[1:] if x.startswith('0') else x, 'missing_leading_zero'),
    ]
    
    corruption_func, corruption_type = random.choice(corruptions)
    try:
        corrupted = corruption_func(pan_str)
        return corrupted, corruption_type
    except:
        return pan_number, 'no_change'

def corrupt_name_spelling(name):
    """Add realistic name spelling variations"""
    if not name or pd.isna(name) or str(name) == 'None':
        return name, 'no_change'
    
    name_str = str(name).strip()
    
    # Common name misspellings and variations
    name_variations = {
        'John': ['Jon', 'Jhon', 'Johnn', 'Jhn'],
        'Michael': ['Micheal', 'Mikael', 'Michale', 'Micheel'],
        'David': ['Davd', 'Davied', 'Daivd', 'Davide'],
        'James': ['Jmes', 'Jame', 'Jamies', 'Jaimes'],
        'Robert': ['Robrt', 'Robbert', 'Robertt', 'Roberet'],
        'William': ['Willam', 'Willim', 'Willian', 'Wlliam'],
        'Richard': ['Richrd', 'Ricard', 'Richardd', 'Ricahrd'],
        'Thomas': ['Tomas', 'Thomass', 'Thmas', 'Thomaas'],
        'Christopher': ['Cristopher', 'Christophr', 'Christofer', 'Christpher'],
        'Daniel': ['Danial', 'Daneil', 'Danniel', 'Danieel'],
        'Paul': ['Pual', 'Paull', 'Pawl', 'Poul'],
        'Mark': ['Marc', 'Markk', 'Marrk', 'Mrak'],
        'Donald': ['Donal', 'Donaldd', 'Donld', 'Donnald'],
        'Steven': ['Stephen', 'Stephan', 'Stevan', 'Stven'],
        'Kenneth': ['Keneth', 'Kennet', 'Kennth', 'Keenneth'],
        'Andrew': ['Andew', 'Andrw', 'Andreu', 'Anddrew'],
        'Joshua': ['Johua', 'Joshau', 'Joshhua', 'Joshuua'],
        'Kevin': ['Kevan', 'Kevinn', 'Kvin', 'Keveen'],
        'Brian': ['Brien', 'Briane', 'Briann', 'Bryaan'],
        'George': ['Gorge', 'Georgee', 'Geroge', 'Geoorge'],
        'Sarah': ['Sara', 'Sarrah', 'Saraah', 'Sahra'],
        'Jennifer': ['Jenifer', 'Jennifr', 'Jeniffer', 'Jennfer'],
        'Lisa': ['Lesa', 'Lisaa', 'Lissa', 'Lysa'],
        'Sandra': ['Snadra', 'Sandre', 'Sandraa', 'Saandra'],
        'Donna': ['Dona', 'Donnaa', 'Danna', 'Donnna'],
        'Carol': ['Carrol', 'Caryl', 'Caroll', 'Caarol'],
        'Ruth': ['Ruthe', 'Ruthh', 'Rooth', 'Ruuth'],
        'Sharon': ['Sharron', 'Sharen', 'Sharyn', 'Sharonn'],
        'Michelle': ['Michel', 'Michell', 'Mishelle', 'Micheelle'],
        'Laura': ['Laure', 'Lauraa', 'Lara', 'Laara'],
        'Emily': ['Emilie', 'Emely', 'Emilly', 'Emiily'],
        'Kimberly': ['Kimberley', 'Kimberly', 'Kimerly', 'Kimbberley'],
        'Deborah': ['Debora', 'Debrah', 'Deboraah', 'Debborah'],
        'Dorothy': ['Dorthy', 'Dorothea', 'Dorothyy', 'Doroothy'],
        'Amy': ['Ammy', 'Amie', 'Aamy', 'Amyy'],
        'Angela': ['Angella', 'Anjela', 'Angelaa', 'Angeela'],
    }
    
    # Check if the name matches any in our variation dictionary
    for correct_name, variations in name_variations.items():
        if name_str.lower() == correct_name.lower():
            misspelled = random.choice(variations)
            return misspelled, 'misspelled_name'
    
    # Generic spelling corruptions for names not in dictionary
    generic_corruptions = [
        # Double letters
        (lambda x: re.sub(r'([aeiou])', r'\1\1', x, count=1), 'doubled_vowel'),
        (lambda x: re.sub(r'([bcdfghjklmnpqrstvwxyz])', r'\1\1', x, count=1), 'doubled_consonant'),
        
        # Missing letters
        (lambda x: x[:-1] if len(x) > 3 else x, 'missing_last_letter'),
        (lambda x: x[1:] if len(x) > 3 else x, 'missing_first_letter'),
        
        # Swapped letters
        (lambda x: x[:-2] + x[-1] + x[-2] if len(x) > 2 else x, 'swapped_last_two'),
        
        # Common letter substitutions
        (lambda x: x.replace('i', 'y'), 'i_to_y'),
        (lambda x: x.replace('y', 'i'), 'y_to_i'),
        (lambda x: x.replace('ph', 'f'), 'ph_to_f'),
        (lambda x: x.replace('ck', 'k'), 'ck_to_k'),
        
        # No change
        (lambda x: x, 'no_change'),
        (lambda x: x, 'no_change'),
    ]
    
    corruption_func, corruption_type = random.choice(generic_corruptions)
    try:
        corrupted = corruption_func(name_str)
        return corrupted, corruption_type
    except:
        return name, 'no_change'

def corrupt_email_domain(email):
    """Add email domain corruptions and typos"""
    if not email or pd.isna(email) or str(email) == 'None':
        return email, 'no_change'
    
    email_str = str(email)
    
    if '@' not in email_str:
        return email, 'no_change'
    
    local_part, domain = email_str.rsplit('@', 1)
    
    domain_corruptions = {
        'gmail.com': ['gmai.com', 'gmail.co', 'gmial.com', 'gmaill.com', 'gimail.com'],
        'yahoo.com': ['yaho.com', 'yahoo.co', 'yahooo.com', 'yahoo.cm', 'yhoo.com'],
        'hotmail.com': ['hotmai.com', 'hotmail.co', 'hotmial.com', 'htomail.com', 'hotmall.com'],
        'outlook.com': ['outlok.com', 'outlook.co', 'outloook.com', 'outllook.com', 'oultook.com'],
        'bt.com': ['bt.co', 'bt.cm', 'bbt.com', 'bt.ccom'],
        'sky.com': ['sky.co', 'sky.cm', 'skyy.com', 'skky.com'],
        'virgin.com': ['virgin.co', 'virign.com', 'virginn.com', 'virgin.cm'],
        'tesco.com': ['tesco.co', 'tescos.com', 'tescco.com', 'tesco.cm'],
    }
    
    for correct_domain, corruptions in domain_corruptions.items():
        if domain.lower() == correct_domain:
            corrupted_domain = random.choice(corruptions)
            return f"{local_part}@{corrupted_domain}", 'corrupted_email_domain'
    
    # Generic domain corruptions
    generic_corruptions = [
        (lambda d: d.replace('.com', '.co'), 'missing_m_domain'),
        (lambda d: d.replace('.com', '.cm'), 'missing_o_domain'),
        (lambda d: d + 'm' if d.endswith('.co') else d, 'extra_m_domain'),
        (lambda d: d.replace('.co.uk', '.com'), 'wrong_tld'),
        (lambda d: d.replace('.', ''), 'missing_dot'),
        (lambda d: d, 'no_change'),
    ]
    
    corruption_func, corruption_type = random.choice(generic_corruptions)
    try:
        corrupted_domain = corruption_func(domain)
        return f"{local_part}@{corrupted_domain}", corruption_type
    except:
        return email, 'no_change'

def corrupt_phone_number(phone):
    """Add phone number corruptions"""
    if not phone or pd.isna(phone) or str(phone) == 'None':
        return phone, 'no_change'
    
    phone_str = str(phone).replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
    
    corruptions = [
        # Wrong country code
        (lambda x: '+44' + x[3:] if x.startswith('+44') else '+44' + x, 'duplicate_country_code'),
        (lambda x: x.replace('+44', '+441'), 'wrong_country_code'),
        (lambda x: x.replace('+44', ''), 'missing_country_code'),
        
        # Missing digits
        (lambda x: x[:-1], 'truncated_phone'),
        (lambda x: x[:-2], 'truncated_phone_2'),
        
        # Extra digits
        (lambda x: x + '1', 'extra_digit_phone'),
        (lambda x: x + '99', 'extra_digits_phone'),
        
        # Wrong formatting
        (lambda x: x[:3] + '-' + x[3:6] + '-' + x[6:], 'hyphenated_phone'),
        (lambda x: x[:2] + ' ' + x[2:5] + ' ' + x[5:], 'spaced_phone'),
        
        # OCR errors
        (lambda x: x.replace('0', 'O'), 'zero_to_O_phone'),
        (lambda x: x.replace('1', 'I'), 'one_to_I_phone'),
        (lambda x: x.replace('5', 'S'), 'five_to_S_phone'),
        
        # No change
        (lambda x: x, 'no_change'),
    ]
    
    corruption_func, corruption_type = random.choice(corruptions)
    try:
        corrupted = corruption_func(phone_str)
        return corrupted, corruption_type
    except:
        return phone, 'no_change'

def corrupt_date_format(date_str):
    """Add date format corruptions"""
    if not date_str or pd.isna(date_str) or str(date_str) == 'None':
        return date_str, 'no_change'
    
    date_string = str(date_str)
    
    corruptions = [
        # Wrong separators
        (lambda x: x.replace('-', '/'), 'dash_to_slash_date'),
        (lambda x: x.replace('/', '-'), 'slash_to_dash_date'), 
        (lambda x: x.replace('-', '.'), 'dash_to_dot_date'),
        (lambda x: x.replace('/', '.'), 'slash_to_dot_date'),
        (lambda x: x.replace('/', ''), 'no_separator_date'),
        (lambda x: x.replace('-', ''), 'no_separator_date'),
        
        # Wrong format (DD/MM/YYYY vs MM/DD/YYYY)
        (lambda x: '/'.join(x.split('/')[::-1]) if '/' in x and len(x.split('/')) == 3 else x, 'reversed_date_format'),
        
        # Missing leading zeros
        (lambda x: re.sub(r'/0(\d)/', r'/\1/', x), 'missing_leading_zero_date'),
        (lambda x: re.sub(r'-0(\d)-', r'-\1-', x), 'missing_leading_zero_date'),
        
        # Two-digit years
        (lambda x: re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\1/\2/\3'[-2:], x), 'two_digit_year'),
        
        # No change
        (lambda x: x, 'no_change'),
    ]
    
    corruption_func, corruption_type = random.choice(corruptions)
    try:
        corrupted = corruption_func(date_string)
        return corrupted, corruption_type
    except:
        return date_str, 'no_change'

def corrupt_nino_format(nino):
    """Add National Insurance Number corruptions"""
    if not nino or pd.isna(nino) or str(nino) == 'None':
        return nino, 'no_change'
    
    nino_str = str(nino).replace(' ', '')
    
    corruptions = [
        # Missing spaces
        (lambda x: x, 'no_spaces_nino'),
        
        # Wrong spacing
        (lambda x: x[:2] + ' ' + x[2:4] + ' ' + x[4:6] + ' ' + x[6:8] + ' ' + x[8:], 'wrong_spaces_nino'),
        
        # Missing final letter
        (lambda x: x[:-1], 'missing_suffix_nino'),
        
        # Wrong case
        (lambda x: x.lower(), 'lowercase_nino'),
        (lambda x: x.upper(), 'uppercase_nino'),
        
        # OCR errors
        (lambda x: x.replace('0', 'O'), 'zero_to_O_nino'),
        (lambda x: x.replace('O', '0'), 'O_to_zero_nino'),
        (lambda x: x.replace('I', '1'), 'I_to_one_nino'),
        (lambda x: x.replace('1', 'I'), 'one_to_I_nino'),
        
        # Hyphens instead of spaces
        (lambda x: x[:2] + '-' + x[2:4] + '-' + x[4:6] + '-' + x[6:8] + '-' + x[8:], 'hyphenated_nino'),
        
        # No change
        (lambda x: x, 'no_change'),
    ]
    
    corruption_func, corruption_type = random.choice(corruptions)
    try:
        corrupted = corruption_func(nino_str)
        return corrupted, corruption_type
    except:
        return nino, 'no_change'

def apply_enhanced_corruptions_to_dataset(filename):
    """Apply enhanced corruptions to a dataset"""
    print(f"🔧 Applying enhanced corruptions to {filename}...")
    
    try:
        df = pd.read_csv(filename)
        
        # Track original columns
        original_columns = list(df.columns)
        
        # Add corruption tracking columns
        corruption_tracking_columns = [
            'place_name_corruption',
            'card_number_corruption', 
            'first_name_corruption',
            'last_name_corruption',
            'email_domain_corruption',
            'phone_corruption',
            'date_corruption',
            'nino_corruption'
        ]
        
        for col in corruption_tracking_columns:
            if col not in df.columns:
                df[col] = 'no_change'
        
        # Apply corruptions to dirty records only
        dirty_mask = df['data_version'] == 'dirty'
        dirty_records = df[dirty_mask].copy()
        
        if len(dirty_records) == 0:
            print("   No dirty records found to corrupt.")
            return
        
        print(f"   Applying enhanced corruptions to {len(dirty_records)} dirty records...")
        
        # Apply place name corruptions to addresses
        if 'address' in df.columns:
            for idx in dirty_records.index:
                corrupted_addr, corruption_type = corrupt_place_names(df.loc[idx, 'address'])
                df.loc[idx, 'address'] = corrupted_addr
                df.loc[idx, 'place_name_corruption'] = corruption_type
        
        # Apply card number corruptions
        if 'pan_number' in df.columns:
            for idx in dirty_records.index:
                corrupted_pan, corruption_type = corrupt_card_number(df.loc[idx, 'pan_number'])
                df.loc[idx, 'pan_number'] = corrupted_pan
                df.loc[idx, 'card_number_corruption'] = corruption_type
        
        # Apply name corruptions
        if 'first_name' in df.columns:
            for idx in dirty_records.index:
                corrupted_name, corruption_type = corrupt_name_spelling(df.loc[idx, 'first_name'])
                df.loc[idx, 'first_name'] = corrupted_name
                df.loc[idx, 'first_name_corruption'] = corruption_type
        
        if 'last_name' in df.columns:
            for idx in dirty_records.index:
                corrupted_name, corruption_type = corrupt_name_spelling(df.loc[idx, 'last_name'])
                df.loc[idx, 'last_name'] = corrupted_name
                df.loc[idx, 'last_name_corruption'] = corruption_type
        
        # Apply email domain corruptions
        if 'email' in df.columns:
            for idx in dirty_records.index:
                corrupted_email, corruption_type = corrupt_email_domain(df.loc[idx, 'email'])
                df.loc[idx, 'email'] = corrupted_email
                df.loc[idx, 'email_domain_corruption'] = corruption_type
        
        # Apply phone corruptions
        if 'phone_number' in df.columns:
            for idx in dirty_records.index:
                corrupted_phone, corruption_type = corrupt_phone_number(df.loc[idx, 'phone_number'])
                df.loc[idx, 'phone_number'] = corrupted_phone
                df.loc[idx, 'phone_corruption'] = corruption_type
        
        # Apply date corruptions
        if 'date_of_birth' in df.columns:
            for idx in dirty_records.index:
                corrupted_date, corruption_type = corrupt_date_format(df.loc[idx, 'date_of_birth'])
                df.loc[idx, 'date_of_birth'] = corrupted_date
                df.loc[idx, 'date_corruption'] = corruption_type
        
        # Apply NINO corruptions
        if 'national_insurance_number' in df.columns:
            for idx in dirty_records.index:
                corrupted_nino, corruption_type = corrupt_nino_format(df.loc[idx, 'national_insurance_number'])
                df.loc[idx, 'national_insurance_number'] = corrupted_nino
                df.loc[idx, 'nino_corruption'] = corruption_type
        
        # Save enhanced dataset
        df.to_csv(filename, index=False)
        
        # Report corruption statistics
        print(f"   ✅ Enhanced corruptions applied and saved to {filename}")
        print(f"   📊 Corruption summary:")
        
        for corruption_col in corruption_tracking_columns:
            if corruption_col in df.columns:
                corruption_counts = df[df['data_version'] == 'dirty'][corruption_col].value_counts()
                active_corruptions = corruption_counts[corruption_counts.index != 'no_change'].sum()
                total_dirty = len(df[df['data_version'] == 'dirty'])
                field_name = corruption_col.replace('_corruption', '')
                print(f"     {field_name}: {active_corruptions}/{total_dirty} records corrupted")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error applying enhanced corruptions: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ENHANCED CORRUPTION SYSTEM")
    print("=" * 40)
    
    # Files to enhance
    files_to_enhance = [
        'standardized_650_clean_and_dirty.csv',
        'standardized_9056_clean_and_dirty.csv',
        'unified_all_clean_and_dirty_pii_data.csv'
    ]
    
    for filename in files_to_enhance:
        print(f"\n📂 Processing {filename}...")
        success = apply_enhanced_corruptions_to_dataset(filename)
        
        if success:
            print(f"   ✅ {filename} enhanced successfully")
        else:
            print(f"   ❌ Failed to enhance {filename}")
    
    print(f"\n🎯 ENHANCED CORRUPTION SYSTEM COMPLETE")
    print("New corruption types added:")
    print("  🏢 Misspelled place names (London → Londan, etc.)")
    print("  💳 Card number corruptions (alpha chars, truncation)")
    print("  📝 Name spelling variations (Michael → Micheal, etc.)")
    print("  📧 Email domain typos (gmail.com → gmai.com, etc.)")
    print("  📞 Phone number format issues")
    print("  📅 Date format corruptions")
    print("  🆔 NINO format variations")
    print("  📊 Full corruption tracking per field")
