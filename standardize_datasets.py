#!/usr/bin/env python3
"""
Standardize field names and structure between the 650-record mplist dataset 
and the 9,056-record merchant dataset to make them identical
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

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

print("📂 Loading both datasets...")

# Load the 650-record mplist-based dataset
df_650 = pd.read_csv('comprehensive_test_data.csv')
print(f"📊 Loaded 650-record dataset: {len(df_650)} records")
print(f"   Current fields: {list(df_650.columns)}")

# Load the 9,056-record merchant-based dataset
df_9k = pd.read_csv('full_merchant_pii_data.csv')
print(f"📊 Loaded merchant dataset: {len(df_9k)} records")
print(f"   Current fields: {list(df_9k.columns)}")

print("\n🔧 Standardizing field structures...")

# Define the standardized field structure (combining all unique fields)
standardized_fields = [
    'record_id',
    'first_name',           # forename in 650 dataset
    'surname',
    'full_name',
    'email',
    'address',
    'phone_number',         # mobile_phone in 650 dataset
    'date_of_birth',        # missing in 650 dataset
    'uk_nino',             # national_insurance in 650 dataset
    'pan_number',
    'uk_sort_code',        # sort_code in 650 dataset
    'uk_account_number',   # account_number in 650 dataset
    'uk_iban',             # iban in 650 dataset
    'merchant_name',       # missing in 650 dataset
    'merchant_types',      # missing in 650 dataset
    'latitude',            # missing in 650 dataset
    'longitude',           # missing in 650 dataset
    'is_open',             # missing in 650 dataset
    'price_level',         # missing in 650 dataset
    'place_id'             # missing in 650 dataset
]

# Standardize the 650-record dataset
print("🔧 Standardizing 650-record dataset...")
df_650_std = pd.DataFrame()

# Map existing fields
df_650_std['record_id'] = df_650['record_id']
df_650_std['first_name'] = df_650['forename']
df_650_std['surname'] = df_650['surname']
df_650_std['full_name'] = df_650['full_name']
df_650_std['email'] = df_650['email']
df_650_std['address'] = df_650['address']
df_650_std['phone_number'] = df_650['mobile_phone']
df_650_std['uk_nino'] = df_650['national_insurance']
df_650_std['pan_number'] = df_650['pan_number']
df_650_std['uk_sort_code'] = df_650['sort_code']
df_650_std['uk_account_number'] = df_650['account_number']
df_650_std['uk_iban'] = df_650['iban']

# Add missing fields for 650-record dataset
print("   Adding missing fields to 650-record dataset...")
df_650_std['date_of_birth'] = [generate_date_of_birth() for _ in range(len(df_650_std))]
df_650_std['merchant_name'] = 'General Location'  # Generic merchant name
df_650_std['merchant_types'] = 'point_of_interest|establishment'  # Generic types
df_650_std['latitude'] = ''  # Empty coordinates
df_650_std['longitude'] = ''
df_650_std['is_open'] = ''
df_650_std['price_level'] = ''
df_650_std['place_id'] = ''

# Standardize the 9,056-record merchant dataset
print("🔧 Standardizing 9,056-record merchant dataset...")
df_9k_std = df_9k[standardized_fields].copy()

# Ensure all fields are in the correct order
df_650_std = df_650_std[standardized_fields]
df_9k_std = df_9k_std[standardized_fields]

print(f"\n✅ FIELD STANDARDIZATION COMPLETE")
print(f"📊 Standardized structure:")
print(f"   Total fields: {len(standardized_fields)}")
for i, field in enumerate(standardized_fields, 1):
    print(f"   {i:2d}. {field}")

# Save standardized datasets
print(f"\n💾 Saving standardized datasets...")

# Save individual standardized datasets
df_650_std.to_csv('standardized_650_records.csv', index=False)
df_9k_std.to_csv('standardized_9056_records.csv', index=False)

print(f"📁 Saved individual datasets:")
print(f"   standardized_650_records.csv: {len(df_650_std)} records")
print(f"   standardized_9056_records.csv: {len(df_9k_std)} records")

# Create combined dataset with source indicator
df_650_std['data_source'] = 'mplist_650'
df_9k_std['data_source'] = 'merchant_9056'

# Combine datasets
df_combined = pd.concat([df_650_std, df_9k_std], ignore_index=True)

# Update record IDs to be unique across combined dataset
df_combined['original_record_id'] = df_combined['record_id']
df_combined['record_id'] = range(1, len(df_combined) + 1)

# Save combined dataset
df_combined.to_csv('combined_standardized_pii_data.csv', index=False)

print(f"\n📁 Combined dataset:")
print(f"   combined_standardized_pii_data.csv: {len(df_combined)} records")
print(f"   650-record source: {len(df_combined[df_combined['data_source'] == 'mplist_650'])} records")
print(f"   9,056-record source: {len(df_combined[df_combined['data_source'] == 'merchant_9056'])} records")

# Verify field consistency
print(f"\n🔍 FIELD CONSISTENCY VERIFICATION:")
print(f"650-record dataset fields: {len(df_650_std.columns)}")
print(f"9,056-record dataset fields: {len(df_9k_std.columns)}")
print(f"Fields match: {'✅' if list(df_650_std.columns) == list(df_9k_std.columns) else '❌'}")

# Show sample data from both sources
print(f"\n📋 SAMPLE DATA COMPARISON:")
print(f"\n650-Record Dataset Sample:")
print(df_650_std.head(2)[['record_id', 'first_name', 'surname', 'email', 'uk_nino', 'merchant_name']].to_string())

print(f"\n9,056-Record Dataset Sample:")
print(df_9k_std.head(2)[['record_id', 'first_name', 'surname', 'email', 'uk_nino', 'merchant_name']].to_string())

# Field completion analysis
print(f"\n📊 FIELD COMPLETION ANALYSIS:")
print(f"{'Field':<20} {'650-Records':<12} {'9K-Records':<12}")
print("-" * 45)

for field in standardized_fields:
    if field not in ['record_id', 'original_record_id', 'data_source']:
        count_650 = df_650_std[field].notna().sum()
        count_9k = df_9k_std[field].notna().sum()
        
        pct_650 = count_650 / len(df_650_std) * 100
        pct_9k = count_9k / len(df_9k_std) * 100
        
        print(f"{field:<20} {pct_650:>7.1f}%     {pct_9k:>7.1f}%")

print(f"\n✅ DATASETS NOW HAVE IDENTICAL FIELD STRUCTURES")
print(f"🎯 Ready for unified PII testing and analysis")
