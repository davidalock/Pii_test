#!/usr/bin/env python3
"""
Remove merchant-specific fields from both standardized datasets to create
cleaner PII-focused datasets
"""

import pandas as pd

# Fields to remove
fields_to_remove = [
    'merchant_name',
    'merchant_types', 
    'latitude',  # Note: user wrote "lattidude" but meant latitude
    'longitude', # Note: user wrote "logitude" but meant longitude
    'is_open',
    'price_level',
    'place_id'
]

print("🧹 REMOVING MERCHANT-SPECIFIC FIELDS FROM STANDARDIZED DATASETS")
print("=" * 65)

# Files to process
files_to_process = [
    'standardized_650_clean_and_dirty.csv',
    'standardized_9056_clean_and_dirty.csv',
    'unified_all_clean_and_dirty_pii_data.csv'
]

for filename in files_to_process:
    print(f"\n📂 Processing {filename}...")
    
    try:
        # Load the dataset
        df = pd.read_csv(filename)
        print(f"   Loaded: {len(df):,} records with {len(df.columns)} fields")
        
        # Show current fields
        print(f"   Current fields: {list(df.columns)}")
        
        # Remove specified fields if they exist
        fields_removed = []
        for field in fields_to_remove:
            if field in df.columns:
                df = df.drop(columns=[field])
                fields_removed.append(field)
        
        print(f"   Removed fields: {fields_removed}")
        print(f"   Remaining fields: {len(df.columns)}")
        
        # Save the cleaned dataset
        clean_filename = filename.replace('.csv', '_cleaned.csv')
        df.to_csv(clean_filename, index=False)
        
        print(f"   ✅ Saved as: {clean_filename}")
        print(f"   Final structure: {len(df):,} records × {len(df.columns)} fields")
        
    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")

print(f"\n📊 FIELD REMOVAL SUMMARY")
print(f"Fields removed: {', '.join(fields_to_remove)}")
print(f"Files processed: {len(files_to_process)}")

# Verify the cleaned datasets
print(f"\n🔍 VERIFYING CLEANED DATASETS")
cleaned_files = [f.replace('.csv', '_cleaned.csv') for f in files_to_process]

for filename in cleaned_files:
    try:
        df = pd.read_csv(filename)
        print(f"\n✅ {filename}:")
        print(f"   Records: {len(df):,}")
        print(f"   Fields: {len(df.columns)}")
        print(f"   Remaining fields: {list(df.columns)}")
        
        # Show sample to verify structure
        print(f"   Sample record fields:")
        sample_fields = ['record_id', 'first_name', 'surname', 'email', 'uk_nino', 'data_version', 'data_source']
        available_sample_fields = [f for f in sample_fields if f in df.columns]
        if available_sample_fields:
            print(f"   {df[available_sample_fields].head(1).to_string()}")
        
    except Exception as e:
        print(f"❌ Could not verify {filename}: {e}")

print(f"\n✅ FIELD REMOVAL COMPLETE")
print(f"🎯 Cleaned datasets focus on core PII fields only")
print(f"📁 Original files preserved, cleaned versions created with '_cleaned' suffix")
