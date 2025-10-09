#!/usr/bin/env python3
"""
Test PII detection on the full merchant PII dataset
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import spacy
from collections import defaultdict

# Initialize Presidio analyzer with spaCy
print("🔧 Initializing Presidio analyzer...")
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Add UK-specific custom recognizers
print("🎯 Adding UK-specific recognizers...")

# UK Postcode recognizer
uk_postcode_pattern = PatternRecognizer(
    supported_entity="UK_POSTCODE", 
    patterns=[Pattern(
        name="uk_postcode_pattern",
        regex=r"\b[A-Z]{1,2}[0-9R][0-9A-Z]?\s*[0-9][A-Z]{2}\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_postcode_pattern)

# UK National Insurance Number recognizer
uk_nino_pattern = PatternRecognizer(
    supported_entity="UK_NINO",
    patterns=[Pattern(
        name="uk_nino_pattern", 
        regex=r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_nino_pattern)

# UK Bank Sort Code recognizer
uk_sort_code_pattern = PatternRecognizer(
    supported_entity="UK_SORT_CODE",
    patterns=[Pattern(
        name="uk_sort_code_pattern",
        regex=r"\b\d{2}-\d{2}-\d{2}\b",
        score=0.85
    )]
)
analyzer.registry.add_recognizer(uk_sort_code_pattern)

# UK IBAN recognizer
uk_iban_pattern = PatternRecognizer(
    supported_entity="UK_IBAN",
    patterns=[Pattern(
        name="uk_iban_pattern",
        regex=r"\bGB\d{2}[A-Z0-9]{4}\d{6}\d{8}\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_iban_pattern)

# Load the full merchant PII data
print("📂 Loading full merchant PII data...")
df = pd.read_csv('full_merchant_pii_data.csv')
print(f"📊 Loaded {len(df):,} records")

# Test on a sample of records (1000 to avoid overwhelming output)
sample_size = 1000
test_df = df.sample(n=sample_size, random_state=42).copy()

print(f"🧪 Testing PII detection on {sample_size:,} sample records...")

# Fields to test
pii_fields = [
    'first_name', 'surname', 'full_name', 'email', 'address', 
    'phone_number', 'date_of_birth', 'uk_nino', 'pan_number',
    'uk_sort_code', 'uk_account_number', 'uk_iban'
]

results = defaultdict(lambda: defaultdict(int))
entity_counts = defaultdict(int)

for index, row in test_df.iterrows():
    if (index - test_df.index[0]) % 100 == 0:
        print(f"   Processing sample {index - test_df.index[0] + 1}/{sample_size}")
    
    for field in pii_fields:
        if pd.notna(row[field]):
            text = str(row[field])
            
            # Analyze the text
            analysis_results = analyzer.analyze(text=text, language='en')
            
            if analysis_results:
                results[field]['detected'] += 1
                for result in analysis_results:
                    entity_counts[result.entity_type] += 1
            else:
                results[field]['not_detected'] += 1

print(f"\n📊 PII DETECTION RESULTS ON FULL MERCHANT DATASET")
print(f"=" * 65)
print(f"Sample size: {sample_size:,} records from {len(df):,} total records")

print(f"\n🎯 Detection rates by field:")
print(f"{'Field':<20} {'Detected':<10} {'Total':<10} {'Rate':<10}")
print(f"-" * 50)

for field in pii_fields:
    detected = results[field]['detected']
    not_detected = results[field]['not_detected']
    total = detected + not_detected
    rate = (detected / total * 100) if total > 0 else 0
    print(f"{field:<20} {detected:<10} {total:<10} {rate:>5.1f}%")

print(f"\n🏷️  Entity types detected:")
print(f"{'Entity Type':<20} {'Count':<10} {'% of Detections':<15}")
print(f"-" * 45)

total_entities = sum(entity_counts.values())
for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_entities * 100) if total_entities > 0 else 0
    print(f"{entity_type:<20} {count:<10} {percentage:>10.1f}%")

# Show some examples
print(f"\n📋 DETECTION EXAMPLES:")
print(f"=" * 50)

example_records = test_df.head(5)
for idx, (index, row) in enumerate(example_records.iterrows()):
    print(f"\nExample {idx + 1}:")
    print(f"   Name: {row['full_name']}")
    print(f"   Email: {row['email']}")
    print(f"   Address: {row['address'][:70]}{'...' if len(row['address']) > 70 else ''}")
    print(f"   NINO: {row['uk_nino']}")
    print(f"   PAN: {row['pan_number']}")
    
    # Test the full combined record
    combined_text = f"{row['full_name']} {row['email']} {row['address']} {row['phone_number']} {row['uk_nino']} {row['pan_number']}"
    analysis = analyzer.analyze(text=combined_text, language='en')
    
    detected_entities = [f"{r.entity_type}:{combined_text[r.start:r.end]}" for r in analysis]
    print(f"   Detected: {', '.join(detected_entities[:5])}{'...' if len(detected_entities) > 5 else ''}")

print(f"\n✅ FULL MERCHANT PII ANALYSIS COMPLETE")
print(f"📈 Dataset statistics:")
print(f"   Total records: {len(df):,}")
print(f"   Unique addresses: {df['address'].nunique():,}")
print(f"   Address coverage: UK-wide merchant/ATM locations")
print(f"   PII fields per record: {len(pii_fields)}")
print(f"   Total PII data points: {len(df) * len(pii_fields):,}")
