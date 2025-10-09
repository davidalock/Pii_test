#!/usr/bin/env python3
"""
Test PII detection on the full merchant clean vs dirty dataset
with comprehensive analysis of corruption impact
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import spacy
from collections import defaultdict
import random

# Set random seed for reproducible sampling
random.seed(42)

# Initialize Presidio analyzer with spaCy
print("🔧 Initializing Presidio analyzer with UK recognizers...")
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Add UK-specific custom recognizers
uk_postcode_pattern = PatternRecognizer(
    supported_entity="UK_POSTCODE", 
    patterns=[Pattern(
        name="uk_postcode_pattern",
        regex=r"\b[A-Z]{1,2}[0-9R][0-9A-Z]?\s*[0-9][A-Z]{2}\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_postcode_pattern)

uk_nino_pattern = PatternRecognizer(
    supported_entity="UK_NINO",
    patterns=[Pattern(
        name="uk_nino_pattern", 
        regex=r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s*\d{6}\s*[A-D]\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_nino_pattern)

uk_sort_code_pattern = PatternRecognizer(
    supported_entity="UK_SORT_CODE",
    patterns=[Pattern(
        name="uk_sort_code_pattern",
        regex=r"\b\d{2}[-\s:.]?\d{2}[-\s:.]?\d{2}\b",
        score=0.85
    )]
)
analyzer.registry.add_recognizer(uk_sort_code_pattern)

uk_iban_pattern = PatternRecognizer(
    supported_entity="UK_IBAN",
    patterns=[Pattern(
        name="uk_iban_pattern",
        regex=r"\b(GB|UK)\s*\d{2}\s*[A-Z0-9]{4}\s*\d{6}\s*\d{8}\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_iban_pattern)

# Load the combined clean and dirty data
print("📂 Loading full merchant clean and dirty PII data...")
df = pd.read_csv('full_merchant_clean_and_dirty_pii_data.csv')
print(f"📊 Loaded {len(df):,} total records")

# Separate clean and dirty data
df_clean = df[df['data_version'] == 'clean'].copy()
df_dirty = df[df['data_version'] == 'dirty'].copy()

print(f"   Clean records: {len(df_clean):,}")
print(f"   Dirty records: {len(df_dirty):,}")

# Test on samples to avoid overwhelming output
sample_size = 1000
clean_sample = df_clean.sample(n=sample_size, random_state=42)
dirty_sample = df_dirty.sample(n=sample_size, random_state=42)

print(f"\n🧪 Testing PII detection on {sample_size:,} samples each (clean vs dirty)...")

# Fields to test
pii_fields = [
    'first_name', 'surname', 'full_name', 'email', 'address', 
    'phone_number', 'date_of_birth', 'uk_nino', 'pan_number',
    'uk_sort_code', 'uk_account_number', 'uk_iban'
]

def analyze_sample(sample_df, version_name):
    """Analyze PII detection for a sample"""
    results = defaultdict(lambda: defaultdict(int))
    entity_counts = defaultdict(int)
    
    for index, row in sample_df.iterrows():
        if (index - sample_df.index[0]) % 200 == 0:
            print(f"   Processing {version_name} sample {index - sample_df.index[0] + 1}/{len(sample_df)}")
        
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
    
    return results, entity_counts

# Analyze clean and dirty samples
print("🧼 Analyzing clean data...")
clean_results, clean_entities = analyze_sample(clean_sample, "clean")

print("🧽 Analyzing dirty data...")
dirty_results, dirty_entities = analyze_sample(dirty_sample, "dirty")

# Generate comprehensive comparison report
print(f"\n📊 COMPREHENSIVE CLEAN VS DIRTY PII DETECTION ANALYSIS")
print(f"=" * 70)
print(f"Sample size: {sample_size:,} records each")

print(f"\n🎯 DETECTION RATES COMPARISON:")
print(f"{'Field':<20} {'Clean Rate':<12} {'Dirty Rate':<12} {'Difference':<12} {'Impact':<8}")
print("-" * 70)

impact_summary = {}
for field in pii_fields:
    clean_detected = clean_results[field]['detected']
    clean_total = clean_detected + clean_results[field]['not_detected']
    clean_rate = (clean_detected / clean_total * 100) if clean_total > 0 else 0
    
    dirty_detected = dirty_results[field]['detected']
    dirty_total = dirty_detected + dirty_results[field]['not_detected']
    dirty_rate = (dirty_detected / dirty_total * 100) if dirty_total > 0 else 0
    
    difference = dirty_rate - clean_rate
    impact = "📈" if difference > 2 else "📉" if difference < -2 else "➡️"
    
    print(f"{field:<20} {clean_rate:>8.1f}%    {dirty_rate:>8.1f}%    {difference:>+8.1f}%    {impact}")
    
    impact_summary[field] = {
        'clean_rate': clean_rate,
        'dirty_rate': dirty_rate,
        'difference': difference,
        'resilient': abs(difference) < 5.0
    }

print(f"\n🏷️  ENTITY TYPE COMPARISON:")
print(f"{'Entity Type':<20} {'Clean Count':<12} {'Dirty Count':<12} {'Change':<10}")
print("-" * 55)

all_entities = set(clean_entities.keys()) | set(dirty_entities.keys())
for entity_type in sorted(all_entities):
    clean_count = clean_entities.get(entity_type, 0)
    dirty_count = dirty_entities.get(entity_type, 0)
    change = dirty_count - clean_count
    change_str = f"{change:+d}" if change != 0 else "0"
    print(f"{entity_type:<20} {clean_count:<12} {dirty_count:<12} {change_str}")

# Resilience analysis
print(f"\n🛡️  CORRUPTION RESILIENCE ANALYSIS:")
resilient_fields = [field for field, data in impact_summary.items() if data['resilient']]
vulnerable_fields = [field for field, data in impact_summary.items() if not data['resilient']]

print(f"✅ Resilient fields ({len(resilient_fields)}): {', '.join(resilient_fields)}")
print(f"⚠️  Vulnerable fields ({len(vulnerable_fields)}): {', '.join(vulnerable_fields)}")

# Show specific corruption impact examples
print(f"\n📋 CORRUPTION IMPACT EXAMPLES:")
print("=" * 50)

# Compare matching records (same original_record_id)
sample_ids = clean_sample['original_record_id'].head(5).tolist()
for i, record_id in enumerate(sample_ids):
    clean_record = clean_sample[clean_sample['original_record_id'] == record_id].iloc[0]
    dirty_record = dirty_sample[dirty_sample['original_record_id'] == record_id].iloc[0]
    
    print(f"\nExample {i+1} (Original Record {record_id}):")
    
    # Test key fields
    test_fields = ['full_name', 'email', 'uk_nino', 'pan_number']
    for field in test_fields:
        if field in clean_record and field in dirty_record:
            clean_val = str(clean_record[field])
            dirty_val = str(dirty_record[field])
            
            if clean_val != dirty_val:
                # Test detection
                clean_analysis = analyzer.analyze(text=clean_val, language='en')
                dirty_analysis = analyzer.analyze(text=dirty_val, language='en')
                
                clean_detected = len(clean_analysis) > 0
                dirty_detected = len(dirty_analysis) > 0
                
                status = "✅" if clean_detected == dirty_detected else "❌"
                
                print(f"   {field} {status}:")
                print(f"     Clean: '{clean_val}' → {'Detected' if clean_detected else 'Not detected'}")
                print(f"     Dirty: '{dirty_val}' → {'Detected' if dirty_detected else 'Not detected'}")

# Overall summary
clean_total_detections = sum([data['detected'] for data in clean_results.values()])
dirty_total_detections = sum([data['detected'] for data in dirty_results.values()])
total_tests = len(pii_fields) * sample_size

retention_rate = (dirty_total_detections / clean_total_detections * 100) if clean_total_detections > 0 else 0

print(f"\n📈 OVERALL DETECTION PERFORMANCE:")
print(f"   Clean data detections: {clean_total_detections:,}/{total_tests:,} ({clean_total_detections/total_tests*100:.1f}%)")
print(f"   Dirty data detections: {dirty_total_detections:,}/{total_tests:,} ({dirty_total_detections/total_tests*100:.1f}%)")
print(f"   Detection retention rate: {retention_rate:.1f}%")

print(f"\n🔍 KEY INSIGHTS:")
most_resilient = max(impact_summary.items(), key=lambda x: x[1]['dirty_rate'])
most_vulnerable = min(impact_summary.items(), key=lambda x: x[1]['difference'])

print(f"   • Most resilient field: {most_resilient[0]} ({most_resilient[1]['dirty_rate']:.1f}% detection)")
print(f"   • Most vulnerable field: {most_vulnerable[0]} ({most_vulnerable[1]['difference']:+.1f}% impact)")
print(f"   • Overall system retention: {retention_rate:.1f}%")
print(f"   • Fields with >95% retention: {len([f for f, d in impact_summary.items() if abs(d['difference']) < 5])}/{len(pii_fields)}")

print(f"\n✅ COMPREHENSIVE PII ROBUSTNESS ANALYSIS COMPLETE")
print(f"📊 Dataset: {len(df):,} total records ready for production PII testing")
