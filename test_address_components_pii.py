#!/usr/bin/env python3
"""
Test PII detection on the new address components (first line, town/city, postcode)
and compare clean vs dirty performance
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import random

# Set random seed for reproducible sampling
random.seed(42)

print("🔧 Testing PII Detection on Address Components")
print("=" * 50)

# Initialize Presidio analyzer
print("Initializing Presidio analyzer...")
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Add UK postcode recognizer
uk_postcode_pattern = PatternRecognizer(
    supported_entity="UK_POSTCODE", 
    patterns=[Pattern(
        name="uk_postcode_pattern",
        regex=r"\b[A-Z]{1,2}[0-9R][0-9A-Z]?\s*[0-9][A-Z]{2}\b",
        score=0.9
    )]
)
analyzer.registry.add_recognizer(uk_postcode_pattern)

# Load sample data
print("Loading sample data...")
df = pd.read_csv('standardized_650_clean_and_dirty.csv')

# Test on a sample
sample_size = 100
sample_df = df.sample(n=sample_size, random_state=42)

print(f"\nTesting on {sample_size} sample records...")

# Address component fields to test
address_fields = [
    ('address_first_line', 'First Line'),
    ('address_town_city', 'Town/City'),  
    ('address_postcode', 'Postcode'),
    ('address_first_line_dirty', 'First Line (Dirty)'),
    ('address_town_city_dirty', 'Town/City (Dirty)'),
    ('address_postcode_dirty', 'Postcode (Dirty)')
]

results = {}

for field, display_name in address_fields:
    detected_count = 0
    total_count = 0
    entity_types = {}
    
    for _, row in sample_df.iterrows():
        if pd.notna(row[field]) and row[field] != '':
            text = str(row[field])
            total_count += 1
            
            # Analyze the text
            analysis_results = analyzer.analyze(text=text, language='en')
            
            if analysis_results:
                detected_count += 1
                for result in analysis_results:
                    entity_types[result.entity_type] = entity_types.get(result.entity_type, 0) + 1
    
    detection_rate = (detected_count / total_count * 100) if total_count > 0 else 0
    results[field] = {
        'display_name': display_name,
        'detected': detected_count,
        'total': total_count,
        'rate': detection_rate,
        'entities': entity_types
    }

print(f"\n📊 ADDRESS COMPONENT PII DETECTION RESULTS")
print(f"=" * 55)
print(f"{'Component':<25} {'Detected':<10} {'Total':<8} {'Rate':<8}")
print("-" * 55)

for field, data in results.items():
    print(f"{data['display_name']:<25} {data['detected']:<10} {data['total']:<8} {data['rate']:>5.1f}%")

print(f"\n🏷️  Entity Types Found:")
all_entities = {}
for field_data in results.values():
    for entity, count in field_data['entities'].items():
        all_entities[entity] = all_entities.get(entity, 0) + count

for entity, count in sorted(all_entities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {entity}: {count}")

# Compare clean vs dirty performance
print(f"\n📈 CLEAN VS DIRTY PERFORMANCE COMPARISON:")
print(f"=" * 45)

comparisons = [
    ('address_first_line', 'address_first_line_dirty', 'First Line'),
    ('address_town_city', 'address_town_city_dirty', 'Town/City'),
    ('address_postcode', 'address_postcode_dirty', 'Postcode')
]

for clean_field, dirty_field, component_name in comparisons:
    clean_rate = results[clean_field]['rate']
    dirty_rate = results[dirty_field]['rate']
    difference = dirty_rate - clean_rate
    
    impact = "📈" if difference > 2 else "📉" if difference < -2 else "➡️"
    
    print(f"{component_name}:")
    print(f"  Clean: {clean_rate:5.1f}%  |  Dirty: {dirty_rate:5.1f}%  |  Diff: {difference:+5.1f}% {impact}")

print(f"\n📋 SAMPLE DETECTIONS:")
print("=" * 30)

# Show a few detection examples
sample_records = sample_df.head(3)
for i, (_, row) in enumerate(sample_records.iterrows()):
    print(f"\nExample {i+1}:")
    print(f"  Address: {row['address']}")
    
    # Test each component
    for field, display_name in address_fields[:3]:  # Just clean versions for examples
        if pd.notna(row[field]) and row[field] != '':
            text = str(row[field])
            analysis = analyzer.analyze(text=text, language='en')
            
            entities = [f"{r.entity_type}" for r in analysis]
            status = "✅" if entities else "❌"
            
            print(f"    {display_name}: '{text}' {status} {entities}")

print(f"\n✅ ADDRESS COMPONENT PII TESTING COMPLETE")
print(f"🎯 Address components successfully extracted and PII detection tested")
print(f"📊 Ready for granular address-level PII analysis")
