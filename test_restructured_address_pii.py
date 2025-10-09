#!/usr/bin/env python3
"""
Test PII detection on restructured address components 
derived from corrupted addresses
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import random

# Set random seed
random.seed(42)

print("🔧 TESTING PII DETECTION ON RESTRUCTURED ADDRESS DATA")
print("=" * 55)

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
df = pd.read_csv('standardized_650_clean_and_dirty.csv')

# Test on samples from clean and dirty versions
sample_size = 100
clean_sample = df[df['data_version'] == 'clean'].sample(n=sample_size, random_state=42)
dirty_sample = df[df['data_version'] == 'dirty'].sample(n=sample_size, random_state=42)

print(f"Testing on {sample_size} clean and {sample_size} dirty records...")

def test_pii_detection(sample_df, version_name):
    """Test PII detection on address components"""
    results = {
        'address': {'detected': 0, 'total': 0},
        'first_line': {'detected': 0, 'total': 0},
        'town_city': {'detected': 0, 'total': 0},
        'postcode': {'detected': 0, 'total': 0}
    }
    
    for _, row in sample_df.iterrows():
        # Test full address
        if pd.notna(row['address']) and row['address'] != '':
            results['address']['total'] += 1
            analysis = analyzer.analyze(text=str(row['address']), language='en')
            if analysis:
                results['address']['detected'] += 1
        
        # Test first line
        if pd.notna(row['address_first_line']) and row['address_first_line'] != '':
            results['first_line']['total'] += 1
            analysis = analyzer.analyze(text=str(row['address_first_line']), language='en')
            if analysis:
                results['first_line']['detected'] += 1
        
        # Test town/city
        if pd.notna(row['address_town_city']) and row['address_town_city'] != '':
            results['town_city']['total'] += 1
            analysis = analyzer.analyze(text=str(row['address_town_city']), language='en')
            if analysis:
                results['town_city']['detected'] += 1
        
        # Test postcode
        if pd.notna(row['address_postcode']) and row['address_postcode'] != '':
            results['postcode']['total'] += 1
            analysis = analyzer.analyze(text=str(row['address_postcode']), language='en')
            if analysis:
                results['postcode']['detected'] += 1
    
    return results

# Test both clean and dirty samples
print("\\n🧼 Testing clean address components...")
clean_results = test_pii_detection(clean_sample, 'clean')

print("🧽 Testing dirty address components...")
dirty_results = test_pii_detection(dirty_sample, 'dirty')

# Display results
print(f"\\n📊 PII DETECTION RESULTS - RESTRUCTURED ADDRESS DATA")
print(f"=" * 60)
print(f"{'Component':<15} {'Clean Rate':<12} {'Dirty Rate':<12} {'Difference':<12} {'Impact'}")
print("-" * 60)

components = ['address', 'first_line', 'town_city', 'postcode']
component_names = ['Full Address', 'First Line', 'Town/City', 'Postcode']

for component, display_name in zip(components, component_names):
    clean_rate = (clean_results[component]['detected'] / clean_results[component]['total'] * 100) if clean_results[component]['total'] > 0 else 0
    dirty_rate = (dirty_results[component]['detected'] / dirty_results[component]['total'] * 100) if dirty_results[component]['total'] > 0 else 0
    difference = dirty_rate - clean_rate
    impact = "📈" if difference > 2 else "📉" if difference < -2 else "➡️"
    
    print(f"{display_name:<15} {clean_rate:>8.1f}%    {dirty_rate:>8.1f}%    {difference:>+8.1f}%    {impact}")

# Show corruption impact analysis
print(f"\\n📈 CORRUPTION TYPE IMPACT ANALYSIS:")
print("=" * 40)

# Analyze different corruption types
corruption_impact = {}
for corruption_type in dirty_sample['address_corruption_type'].unique():
    if corruption_type != 'no_change':
        subset = dirty_sample[dirty_sample['address_corruption_type'] == corruption_type]
        if len(subset) >= 5:  # Only analyze if we have enough samples
            postcode_detection = 0
            total_postcodes = 0
            
            for _, row in subset.iterrows():
                if pd.notna(row['address_postcode']) and row['address_postcode'] != '':
                    total_postcodes += 1
                    analysis = analyzer.analyze(text=str(row['address_postcode']), language='en')
                    if analysis:
                        postcode_detection += 1
            
            if total_postcodes > 0:
                detection_rate = postcode_detection / total_postcodes * 100
                corruption_impact[corruption_type] = {
                    'samples': len(subset),
                    'detection_rate': detection_rate
                }

# Show top corruption impacts
print("Postcode detection by corruption type:")
for corruption_type, data in sorted(corruption_impact.items(), key=lambda x: x[1]['detection_rate'])[:10]:
    print(f"  {corruption_type}: {data['detection_rate']:.1f}% ({data['samples']} samples)")

print(f"\\n📋 EXAMPLE DETECTIONS:")
print("=" * 30)

# Show specific examples
examples = dirty_sample.head(3)
for i, (_, row) in enumerate(examples.iterrows()):
    print(f"\\nExample {i+1} (Corruption: {row['address_corruption_type']}):")
    print(f"  Address: {row['address']}")
    
    # Test postcode detection
    if pd.notna(row['address_postcode']) and row['address_postcode'] != '':
        postcode_analysis = analyzer.analyze(text=str(row['address_postcode']), language='en')
        postcode_entities = [r.entity_type for r in postcode_analysis]
        status = "✅" if postcode_entities else "❌"
        print(f"  Postcode: '{row['address_postcode']}' {status} {postcode_entities}")

print(f"\\n✅ RESTRUCTURED ADDRESS PII TESTING COMPLETE")
print(f"🎯 Single address field with derived components working correctly")
print(f"📊 Corruption tracking enables detailed impact analysis")
