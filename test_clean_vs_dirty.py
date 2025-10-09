#!/usr/bin/env python3
"""
Script to test PII detection robustness on clean vs dirty data.
Compares detection rates between properly formatted and corrupted data.
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

def create_enhanced_analyzer():
    """Create analyzer with UK-specific recognizers"""
    analyzer = AnalyzerEngine()
    
    # UK Postcode - more flexible pattern to handle corrupted formats
    uk_postcode = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[
            Pattern(name="standard_postcode", regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b', score=0.9),
            Pattern(name="no_space_postcode", regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}\b', score=0.8),
            Pattern(name="lowercase_postcode", regex=r'\b[a-z]{1,2}[0-9][a-z0-9]?\s?[0-9][a-z]{2}\b', score=0.7),
            Pattern(name="spaced_postcode", regex=r'\b[A-Z]\s[0-9]\s[0-9][A-Z]{2}\b', score=0.7),
        ]
    )
    
    # UK National Insurance Number - flexible patterns
    uk_nino = PatternRecognizer(
        supported_entity="UK_NINO",
        patterns=[
            Pattern(name="standard_nino", regex=r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b', score=0.9),
            Pattern(name="no_space_nino", regex=r'\b[A-Z]{2}\d{6}[A-Z]\b', score=0.8),
            Pattern(name="lowercase_nino", regex=r'\b[a-z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[a-z]\b', score=0.7),
            Pattern(name="hyphen_nino", regex=r'\b[A-Z]{2}-\d{2}-\d{2}-\d{2}-[A-Z]\b', score=0.8),
            Pattern(name="dot_nino", regex=r'\b[A-Z]{2}\.\d{2}\.\d{2}\.\d{2}\.[A-Z]\b', score=0.8),
        ]
    )
    
    # UK Sort Code - flexible patterns
    uk_sort = PatternRecognizer(
        supported_entity="UK_SORT_CODE",
        patterns=[
            Pattern(name="standard_sort", regex=r'\b\d{2}-\d{2}-\d{2}\b', score=0.9),
            Pattern(name="no_hyphen_sort", regex=r'\b\d{6}\b', score=0.6),
            Pattern(name="space_sort", regex=r'\b\d{2}\s\d{2}\s\d{2}\b', score=0.8),
            Pattern(name="dot_sort", regex=r'\b\d{2}\.\d{2}\.\d{2}\b', score=0.8),
        ]
    )
    
    # PAN Number - flexible patterns for credit cards
    uk_pan = PatternRecognizer(
        supported_entity="CREDIT_CARD_ENHANCED",
        patterns=[
            Pattern(name="standard_pan", regex=r'\b(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13})\b', score=0.9),
            Pattern(name="spaced_pan", regex=r'\b(?:4\d{3}\s\d{4}\s\d{4}\s\d{4}|5[1-5]\d{2}\s\d{4}\s\d{4}\s\d{4}|3[47]\d{1}\s\d{4}\s\d{6}\s\d{2})\b', score=0.8),
            Pattern(name="hyphen_pan", regex=r'\b(?:4\d{3}-\d{4}-\d{4}-\d{4}|5[1-5]\d{2}-\d{4}-\d{4}-\d{4}|3[47]\d{1}-\d{4}-\d{6}-\d{2})\b', score=0.8),
        ]
    )
    
    analyzer.registry.add_recognizer(uk_postcode)
    analyzer.registry.add_recognizer(uk_nino)
    analyzer.registry.add_recognizer(uk_sort)
    analyzer.registry.add_recognizer(uk_pan)
    
    return analyzer

def test_record_pii(analyzer, record):
    """Test PII detection on a single record"""
    # Create a text representation of the record
    test_text = f"Name: {record['full_name']}, Email: {record['email']}, Address: {record['address']}, Mobile: {record['mobile_phone']}, NI: {record['national_insurance']}, PAN: {record['pan_number']}, Sort: {record['sort_code']}, Account: {record['account_number']}, IBAN: {record['iban']}"
    
    results = analyzer.analyze(text=test_text, language='en')
    
    # Count detected entity types
    entity_counts = {}
    for result in results:
        entity_counts[result.entity_type] = entity_counts.get(result.entity_type, 0) + 1
    
    return len(results), entity_counts, test_text

def compare_clean_vs_dirty():
    """Compare PII detection between clean and dirty data"""
    
    print("🔍 TESTING PII DETECTION ROBUSTNESS")
    print("=" * 60)
    
    # Load combined data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/clean_and_dirty_test_data.csv'
    df = pd.read_csv(data_file)
    
    print(f"📊 Loaded {len(df):,} total records")
    
    # Split into clean and dirty
    clean_df = df[df['data_quality'] == 'clean']
    dirty_df = df[df['data_quality'] == 'dirty']
    
    print(f"   Clean records: {len(clean_df):,}")
    print(f"   Dirty records: {len(dirty_df):,}")
    
    # Create enhanced analyzer
    analyzer = create_enhanced_analyzer()
    
    # Test samples from each group
    sample_size = 100
    print(f"\n🧪 Testing {sample_size} samples from each group...")
    
    clean_sample = clean_df.sample(n=min(sample_size, len(clean_df)))
    dirty_sample = dirty_df.sample(n=min(sample_size, len(dirty_df)))
    
    # Analyze clean data
    print("📋 Analyzing clean data...")
    clean_stats = {'total_entities': 0, 'entity_types': {}, 'records_analyzed': 0}
    
    for _, record in clean_sample.iterrows():
        entity_count, entity_types, _ = test_record_pii(analyzer, record)
        clean_stats['total_entities'] += entity_count
        clean_stats['records_analyzed'] += 1
        
        for entity_type, count in entity_types.items():
            clean_stats['entity_types'][entity_type] = clean_stats['entity_types'].get(entity_type, 0) + count
    
    # Analyze dirty data
    print("📋 Analyzing dirty data...")
    dirty_stats = {'total_entities': 0, 'entity_types': {}, 'records_analyzed': 0, 'corruption_performance': {}}
    
    for _, record in dirty_sample.iterrows():
        entity_count, entity_types, _ = test_record_pii(analyzer, record)
        dirty_stats['total_entities'] += entity_count
        dirty_stats['records_analyzed'] += 1
        
        for entity_type, count in entity_types.items():
            dirty_stats['entity_types'][entity_type] = dirty_stats['entity_types'].get(entity_type, 0) + count
        
        # Track performance by corruption type
        corruptions = record['corruptions_applied']
        if corruptions not in dirty_stats['corruption_performance']:
            dirty_stats['corruption_performance'][corruptions] = {'entities': 0, 'count': 0}
        dirty_stats['corruption_performance'][corruptions]['entities'] += entity_count
        dirty_stats['corruption_performance'][corruptions]['count'] += 1
    
    return clean_stats, dirty_stats

def print_comparison_results(clean_stats, dirty_stats):
    """Print detailed comparison results"""
    
    print(f"\n📊 DETECTION COMPARISON RESULTS")
    print("=" * 60)
    
    # Overall statistics
    clean_avg = clean_stats['total_entities'] / clean_stats['records_analyzed']
    dirty_avg = dirty_stats['total_entities'] / dirty_stats['records_analyzed']
    detection_retention = (dirty_avg / clean_avg) * 100 if clean_avg > 0 else 0
    
    print(f"Overall Performance:")
    print(f"   Clean data average entities per record: {clean_avg:.2f}")
    print(f"   Dirty data average entities per record: {dirty_avg:.2f}")
    print(f"   Detection retention rate: {detection_retention:.1f}%")
    
    # Entity type comparison
    print(f"\n🎯 Entity Type Comparison:")
    print(f"{'Entity Type':<25} {'Clean':<8} {'Dirty':<8} {'Retention':<10}")
    print("-" * 55)
    
    all_entities = set(clean_stats['entity_types'].keys()) | set(dirty_stats['entity_types'].keys())
    
    for entity in sorted(all_entities):
        clean_count = clean_stats['entity_types'].get(entity, 0)
        dirty_count = dirty_stats['entity_types'].get(entity, 0)
        retention = (dirty_count / clean_count * 100) if clean_count > 0 else 0
        
        print(f"{entity:<25} {clean_count:<8} {dirty_count:<8} {retention:.1f}%")
    
    # Corruption impact analysis
    print(f"\n🔬 Impact by Corruption Type:")
    print(f"{'Corruption Type':<40} {'Count':<6} {'Avg Entities':<12}")
    print("-" * 60)
    
    corruption_performance = []
    for corruption, stats in dirty_stats['corruption_performance'].items():
        avg_entities = stats['entities'] / stats['count'] if stats['count'] > 0 else 0
        corruption_performance.append((corruption, stats['count'], avg_entities))
    
    # Sort by average entities (worst performing first)
    corruption_performance.sort(key=lambda x: x[2])
    
    for corruption, count, avg_entities in corruption_performance[:15]:  # Show top 15
        corruption_short = corruption[:37] + "..." if len(corruption) > 40 else corruption
        print(f"{corruption_short:<40} {count:<6} {avg_entities:.2f}")

def show_detection_examples():
    """Show specific examples of clean vs dirty detection"""
    
    print(f"\n📋 DETECTION EXAMPLES")
    print("=" * 60)
    
    # Load data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/clean_and_dirty_test_data.csv'
    df = pd.read_csv(data_file)
    
    analyzer = create_enhanced_analyzer()
    anonymizer = AnonymizerEngine()
    
    # Find a clean record and its corresponding corrupted version
    clean_sample = df[df['data_quality'] == 'clean'].sample(n=1).iloc[0]
    dirty_sample = df[df['data_quality'] == 'dirty'].sample(n=1).iloc[0]
    
    print("🟢 CLEAN RECORD EXAMPLE:")
    clean_count, clean_entities, clean_text = test_record_pii(analyzer, clean_sample)
    print(f"Original: {clean_text}")
    
    results = analyzer.analyze(text=clean_text, language='en')
    if results:
        anonymized = anonymizer.anonymize(text=clean_text, analyzer_results=results)
        print(f"Detected: {clean_count} entities - {clean_entities}")
        print(f"Masked: {anonymized.text}")
    
    print(f"\n🔴 DIRTY RECORD EXAMPLE:")
    print(f"Corruptions applied: {dirty_sample['corruptions_applied']}")
    dirty_count, dirty_entities, dirty_text = test_record_pii(analyzer, dirty_sample)
    print(f"Original: {dirty_text}")
    
    results = analyzer.analyze(text=dirty_text, language='en')
    if results:
        anonymized = anonymizer.anonymize(text=dirty_text, analyzer_results=results)
        print(f"Detected: {dirty_count} entities - {dirty_entities}")
        print(f"Masked: {anonymized.text}")

if __name__ == "__main__":
    # Run comparison
    clean_stats, dirty_stats = compare_clean_vs_dirty()
    
    # Print results
    print_comparison_results(clean_stats, dirty_stats)
    
    # Show examples
    show_detection_examples()
    
    print(f"\n✅ ROBUSTNESS TEST COMPLETE")
    print("=" * 60)
    print("🎯 Key Insights:")
    print("   • Shows how data quality affects PII detection")
    print("   • Identifies which corruptions are most problematic")
    print("   • Helps improve pattern recognition robustness")
    print("   • Guides data cleaning strategies")
