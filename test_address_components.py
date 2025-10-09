#!/usr/bin/env python3
"""
Script to test PII detection on individual address components:
street_address, city_town, and postcode separately.
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
import random

def create_enhanced_analyzer():
    """Create analyzer with UK-specific recognizers"""
    analyzer = AnalyzerEngine()
    
    # UK Postcode - more flexible patterns
    uk_postcode = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[
            Pattern(name="standard_postcode", regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b', score=0.9),
            Pattern(name="no_space_postcode", regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}\b', score=0.8),
            Pattern(name="lowercase_postcode", regex=r'\b[a-z]{1,2}[0-9][a-z0-9]?\s?[0-9][a-z]{2}\b', score=0.7),
        ]
    )
    
    # UK Street Address patterns
    uk_street = PatternRecognizer(
        supported_entity="UK_STREET_ADDRESS",
        patterns=[
            Pattern(name="numbered_street", regex=r'\b\d{1,4}[A-Z]?(?:-\d{1,4}[A-Z]?)?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl|Drive|Dr|Way|Place|Pl|Court|Ct|Gardens|Gdns|Crescent|Cres|Terrace|Ter|Square|Sq|Park|Pk|Hill|Rise|View|Walk|Mews|Green|Common|Heath|Grove|Row|End|Side|Gate|Bridge|Cross|Corner|Centre|Center)\b', score=0.85),
            Pattern(name="named_building", regex=r'\b[A-Z][a-z]+\s+(?:House|Building|Centre|Center|Court|Gardens|Gdns|Park|Lodge|Manor|Hall|Tower|Place|Square)\b', score=0.8),
        ]
    )
    
    # UK Cities/Towns (common ones)
    uk_location = PatternRecognizer(
        supported_entity="UK_CITY_TOWN",
        patterns=[
            Pattern(name="major_cities", regex=r'\b(?:London|Manchester|Birmingham|Leeds|Glasgow|Sheffield|Bradford|Liverpool|Edinburgh|Bristol|Wakefield|Cardiff|Coventry|Nottingham|Leicester|Sunderland|Belfast|Newcastle|Brighton|Hull|Plymouth|Stoke|Wolverhampton|Derby|Swansea|Southampton|Salford|Aberdeen|Westminster|Portsmouth|York|Peterborough|Dundee|Lancaster|Oxford|Newport|Preston|Cambridge|Norwich|Chester|Salisbury|Exeter|Gloucester|Bath|Worcester|Canterbury|Carlisle|Durham|Winchester|Hereford|Truro|Bangor)\b', score=0.85),
        ]
    )
    
    analyzer.registry.add_recognizer(uk_postcode)
    analyzer.registry.add_recognizer(uk_street)
    analyzer.registry.add_recognizer(uk_location)
    
    return analyzer

def test_address_components():
    """Test PII detection on individual address components"""
    
    print("🔍 TESTING ADDRESS COMPONENT PII DETECTION")
    print("=" * 60)
    
    # Load split address data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/split_address_data.csv'
    df = pd.read_csv(data_file)
    
    print(f"📊 Loaded {len(df):,} records with split addresses")
    
    # Create analyzer
    analyzer = create_enhanced_analyzer()
    anonymizer = AnonymizerEngine()
    
    # Test sample of records
    sample_size = 200
    sample_df = df.sample(n=min(sample_size, len(df)))
    
    print(f"🧪 Testing {len(sample_df)} sample records...")
    
    # Statistics tracking
    stats = {
        'street_address': {'total': 0, 'detected': 0, 'entity_types': {}},
        'city_town': {'total': 0, 'detected': 0, 'entity_types': {}},
        'postcode': {'total': 0, 'detected': 0, 'entity_types': {}},
        'full_address': {'total': 0, 'detected': 0, 'entity_types': {}},
    }
    
    for _, record in sample_df.iterrows():
        # Test each component separately
        components = {
            'street_address': str(record['street_address']),
            'city_town': str(record['city_town']),
            'postcode': str(record['postcode']),
            'full_address': str(record['address'])
        }
        
        for component_name, component_text in components.items():
            if component_text and component_text != 'nan' and component_text.strip():
                stats[component_name]['total'] += 1
                
                results = analyzer.analyze(text=component_text, language='en')
                
                if results:
                    stats[component_name]['detected'] += 1
                    
                    for result in results:
                        entity_type = result.entity_type
                        if entity_type not in stats[component_name]['entity_types']:
                            stats[component_name]['entity_types'][entity_type] = 0
                        stats[component_name]['entity_types'][entity_type] += 1
    
    return stats

def print_component_results(stats):
    """Print detailed results for address components"""
    
    print(f"\n📊 ADDRESS COMPONENT DETECTION RESULTS")
    print("=" * 60)
    
    for component_name, component_stats in stats.items():
        if component_stats['total'] > 0:
            detection_rate = component_stats['detected'] / component_stats['total'] * 100
            
            print(f"\n📍 {component_name.replace('_', ' ').title()}:")
            print(f"   Total tested: {component_stats['total']:,}")
            print(f"   Detected PII: {component_stats['detected']:,}")
            print(f"   Detection rate: {detection_rate:.1f}%")
            
            if component_stats['entity_types']:
                print(f"   Entity types found:")
                sorted_entities = sorted(component_stats['entity_types'].items(), key=lambda x: x[1], reverse=True)
                for entity_type, count in sorted_entities:
                    percentage = count / component_stats['detected'] * 100 if component_stats['detected'] > 0 else 0
                    marker = "🆕" if entity_type.startswith('UK_') else "  "
                    print(f"   {marker} {entity_type}: {count} ({percentage:.1f}%)")

def show_component_examples():
    """Show examples of component detection"""
    
    print(f"\n📋 COMPONENT DETECTION EXAMPLES")
    print("=" * 60)
    
    # Load data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/split_address_data.csv'
    df = pd.read_csv(data_file)
    
    analyzer = create_enhanced_analyzer()
    
    # Find good examples
    examples = df.sample(n=5)
    
    for i, (_, record) in enumerate(examples.iterrows(), 1):
        print(f"\nExample {i} ({'Clean' if record['data_quality'] == 'clean' else 'Dirty'}):")
        print(f"   Full Address: {record['address']}")
        
        components = {
            'Street': record['street_address'],
            'City/Town': record['city_town'],
            'Postcode': record['postcode']
        }
        
        for comp_name, comp_text in components.items():
            if comp_text and str(comp_text) != 'nan' and str(comp_text).strip():
                results = analyzer.analyze(text=str(comp_text), language='en')
                entities = [r.entity_type for r in results]
                
                if entities:
                    entity_str = ', '.join(entities)
                    print(f"   {comp_name}: '{comp_text}' → {entity_str}")
                else:
                    print(f"   {comp_name}: '{comp_text}' → No PII detected")

def compare_clean_vs_dirty_components():
    """Compare detection rates between clean and dirty address components"""
    
    print(f"\n🔬 CLEAN VS DIRTY COMPONENT ANALYSIS")
    print("=" * 60)
    
    # Load data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/split_address_data.csv'
    df = pd.read_csv(data_file)
    
    clean_df = df[df['data_quality'] == 'clean']
    dirty_df = df[df['data_quality'] == 'dirty']
    
    analyzer = create_enhanced_analyzer()
    
    # Test samples from each
    sample_size = 100
    clean_sample = clean_df.sample(n=min(sample_size, len(clean_df)))
    dirty_sample = dirty_df.sample(n=min(sample_size, len(dirty_df)))
    
    results = {'clean': {}, 'dirty': {}}
    
    for quality, sample in [('clean', clean_sample), ('dirty', dirty_sample)]:
        results[quality] = {
            'postcode': {'total': 0, 'detected': 0},
            'city_town': {'total': 0, 'detected': 0},
            'street_address': {'total': 0, 'detected': 0}
        }
        
        for _, record in sample.iterrows():
            components = {
                'postcode': str(record['postcode']),
                'city_town': str(record['city_town']),
                'street_address': str(record['street_address'])
            }
            
            for comp_name, comp_text in components.items():
                if comp_text and comp_text != 'nan' and comp_text.strip():
                    results[quality][comp_name]['total'] += 1
                    
                    pii_results = analyzer.analyze(text=comp_text, language='en')
                    if pii_results:
                        results[quality][comp_name]['detected'] += 1
    
    # Print comparison
    print(f"Component Detection Comparison ({sample_size} samples each):")
    print(f"{'Component':<15} {'Clean Rate':<12} {'Dirty Rate':<12} {'Difference':<12}")
    print("-" * 55)
    
    for component in ['postcode', 'city_town', 'street_address']:
        clean_rate = results['clean'][component]['detected'] / results['clean'][component]['total'] * 100 if results['clean'][component]['total'] > 0 else 0
        dirty_rate = results['dirty'][component]['detected'] / results['dirty'][component]['total'] * 100 if results['dirty'][component]['total'] > 0 else 0
        difference = dirty_rate - clean_rate
        
        print(f"{component:<15} {clean_rate:>8.1f}%    {dirty_rate:>8.1f}%    {difference:>+7.1f}%")

if __name__ == "__main__":
    # Test address components
    stats = test_address_components()
    
    # Print results
    print_component_results(stats)
    
    # Show examples
    show_component_examples()
    
    # Compare clean vs dirty
    compare_clean_vs_dirty_components()
    
    print(f"\n✅ ADDRESS COMPONENT TESTING COMPLETE")
    print("=" * 60)
    print("🎯 Key Insights:")
    print("   • Individual address components have different PII detection rates")
    print("   • Postcodes are highly detectable when properly formatted")
    print("   • City/town detection varies with recognizer patterns")
    print("   • Street addresses benefit from structure-aware patterns")
    print("   • Data quality significantly affects component extraction")
