#!/usr/bin/env python3
"""
Enhanced chat analyzer with UK postcode detection.
Updated version of the original analyzer that includes UK postcode recognition.
"""

import pandas as pd
import csv
import random
import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

def create_enhanced_analyzer_with_uk_postcodes():
    """Create an enhanced analyzer that includes UK postcode detection"""
    
    # Create the base analyzer
    analyzer = AnalyzerEngine()
    
    # Create UK postcode recognizer
    uk_postcode_pattern = Pattern(
        name="uk_postcode_pattern",
        regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b',
        score=0.9
    )
    
    uk_postcode_recognizer = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[uk_postcode_pattern]
    )
    
    # Add the custom recognizer to the analyzer
    analyzer.registry.add_recognizer(uk_postcode_recognizer)
    
    return analyzer

def extract_address_parts(address):
    """Extract different parts of an address"""
    if pd.isna(address) or not address:
        return "", "", ""
    
    # Split address by commas
    parts = [part.strip() for part in address.split(',')]
    
    # First portion (usually street address)
    first_portion = parts[0] if parts else ""
    
    # Last portion (usually country)
    last_portion = parts[-1] if parts else ""
    
    # Extract UK postcode pattern (letters and numbers at the end)
    postcode_match = re.search(r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b', address)
    postcode = postcode_match.group() if postcode_match else ""
    
    return first_portion, postcode, last_portion

def test_enhanced_analyzer():
    """Test the enhanced analyzer with some examples"""
    
    print("=== TESTING ENHANCED ANALYZER WITH UK POSTCODES ===\n")
    
    # Create enhanced analyzer
    analyzer = create_enhanced_analyzer_with_uk_postcodes()
    anonymizer = AnonymizerEngine()
    
    # Show supported entities
    entities = analyzer.get_supported_entities()
    print(f"Enhanced analyzer supports {len(entities)} entities:")
    for entity in sorted(entities):
        marker = "🆕" if entity == "UK_POSTCODE" else "  "
        print(f"{marker} {entity}")
    
    print(f"\n=== TESTING WITH SAMPLE TEXTS ===\n")
    
    # Test samples with UK postcodes
    test_samples = [
        "Hi! I'm John Smith and I live in SW1A 1AA area.",
        "My postcode is M1 1AA if you need directions.",
        "Contact me at john@example.com or visit CF48 3SH.",
        "I'm located at 123 High Street, London W1A 0AX.",
        "Email: mary@test.com, postcode: EX4 6NN, phone: 07123456789"
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"{i}. Text: {sample}")
        
        # Analyze with enhanced analyzer
        results = analyzer.analyze(text=sample, language='en')
        
        print("   Detected entities:")
        for result in results:
            detected_text = sample[result.start:result.end]
            marker = "🆕" if result.entity_type == "UK_POSTCODE" else "  "
            print(f"   {marker} {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")
        
        # Create masked version
        if results:
            anonymized_result = anonymizer.anonymize(text=sample, analyzer_results=results)
            print(f"   Masked: {anonymized_result.text}")
        
        print()
    
    return analyzer

def run_enhanced_analysis_sample():
    """Run a small sample analysis using the enhanced analyzer"""
    
    print("=== RUNNING ENHANCED ANALYSIS SAMPLE ===\n")
    
    # Create enhanced analyzer
    analyzer = create_enhanced_analyzer_with_uk_postcodes()
    anonymizer = AnonymizerEngine()
    
    # Sample data - using some from our previous analysis
    sample_inputs = [
        "Hi! I'm Sarah Johnson and my postcode is M1 1AA.",
        "You can find me at john.smith@parliament.uk or SW1A 1AA area.",
        "My details: Dr. Brown, email: test@example.com, area: CF48 3SH.",
        "Contact me at 123 High Street, London W1A 0AX for meetings.",
        "I'm located in EX4 6NN postcode near the city center."
    ]
    
    results = []
    
    for i, chat_input in enumerate(sample_inputs, 1):
        # Analyze
        analysis_results = analyzer.analyze(text=chat_input, language='en')
        
        # Create masked version
        if analysis_results:
            anonymized_result = anonymizer.anonymize(text=chat_input, analyzer_results=analysis_results)
            masked_input = anonymized_result.text
        else:
            masked_input = chat_input
        
        # Check for UK postcode detection
        uk_postcode_detected = any(r.entity_type == "UK_POSTCODE" for r in analysis_results)
        
        # Extract entity information
        entity_info = []
        for result in analysis_results:
            detected_text = chat_input[result.start:result.end]
            entity_info.append(f"{result.entity_type}:{detected_text}")
        
        result_data = {
            'input_id': i,
            'original_input': chat_input,
            'masked_input': masked_input,
            'uk_postcode_detected': uk_postcode_detected,
            'total_entities': len(analysis_results),
            'detected_entities': '; '.join(entity_info) if entity_info else 'None'
        }
        
        results.append(result_data)
        
        print(f"{i}. Original: {chat_input}")
        print(f"   Masked:   {masked_input}")
        print(f"   UK Postcode: {'✅ YES' if uk_postcode_detected else '❌ NO'}")
        print(f"   Entities:  {result_data['detected_entities']}")
        print()
    
    # Summary
    uk_detected_count = sum(1 for r in results if r['uk_postcode_detected'])
    total_entities = sum(r['total_entities'] for r in results)
    
    print(f"📊 SUMMARY:")
    print(f"   Total inputs: {len(results)}")
    print(f"   UK postcodes detected: {uk_detected_count}/{len(results)}")
    print(f"   Total entities found: {total_entities}")
    print(f"   Average entities per input: {total_entities/len(results):.1f}")
    
    return results

if __name__ == "__main__":
    # Test the enhanced analyzer
    enhanced_analyzer = test_enhanced_analyzer()
    
    # Run sample analysis
    sample_results = run_enhanced_analysis_sample()
    
    print("\n💡 HOW TO USE IN YOUR EXISTING CODE:")
    print("=" * 60)
    print("1. Replace this line:")
    print("   analyzer = AnalyzerEngine()")
    print("\n2. With this:")
    print("   analyzer = create_enhanced_analyzer_with_uk_postcodes()")
    print("\n3. Everything else works the same!")
    print("   - analyzer.analyze() now detects UK postcodes")
    print("   - UK_POSTCODE appears in get_supported_entities()")
    print("   - Anonymizer will mask UK postcodes automatically")
    
    print(f"\n✅ UK postcode detection successfully integrated!")
    print(f"   The enhanced analyzer maintains all original functionality")
    print(f"   while adding robust UK postcode recognition.")
