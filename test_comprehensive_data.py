#!/usr/bin/env python3
"""
Script to test Presidio PII detection on the comprehensive test data.
"""

import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern

def create_enhanced_analyzer():
    """Create analyzer with UK-specific recognizers"""
    analyzer = AnalyzerEngine()
    
    # UK Postcode
    uk_postcode = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[Pattern(
            name="uk_postcode",
            regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b',
            score=0.9
        )]
    )
    
    # UK National Insurance Number
    uk_nino = PatternRecognizer(
        supported_entity="UK_NINO",
        patterns=[Pattern(
            name="uk_nino",
            regex=r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b',
            score=0.9
        )]
    )
    
    # UK Sort Code
    uk_sort = PatternRecognizer(
        supported_entity="UK_SORT_CODE",
        patterns=[Pattern(
            name="uk_sort",
            regex=r'\b\d{2}-\d{2}-\d{2}\b',
            score=0.9
        )]
    )
    
    analyzer.registry.add_recognizer(uk_postcode)
    analyzer.registry.add_recognizer(uk_nino)
    analyzer.registry.add_recognizer(uk_sort)
    
    return analyzer

def test_comprehensive_data():
    """Test PII detection on comprehensive test data"""
    
    # Load test data
    data_file = '/Users/davidlock/Downloads/soccer data python/testing poe/comprehensive_test_data.csv'
    df = pd.read_csv(data_file)
    
    analyzer = create_enhanced_analyzer()
    
    print(f"Testing PII detection on {len(df)} records...")
    
    # Test a sample record
    sample_record = df.iloc[0]
    test_text = f"My name is {sample_record['full_name']}, email: {sample_record['email']}, address: {sample_record['address']}, mobile: {sample_record['mobile_phone']}, NI: {sample_record['national_insurance']}, PAN: {sample_record['pan_number']}, sort code: {sample_record['sort_code']}, account: {sample_record['account_number']}, IBAN: {sample_record['iban']}"
    
    results = analyzer.analyze(text=test_text, language='en')
    
    print(f"\nTest text: {test_text}")
    print(f"\nDetected entities:")
    for result in results:
        detected_text = test_text[result.start:result.end]
        print(f"   {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")

if __name__ == "__main__":
    test_comprehensive_data()
