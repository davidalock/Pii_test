#!/usr/bin/env python3
"""
Script to explore Presidio's available entity recognizers and supported entity types.
"""

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import *
import inspect

def explore_presidio_entities():
    # Initialize the analyzer
    analyzer = AnalyzerEngine()
    
    print("=== PRESIDIO ENTITY DETECTION CAPABILITIES ===\n")
    
    # Get all supported entities from the analyzer
    supported_entities = analyzer.get_supported_entities()
    print(f"📋 SUPPORTED ENTITIES ({len(supported_entities)} total):")
    print("=" * 50)
    for i, entity in enumerate(sorted(supported_entities), 1):
        print(f"{i:2d}. {entity}")
    
    print(f"\n🔍 DETAILED RECOGNIZER INFORMATION:")
    print("=" * 50)
    
    # Get all recognizers in the registry
    recognizers = analyzer.registry.recognizers
    
    for recognizer in recognizers:
        print(f"\n🏷️  {recognizer.__class__.__name__}")
        print(f"   Entities: {recognizer.supported_entities}")
        if hasattr(recognizer, 'supported_language'):
            print(f"   Language: {recognizer.supported_language}")
        elif hasattr(recognizer, 'supported_languages'):
            print(f"   Languages: {recognizer.supported_languages}")
        
        # Try to get description from docstring
        if recognizer.__doc__:
            doc_lines = recognizer.__doc__.strip().split('\n')
            if doc_lines:
                print(f"   Description: {doc_lines[0].strip()}")
    
    print(f"\n📚 PREDEFINED RECOGNIZER CLASSES:")
    print("=" * 50)
    
    # Get all predefined recognizer classes
    predefined_recognizers = []
    
    # Import the predefined recognizers module and inspect it
    import presidio_analyzer.predefined_recognizers as pred_rec
    
    for name, obj in inspect.getmembers(pred_rec, inspect.isclass):
        if name.endswith('Recognizer') and name != 'PatternRecognizer':
            try:
                # Try to instantiate to get supported entities
                instance = obj()
                if hasattr(instance, 'supported_entities'):
                    predefined_recognizers.append((name, instance.supported_entities))
            except:
                predefined_recognizers.append((name, "Could not instantiate"))
    
    for name, entities in sorted(predefined_recognizers):
        print(f"• {name}: {entities}")
    
    print(f"\n🌍 LANGUAGE SUPPORT:")
    print("=" * 50)
    
    # Check language support
    languages = ["en", "es", "fr", "de", "it", "pt", "he", "ar"]
    for lang in languages:
        try:
            lang_analyzer = AnalyzerEngine()
            supported = lang_analyzer.get_supported_entities(language=lang)
            print(f"{lang.upper()}: {len(supported)} entities - {sorted(supported)}")
        except Exception as e:
            print(f"{lang.upper()}: Error - {e}")
    
    print(f"\n🧪 TESTING SAMPLE TEXTS:")
    print("=" * 50)
    
    # Test with sample texts to see what gets detected
    test_samples = [
        "My name is John Doe and my email is john.doe@gmail.com",
        "Call me at +1-555-123-4567 or visit 123 Main St, New York, NY 10001",
        "My SSN is 123-45-6789 and credit card is 4111-1111-1111-1111",
        "Date of birth: 01/01/1990, IP address: 192.168.1.1",
        "License plate ABC123, IBAN GB29 NWBK 6016 1331 9268 19"
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. Testing: {sample}")
        results = analyzer.analyze(text=sample, language='en')
        if results:
            for result in results:
                detected_text = sample[result.start:result.end]
                print(f"   → {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")
        else:
            print("   → No entities detected")
    
    return supported_entities, recognizers

def test_specific_entities():
    """Test detection of specific entity types with examples"""
    
    analyzer = AnalyzerEngine()
    
    print(f"\n🎯 ENTITY-SPECIFIC TESTING:")
    print("=" * 50)
    
    # Define test cases for different entity types
    entity_tests = {
        "PERSON": [
            "John Smith", "Mary Johnson", "Dr. Robert Brown", "Ms. Sarah Wilson"
        ],
        "EMAIL_ADDRESS": [
            "test@example.com", "user.name@domain.co.uk", "admin@test-site.org"
        ],
        "PHONE_NUMBER": [
            "+1-555-123-4567", "(555) 123-4567", "555.123.4567", "+44 20 1234 5678"
        ],
        "LOCATION": [
            "New York", "London, UK", "123 Main Street, Boston", "California"
        ],
        "DATE_TIME": [
            "January 1, 2023", "01/01/2023", "2023-01-01", "next Monday"
        ],
        "US_SSN": [
            "123-45-6789", "987654321"
        ],
        "CREDIT_CARD": [
            "4111-1111-1111-1111", "5555 5555 5555 4444"
        ],
        "IP_ADDRESS": [
            "192.168.1.1", "10.0.0.1", "2001:db8::1"
        ],
        "US_DRIVER_LICENSE": [
            "D1234567", "DL123456789"
        ]
    }
    
    for entity_type, test_cases in entity_tests.items():
        print(f"\n📝 Testing {entity_type}:")
        for test_case in test_cases:
            results = analyzer.analyze(text=test_case, language='en')
            detected = [r for r in results if r.entity_type == entity_type]
            if detected:
                for result in detected:
                    detected_text = test_case[result.start:result.end]
                    print(f"   ✅ '{test_case}' → {result.entity_type} (confidence: {result.score:.3f})")
                    break
            else:
                # Check if any other entity was detected
                other_detected = [f"{r.entity_type}({r.score:.2f})" for r in results]
                if other_detected:
                    print(f"   ❌ '{test_case}' → Not detected as {entity_type}, found: {', '.join(other_detected)}")
                else:
                    print(f"   ❌ '{test_case}' → Not detected")

if __name__ == "__main__":
    supported_entities, recognizers = explore_presidio_entities()
    test_specific_entities()
    
    print(f"\n📊 SUMMARY:")
    print("=" * 50)
    print(f"Total supported entities: {len(supported_entities)}")
    print(f"Total recognizers loaded: {len(recognizers)}")
    print(f"Main entity categories: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME, CREDIT_CARD, etc.")
    print(f"\nFor more details, visit: https://microsoft.github.io/presidio/supported_entities/")
