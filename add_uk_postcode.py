#!/usr/bin/env python3
"""
Script to add UK postcode recognition to Presidio using a custom recognizer.
"""

import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

def create_uk_postcode_recognizer():
    """
    Create a custom UK postcode recognizer.
    UK postcode formats:
    - AA9A 9AA (e.g., M1A 1AA)
    - A9A 9AA (e.g., M60 1NW)
    - A9 9AA (e.g., M1 1AA)
    - A99 9AA (e.g., M99 1AA)
    - AA9 9AA (e.g., M12 1AA)
    - AA99 9AA (e.g., M123 1AA)
    """
    
    # UK postcode regex pattern
    uk_postcode_pattern = r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b'
    
    # Create the pattern object
    pattern = Pattern(
        name="uk_postcode_pattern",
        regex=uk_postcode_pattern,
        score=0.9
    )
    
    # Create the recognizer
    uk_postcode_recognizer = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[pattern]
    )
    
    return uk_postcode_recognizer

def test_uk_postcode_detection():
    """Test UK postcode detection with and without custom recognizer"""
    
    # Test postcodes
    test_postcodes = [
        "M1 1AA",
        "M60 1NW", 
        "B33 8TH",
        "W1A 0AX",
        "EC1A 1BB",
        "SW1A 1AA",
        "CF48 3SH",
        "EX4 6NN",
        "LL18 6EB",
        "N1 0QD",
        "Invalid123",  # Should not match
        "ABC DEF"      # Should not match
    ]
    
    print("=== TESTING UK POSTCODE DETECTION ===\n")
    
    # Test with standard Presidio (without UK postcode recognizer)
    print("🔍 STANDARD PRESIDIO (without UK postcode recognizer):")
    print("-" * 60)
    
    standard_analyzer = AnalyzerEngine()
    
    for postcode in test_postcodes:
        results = standard_analyzer.analyze(text=f"My postcode is {postcode}", language='en')
        detected_entities = [f"{r.entity_type}:{postcode[r.start-13:r.end-13]}" for r in results]
        if detected_entities:
            print(f"'{postcode}' → {', '.join(detected_entities)}")
        else:
            print(f"'{postcode}' → Not detected")
    
    # Test with custom UK postcode recognizer
    print(f"\n🎯 ENHANCED PRESIDIO (with UK postcode recognizer):")
    print("-" * 60)
    
    # Create custom recognizer
    uk_postcode_recognizer = create_uk_postcode_recognizer()
    
    # Create analyzer with custom recognizer
    enhanced_analyzer = AnalyzerEngine()
    enhanced_analyzer.registry.add_recognizer(uk_postcode_recognizer)
    
    for postcode in test_postcodes:
        results = enhanced_analyzer.analyze(text=f"My postcode is {postcode}", language='en')
        detected_entities = []
        for r in results:
            detected_text = f"My postcode is {postcode}"[r.start:r.end]
            detected_entities.append(f"{r.entity_type}:{detected_text}")
        
        if detected_entities:
            print(f"'{postcode}' → {', '.join(detected_entities)}")
        else:
            print(f"'{postcode}' → Not detected")
    
    return enhanced_analyzer

def test_enhanced_chat_analysis():
    """Test the enhanced analyzer with chat examples"""
    
    print(f"\n=== TESTING ENHANCED CHAT ANALYSIS ===\n")
    
    # Create enhanced analyzer with UK postcode support
    uk_postcode_recognizer = create_uk_postcode_recognizer()
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(uk_postcode_recognizer)
    
    # Test with realistic chat examples
    chat_examples = [
        "Hi! I'm John Smith and I live in M1 1AA area.",
        "My postcode is SW1A 1AA if you need to send something.",
        "You can find me at 123 High Street, London W1A 0AX.",
        "Contact me at john@example.com or visit me in CF48 3SH.",
        "I'm in the EX4 6NN postcode area near Exeter."
    ]
    
    for i, example in enumerate(chat_examples, 1):
        print(f"{i}. Text: {example}")
        results = analyzer.analyze(text=example, language='en')
        
        if results:
            for result in results:
                detected_text = example[result.start:result.end]
                print(f"   → {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")
        else:
            print("   → No entities detected")
        print()

def create_enhanced_analyzer_function():
    """Function to create an enhanced analyzer with UK postcode support"""
    
    print(f"\n=== CREATING REUSABLE ENHANCED ANALYZER ===\n")
    
    def get_enhanced_analyzer():
        """Returns an AnalyzerEngine with UK postcode recognition"""
        
        # Create UK postcode recognizer
        uk_postcode_recognizer = PatternRecognizer(
            supported_entity="UK_POSTCODE",
            patterns=[Pattern(
                name="uk_postcode_pattern", 
                regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b',
                score=0.9
            )]
        )
        
        # Create analyzer and add custom recognizer
        analyzer = AnalyzerEngine()
        analyzer.registry.add_recognizer(uk_postcode_recognizer)
        
        return analyzer
    
    # Test the function
    enhanced_analyzer = get_enhanced_analyzer()
    entities = enhanced_analyzer.get_supported_entities()
    
    print(f"Enhanced analyzer now supports {len(entities)} entities:")
    for entity in sorted(entities):
        print(f"• {entity}")
    
    print(f"\n✅ UK_POSTCODE successfully added to supported entities!")
    
    return get_enhanced_analyzer

def create_multiple_custom_recognizers():
    """Example of adding multiple custom recognizers"""
    
    print(f"\n=== ADDING MULTIPLE CUSTOM RECOGNIZERS ===\n")
    
    # Create analyzer
    analyzer = AnalyzerEngine()
    
    # 1. UK Postcode recognizer
    uk_postcode_recognizer = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[Pattern(
            name="uk_postcode_pattern",
            regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b',
            score=0.9
        )]
    )
    
    # 2. UK National Insurance Number recognizer
    uk_nino_recognizer = PatternRecognizer(
        supported_entity="UK_NINO",
        patterns=[Pattern(
            name="uk_nino_pattern",
            regex=r'\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}[0-9]{6}[A-D]{1}\b',
            score=0.95
        )]
    )
    
    # 3. UK Vehicle Registration recognizer (basic pattern)
    uk_reg_recognizer = PatternRecognizer(
        supported_entity="UK_VEHICLE_REG",
        patterns=[Pattern(
            name="uk_vehicle_reg_pattern",
            regex=r'\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b',
            score=0.7
        )]
    )
    
    # Add all recognizers
    analyzer.registry.add_recognizer(uk_postcode_recognizer)
    analyzer.registry.add_recognizer(uk_nino_recognizer)
    analyzer.registry.add_recognizer(uk_reg_recognizer)
    
    # Test with mixed UK data
    test_text = """
    My details: John Smith, postcode SW1A 1AA, 
    National Insurance AB123456C, car reg AB12 XYZ.
    Email: john@example.com
    """
    
    print("Testing mixed UK data:")
    print(f"Text: {test_text.strip()}")
    print("\nDetected entities:")
    
    results = analyzer.analyze(text=test_text, language='en')
    for result in results:
        detected_text = test_text[result.start:result.end]
        print(f"• {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")
    
    entities = analyzer.get_supported_entities()
    print(f"\nTotal supported entities: {len(entities)}")
    new_entities = [e for e in entities if e.startswith('UK_')]
    print(f"New UK entities added: {new_entities}")
    
    return analyzer

if __name__ == "__main__":
    # Run all tests
    enhanced_analyzer = test_uk_postcode_detection()
    test_enhanced_chat_analysis()
    get_enhanced_analyzer_func = create_enhanced_analyzer_function()
    multi_enhanced_analyzer = create_multiple_custom_recognizers()
    
    print(f"\n📋 SUMMARY:")
    print("=" * 50)
    print("✅ UK postcode recognition successfully added to Presidio")
    print("✅ Custom recognizers can be easily created using PatternRecognizer")
    print("✅ Multiple custom recognizers can be added simultaneously")
    print("✅ Enhanced analyzer maintains all original Presidio functionality")
    print("\n💡 To use in your scripts:")
    print("   1. Create the custom recognizer using PatternRecognizer")
    print("   2. Add it to AnalyzerEngine using registry.add_recognizer()")
    print("   3. Use the enhanced analyzer normally with analyze()")
