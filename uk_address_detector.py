#!/usr/bin/env python3
"""
Script to add UK address recognition to Presidio using custom recognizers.
Covers various UK address patterns including street addresses, house numbers, and common UK address formats.
"""

import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

def create_uk_address_recognizers():
    """
    Create multiple UK address recognizers for different address components.
    UK addresses typically follow patterns like:
    - [House Number] [Street Name] [Street Type]
    - [House Name], [Street Name]
    - [Flat/Unit Number] [Building Name], [Street Name]
    """
    
    recognizers = []
    
    # 1. UK House Numbers (including ranges and letters)
    uk_house_number_pattern = Pattern(
        name="uk_house_number",
        regex=r'\b\d{1,4}[A-Z]?(?:-\d{1,4}[A-Z]?)?\b(?=\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl|Drive|Dr|Way|Place|Pl|Court|Ct|Gardens|Gdns|Crescent|Cres|Terrace|Ter|Square|Sq|Park|Pk|Hill|Rise|View|Walk|Mews|Green|Common|Heath|Grove|Row|End|Side|Gate|Bridge|Cross|Corner|Centre|Center))?)',
        score=0.8
    )
    
    uk_house_number_recognizer = PatternRecognizer(
        supported_entity="UK_HOUSE_NUMBER",
        patterns=[uk_house_number_pattern]
    )
    
    # 2. UK Street Names with common street types
    uk_street_pattern = Pattern(
        name="uk_street_name",
        regex=r'\b(?:\d{1,4}[A-Z]?(?:-\d{1,4}[A-Z]?)?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl|Drive|Dr|Way|Place|Pl|Court|Ct|Gardens|Gdns|Crescent|Cres|Terrace|Ter|Square|Sq|Park|Pk|Hill|Rise|View|Walk|Mews|Green|Common|Heath|Grove|Row|End|Side|Gate|Bridge|Cross|Corner|Centre|Center)\b',
        score=0.85
    )
    
    uk_street_recognizer = PatternRecognizer(
        supported_entity="UK_STREET_ADDRESS",
        patterns=[uk_street_pattern]
    )
    
    # 3. UK Building/Flat references
    uk_building_pattern = Pattern(
        name="uk_building_ref",
        regex=r'\b(?:Flat|Apartment|Apt|Unit|Suite|Floor|Room)\s+\d{1,3}[A-Z]?\b|\b\d{1,3}[A-Z]?\s+(?:Flat|Apartment|Apt|Unit|Suite|Floor|Room)\b',
        score=0.8
    )
    
    uk_building_recognizer = PatternRecognizer(
        supported_entity="UK_BUILDING_REF",
        patterns=[uk_building_pattern]
    )
    
    # 4. UK Area/District names (common UK place names)
    uk_area_pattern = Pattern(
        name="uk_area_name",
        regex=r'\b(?:London|Manchester|Birmingham|Leeds|Glasgow|Sheffield|Bradford|Liverpool|Edinburgh|Bristol|Wakefield|Cardiff|Coventry|Nottingham|Leicester|Sunderland|Belfast|Newcastle|Brighton|Hull|Plymouth|Stoke|Wolverhampton|Derby|Swansea|Southampton|Salford|Aberdeen|Westminster|Portsmouth|York|Peterborough|Dundee|Lancaster|Oxford|Newport|Preston|Cambridge|Norwich|Chester|Salisbury|Exeter|Gloucester|Bath|Worcester|Canterbury|Carlisle|Durham|Winchester|Hereford|Truro|Bangor|St\s+Albans|St\s+Davids)\b',
        score=0.7
    )
    
    uk_area_recognizer = PatternRecognizer(
        supported_entity="UK_AREA_NAME",
        patterns=[uk_area_pattern]
    )
    
    # 5. Full UK Address Pattern (more comprehensive)
    uk_full_address_pattern = Pattern(
        name="uk_full_address",
        regex=r'\b(?:(?:Flat|Apartment|Apt|Unit|Suite|Floor|Room)\s+\d{1,3}[A-Z]?,?\s*)?(?:\d{1,4}[A-Z]?(?:-\d{1,4}[A-Z]?)?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl|Drive|Dr|Way|Place|Pl|Court|Ct|Gardens|Gdns|Crescent|Cres|Terrace|Ter|Square|Sq|Park|Pk|Hill|Rise|View|Walk|Mews|Green|Common|Heath|Grove|Row|End|Side|Gate|Bridge|Cross|Corner|Centre|Center)(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*(?:,\s*[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})?\b',
        score=0.9
    )
    
    uk_full_address_recognizer = PatternRecognizer(
        supported_entity="UK_FULL_ADDRESS",
        patterns=[uk_full_address_pattern]
    )
    
    return [
        uk_house_number_recognizer,
        uk_street_recognizer,
        uk_building_recognizer,
        uk_area_recognizer,
        uk_full_address_recognizer
    ]

def create_uk_postcode_recognizer():
    """Create UK postcode recognizer (from previous example)"""
    uk_postcode_pattern = Pattern(
        name="uk_postcode_pattern",
        regex=r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b',
        score=0.9
    )
    
    return PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[uk_postcode_pattern]
    )

def create_enhanced_uk_analyzer():
    """Create an analyzer with comprehensive UK address detection"""
    
    # Create base analyzer
    analyzer = AnalyzerEngine()
    
    # Add UK postcode recognizer
    uk_postcode_recognizer = create_uk_postcode_recognizer()
    analyzer.registry.add_recognizer(uk_postcode_recognizer)
    
    # Add all UK address recognizers
    uk_address_recognizers = create_uk_address_recognizers()
    for recognizer in uk_address_recognizers:
        analyzer.registry.add_recognizer(recognizer)
    
    return analyzer

def test_uk_address_detection():
    """Test UK address detection with various address formats"""
    
    print("=== TESTING UK ADDRESS DETECTION ===\n")
    
    # Test addresses covering different UK address patterns
    test_addresses = [
        "123 High Street, London",
        "45A Church Lane, Manchester",
        "Flat 2, Victoria Road, Birmingham",
        "Unit 15 Festival Park Factory Outlet Shopping Centre",
        "12-14 Queen Street, Edinburgh",
        "The Old Post Office, Mill Lane, Oxford",
        "Apartment 5B, Kings Court, Cambridge",
        "1st Floor, 89 Oxford Street, London W1A 0AX",
        "Ground Floor, Waterloo Station, London SE1 7LY",
        "Room 301, Student Halls, University Road, Bristol",
        "2 Pant Road, Dowlais, Merthyr Tydfil CF48 3SH",
        "St Peter's Square, Ruthin LL15 1AB",
        "Blue Square Ltd, St Paul's Road, Highbury, London N1 2NA"
    ]
    
    # Test with standard analyzer
    print("🔍 STANDARD PRESIDIO:")
    print("-" * 50)
    standard_analyzer = AnalyzerEngine()
    
    for address in test_addresses[:3]:  # Just test first 3 with standard
        results = standard_analyzer.analyze(text=address, language='en')
        detected = [f"{r.entity_type}:{address[r.start:r.end]}" for r in results]
        print(f"'{address}' → {', '.join(detected) if detected else 'Not detected'}")
    
    # Test with enhanced analyzer
    print(f"\n🎯 ENHANCED ANALYZER (with UK address detection):")
    print("-" * 50)
    
    enhanced_analyzer = create_enhanced_uk_analyzer()
    
    for address in test_addresses:
        results = enhanced_analyzer.analyze(text=address, language='en')
        detected = []
        for r in results:
            detected_text = address[r.start:r.end]
            detected.append(f"{r.entity_type}:{detected_text}")
        
        print(f"'{address}'")
        if detected:
            for detection in detected:
                marker = "🆕" if any(uk_type in detection for uk_type in ["UK_", "UK_HOUSE", "UK_STREET", "UK_BUILDING", "UK_AREA", "UK_FULL"]) else "  "
                print(f"   {marker} {detection}")
        else:
            print("   → Not detected")
        print()

def test_chat_examples_with_addresses():
    """Test realistic chat examples containing UK addresses"""
    
    print("=== TESTING CHAT EXAMPLES WITH UK ADDRESSES ===\n")
    
    enhanced_analyzer = create_enhanced_uk_analyzer()
    anonymizer = AnonymizerEngine()
    
    chat_examples = [
        "Hi! I'm John Smith and I live at 123 High Street, London W1A 0AX.",
        "My address is Flat 2A, Victoria Gardens, Manchester M1 1AA.",
        "You can find me at The Old Mill, Church Lane, Oxford OX1 2AB.",
        "I work at Unit 15, Business Park, Birmingham B1 1AA.",
        "Contact me at john@example.com or visit 45 Queen Street, Edinburgh EH1 1AA.",
        "My office is on the 3rd Floor, City Centre Building, Leeds LS1 1AA.",
        "I live near 12-14 Market Square, Canterbury CT1 1AA.",
        "Visit our shop at Ground Floor, Shopping Centre, Bristol BS1 1AA."
    ]
    
    total_uk_entities = 0
    
    for i, example in enumerate(chat_examples, 1):
        print(f"{i}. Text: {example}")
        
        results = enhanced_analyzer.analyze(text=example, language='en')
        
        uk_entities = [r for r in results if r.entity_type.startswith('UK_')]
        total_uk_entities += len(uk_entities)
        
        print("   Detected entities:")
        for result in results:
            detected_text = example[result.start:result.end]
            marker = "🆕" if result.entity_type.startswith('UK_') else "  "
            print(f"   {marker} {result.entity_type}: '{detected_text}' (confidence: {result.score:.3f})")
        
        # Show masked version
        if results:
            anonymized_result = anonymizer.anonymize(text=example, analyzer_results=results)
            print(f"   Masked: {anonymized_result.text}")
        
        print()
    
    print(f"📊 SUMMARY:")
    print(f"   Total UK-specific entities detected: {total_uk_entities}")
    print(f"   Average UK entities per message: {total_uk_entities/len(chat_examples):.1f}")

def show_supported_entities():
    """Show all supported entities in the enhanced analyzer"""
    
    print("=== SUPPORTED ENTITIES IN ENHANCED ANALYZER ===\n")
    
    enhanced_analyzer = create_enhanced_uk_analyzer()
    entities = enhanced_analyzer.get_supported_entities()
    
    standard_entities = []
    uk_entities = []
    
    for entity in sorted(entities):
        if entity.startswith('UK_'):
            uk_entities.append(entity)
        else:
            standard_entities.append(entity)
    
    print(f"📋 STANDARD ENTITIES ({len(standard_entities)}):")
    for entity in standard_entities:
        print(f"   {entity}")
    
    print(f"\n🆕 UK-SPECIFIC ENTITIES ({len(uk_entities)}):")
    for entity in uk_entities:
        print(f"   {entity}")
    
    print(f"\n📊 TOTAL: {len(entities)} supported entities")

def create_simple_uk_address_function():
    """Create a simple function for practical use"""
    
    print("\n=== SIMPLE FUNCTION FOR YOUR CODE ===\n")
    
    function_code = '''
def create_uk_enhanced_analyzer():
    """Create analyzer with UK postcode and address detection"""
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    
    analyzer = AnalyzerEngine()
    
    # UK Postcode
    uk_postcode = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[Pattern(
            name="uk_postcode",
            regex=r'\\b[A-Z]{1,2}[0-9][A-Z0-9]?\\s?[0-9][A-Z]{2}\\b',
            score=0.9
        )]
    )
    
    # UK Street Address
    uk_street = PatternRecognizer(
        supported_entity="UK_STREET_ADDRESS", 
        patterns=[Pattern(
            name="uk_street",
            regex=r'\\b(?:\\d{1,4}[A-Z]?(?:-\\d{1,4}[A-Z]?)?\\s+)?[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl|Drive|Dr|Way|Place|Pl|Court|Ct|Gardens|Gdns|Crescent|Cres|Terrace|Ter|Square|Sq|Park|Pk|Hill|Rise|View|Walk|Mews|Green|Common|Heath|Grove|Row|End|Side|Gate|Bridge|Cross|Corner|Centre|Center)\\b',
            score=0.85
        )]
    )
    
    analyzer.registry.add_recognizer(uk_postcode)
    analyzer.registry.add_recognizer(uk_street)
    
    return analyzer
'''
    
    print("Copy this function to your code:")
    print("=" * 60)
    print(function_code)
    
    print("Then use it like:")
    print("analyzer = create_uk_enhanced_analyzer()")

if __name__ == "__main__":
    # Run all tests
    test_uk_address_detection()
    test_chat_examples_with_addresses()
    show_supported_entities()
    create_simple_uk_address_function()
    
    print("\n✅ UK ADDRESS DETECTION SUMMARY:")
    print("=" * 50)
    print("🆕 UK_POSTCODE - UK postal codes")
    print("🆕 UK_HOUSE_NUMBER - House numbers (1, 12A, 45-47)")
    print("🆕 UK_STREET_ADDRESS - Street names with types")
    print("🆕 UK_BUILDING_REF - Flat/Unit/Floor references")
    print("🆕 UK_AREA_NAME - Major UK cities/areas")
    print("🆕 UK_FULL_ADDRESS - Complete address patterns")
    print("\n💡 This dramatically improves UK address detection!")
    print("   Addresses will be properly identified and masked.")
