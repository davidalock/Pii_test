#!/usr/bin/env python3
"""
Complete Documentation of All Data Corruptions Applied to Dirty Records
"""

print("🧪 COMPREHENSIVE CORRUPTION TYPES DOCUMENTATION")
print("=" * 50)

print("📋 ALL 24 CORRUPTION TYPES APPLIED TO DIRTY DATA:")
print("=" * 50)

corruptions = [
    {
        'name': 'no_change',
        'category': '🔒 Control',
        'description': 'No corruption applied - keeps some records clean for comparison',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, UK',
        'impact': 'None - baseline for testing'
    },
    {
        'name': 'uppercase',
        'category': '🔤 Case Changes',
        'description': 'Converts entire address to UPPERCASE',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 HIGH STREET, LONDON SW1A 1AA, UK',
        'impact': 'Tests case-insensitive PII detection'
    },
    {
        'name': 'lowercase', 
        'category': '🔤 Case Changes',
        'description': 'Converts entire address to lowercase',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 high street, london sw1a 1aa, uk',
        'impact': 'Tests case-insensitive PII detection'
    },
    {
        'name': 'title_case',
        'category': '🔤 Case Changes', 
        'description': 'Converts to Title Case (first letter of each word capitalized)',
        'example_before': '123 high street, london SW1A 1AA, UK',
        'example_after': '123 High Street, London Sw1A 1Aa, Uk',
        'impact': 'Tests mixed case handling'
    },
    {
        'name': 'extra_spaces',
        'category': '🔳 Spacing Issues',
        'description': 'Adds extra spaces at beginning and end',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': ' 123 High Street, London SW1A 1AA, UK ',
        'impact': 'Tests whitespace trimming'
    },
    {
        'name': 'spaced_commas',
        'category': '🔳 Spacing Issues',
        'description': 'Adds spaces around commas',
        'example_before': '123 High Street, London, SW1A 1AA, UK',
        'example_after': '123 High Street , London , SW1A 1AA , UK',
        'impact': 'Tests punctuation spacing tolerance'
    },
    {
        'name': 'normalized_spaces',
        'category': '🔳 Spacing Issues',
        'description': 'Removes double spaces (normalizes to single spaces)',
        'example_before': '123  High  Street, London  SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, UK',
        'impact': 'Tests multiple space handling'
    },
    {
        'name': 'removed_commas',
        'category': '🔳 Spacing Issues',
        'description': 'Removes all commas from address',
        'example_before': '123 High Street, London, SW1A 1AA, UK',
        'example_after': '123 High Street London SW1A 1AA UK',
        'impact': 'Tests comma-less address parsing'
    },
    {
        'name': 'number_change',
        'category': '🔢 Numeric Changes',
        'description': 'Changes house/building numbers by ±1-3',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '126 High Street, London SW1A 1AA, UK',
        'impact': 'Tests numeric variation tolerance'
    },
    {
        'name': 'street_abbreviation',
        'category': '📝 Abbreviations',
        'description': 'Abbreviates "Street" to "St"',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 High St, London SW1A 1AA, UK',
        'impact': 'Tests street type abbreviation handling'
    },
    {
        'name': 'road_abbreviation',
        'category': '📝 Abbreviations',
        'description': 'Abbreviates "Road" to "Rd"',
        'example_before': '123 High Road, London SW1A 1AA, UK',
        'example_after': '123 High Rd, London SW1A 1AA, UK',
        'impact': 'Tests road type abbreviation handling'
    },
    {
        'name': 'avenue_abbreviation',
        'category': '📝 Abbreviations',
        'description': 'Abbreviates "Avenue" to "Ave"',
        'example_before': '123 High Avenue, London SW1A 1AA, UK',
        'example_after': '123 High Ave, London SW1A 1AA, UK',
        'impact': 'Tests avenue abbreviation handling'
    },
    {
        'name': 'street_expansion',
        'category': '📝 Abbreviations',
        'description': 'Expands "St" to "Street"',
        'example_before': '123 High St, London SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, UK',
        'impact': 'Tests street type expansion handling'
    },
    {
        'name': 'road_expansion',
        'category': '📝 Abbreviations',
        'description': 'Expands "Rd" to "Road"',
        'example_before': '123 High Rd, London SW1A 1AA, UK',
        'example_after': '123 High Road, London SW1A 1AA, UK',
        'impact': 'Tests road type expansion handling'
    },
    {
        'name': 'postcode_no_space',
        'category': '📮 Postcode Format',
        'description': 'Removes space from UK postcode',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A1AA, UK',
        'impact': 'Tests postcode format flexibility'
    },
    {
        'name': 'postcode_add_space',
        'category': '📮 Postcode Format',
        'description': 'Adds space to postcode (if missing)',
        'example_before': '123 High Street, London SW1A1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, UK',
        'impact': 'Tests postcode normalization'
    },
    {
        'name': 'O_to_0',
        'category': '🔀 Character Substitution',
        'description': 'Replaces letter "O" with digit "0"',
        'example_before': '123 High Street, London OX1A 1AA, UK',
        'example_after': '123 High Street, London 0X1A 1AA, UK',
        'impact': 'Tests OCR-like character confusion'
    },
    {
        'name': '0_to_O',
        'category': '🔀 Character Substitution',
        'description': 'Replaces digit "0" with letter "O"',
        'example_before': '123 High Street, London SW10 1AA, UK',
        'example_after': '123 High Street, London SW1O 1AA, UK',
        'impact': 'Tests OCR-like character confusion'
    },
    {
        'name': 'I_to_1',
        'category': '🔀 Character Substitution',
        'description': 'Replaces letter "I" with digit "1"',
        'example_before': '123 High Street, London SW1I 1AA, UK',
        'example_after': '123 High Street, London SW11 1AA, UK',
        'impact': 'Tests OCR-like character confusion'
    },
    {
        'name': '1_to_I',
        'category': '🔀 Character Substitution',
        'description': 'Replaces digit "1" with letter "I"',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': 'I23 High Street, London SWIA IAA, UK',
        'impact': 'Tests OCR-like character confusion'
    },
    {
        'name': 'remove_last_part',
        'category': '✂️ Truncation',
        'description': 'Removes the last comma-separated part',
        'example_before': '123 High Street, London, SW1A 1AA, UK',
        'example_after': '123 High Street, London, SW1A 1AA',
        'impact': 'Tests incomplete address handling'
    },
    {
        'name': 'add_england',
        'category': '➕ Additions',
        'description': 'Adds ", England" to UK addresses',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, UK, England',
        'impact': 'Tests redundant location handling'
    },
    {
        'name': 'expand_uk',
        'category': '➕ Additions',
        'description': 'Expands "UK" to "United Kingdom"',
        'example_before': '123 High Street, London SW1A 1AA, UK',
        'example_after': '123 High Street, London SW1A 1AA, United Kingdom',
        'impact': 'Tests country name variation'
    },
    {
        'name': 'abbreviate_uk',
        'category': '📝 Abbreviations',
        'description': 'Abbreviates "United Kingdom" to "UK"',
        'example_before': '123 High Street, London SW1A 1AA, United Kingdom',
        'example_after': '123 High Street, London SW1A 1AA, UK',
        'impact': 'Tests country abbreviation handling'
    }
]

# Group by category
categories = {}
for corruption in corruptions:
    cat = corruption['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(corruption)

# Display by category
for category, items in categories.items():
    print(f"\\n{category}")
    print("-" * 50)
    
    for item in items:
        print(f"\\n• {item['name']}:")
        print(f"  📝 {item['description']}")
        print(f"  📍 BEFORE: {item['example_before']}")
        print(f"  📍 AFTER:  {item['example_after']}")
        print(f"  🎯 Impact: {item['impact']}")

print(f"\\n📊 CORRUPTION SUMMARY:")
print("=" * 25)
print(f"🎯 Total corruption types: {len(corruptions)}")
print(f"🔥 Active corruptions: {len([c for c in corruptions if c['name'] != 'no_change'])}")
print(f"🔒 Control (no_change): 1")

category_counts = {}
for corruption in corruptions:
    cat = corruption['category'].split()[1]  # Get category name without emoji
    category_counts[cat] = category_counts.get(cat, 0) + 1

print(f"\\n📈 CORRUPTION CATEGORIES:")
for category, count in category_counts.items():
    print(f"  {category}: {count} types")

print(f"\\n🎭 CORRUPTION STRATEGY:")
print("=" * 25)
print("✅ Simulates real-world data quality issues")
print("✅ Tests PII detection resilience to formatting variations") 
print("✅ Covers OCR errors, user input mistakes, system inconsistencies")
print("✅ Maintains realistic UK address structure")
print("✅ Enables quantitative corruption impact analysis")
print("✅ Supports algorithmic PII detection benchmarking")

print(f"\\n🔍 USAGE IN TESTING:")
print("=" * 20)
print("• Compare clean vs dirty detection rates")
print("• Identify corruption types that break PII detection")
print("• Benchmark algorithm resilience to data quality issues")
print("• Validate address parsing under various formatting conditions")
print("• Test postcode extraction accuracy across format variations")

print(f"\\n✅ ALL 24 CORRUPTION TYPES DOCUMENTED")
print("🎯 Ready for comprehensive PII detection testing")
