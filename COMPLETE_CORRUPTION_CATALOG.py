#!/usr/bin/env python3
"""
COMPLETE CORRUPTION CATALOG
All corruption types implemented for dirty data testing
"""

print("🧪 COMPLETE CORRUPTION CATALOG - ALL IMPLEMENTED TYPES")
print("=" * 60)

# Original Address-focused Corruptions (24 types)
print("\\n📍 ORIGINAL ADDRESS CORRUPTIONS (24 types)")
print("=" * 45)

original_corruptions = [
    {
        'name': 'no_change',
        'category': 'Control',
        'description': 'No corruption applied',
        'example': 'Same → Same',
        'frequency': 'High (13.5%)'
    },
    {
        'name': 'uppercase',
        'category': 'Case Changes',
        'description': 'Convert to UPPERCASE',
        'example': 'London → LONDON',
        'frequency': 'Medium (3.1%)'
    },
    {
        'name': 'lowercase', 
        'category': 'Case Changes',
        'description': 'Convert to lowercase',
        'example': 'London → london',
        'frequency': 'Medium (4.2%)'
    },
    {
        'name': 'title_case',
        'category': 'Case Changes',
        'description': 'Convert To Title Case',
        'example': 'london SW1A → London Sw1a',
        'frequency': 'Medium (4.5%)'
    },
    {
        'name': 'extra_spaces',
        'category': 'Spacing',
        'description': 'Add spaces at start/end',
        'example': 'Address → " Address "',
        'frequency': 'Medium (4.3%)'
    },
    {
        'name': 'spaced_commas',
        'category': 'Spacing',
        'description': 'Add spaces around commas',
        'example': 'A,B → A , B',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'normalized_spaces',
        'category': 'Spacing',
        'description': 'Remove double spaces',
        'example': 'A  B → A B',
        'frequency': 'Medium (4.3%)'
    },
    {
        'name': 'removed_commas',
        'category': 'Spacing',
        'description': 'Remove all commas',
        'example': 'A, B, C → A B C',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'number_change',
        'category': 'Numeric',
        'description': 'Change house numbers by ±1-3',
        'example': '123 → 126',
        'frequency': 'Medium (4.8%)'
    },
    {
        'name': 'street_abbreviation',
        'category': 'Abbreviations',
        'description': 'Street → St',
        'example': 'High Street → High St',
        'frequency': 'Medium (4.8%)'
    },
    {
        'name': 'road_abbreviation',
        'category': 'Abbreviations', 
        'description': 'Road → Rd',
        'example': 'High Road → High Rd',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'avenue_abbreviation',
        'category': 'Abbreviations',
        'description': 'Avenue → Ave',
        'example': 'Park Avenue → Park Ave',
        'frequency': 'Medium (3.5%)'
    },
    {
        'name': 'street_expansion',
        'category': 'Expansions',
        'description': 'St → Street',
        'example': 'High St → High Street',
        'frequency': 'Medium (3.1%)'
    },
    {
        'name': 'road_expansion',
        'category': 'Expansions',
        'description': 'Rd → Road',
        'example': 'High Rd → High Road',
        'frequency': 'Medium (3.4%)'
    },
    {
        'name': 'postcode_no_space',
        'category': 'Postcode',
        'description': 'Remove postcode space',
        'example': 'SW1A 1AA → SW1A1AA',
        'frequency': 'Medium (4.3%)'
    },
    {
        'name': 'postcode_add_space',
        'category': 'Postcode',
        'description': 'Add postcode space',
        'example': 'SW1A1AA → SW1A 1AA',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'O_to_0',
        'category': 'OCR Errors',
        'description': 'Letter O → digit 0',
        'example': 'OX1 → 0X1',
        'frequency': 'Low (2.2%)'
    },
    {
        'name': '0_to_O',
        'category': 'OCR Errors',
        'description': 'Digit 0 → letter O',
        'example': 'SW10 → SW1O',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'I_to_1',
        'category': 'OCR Errors',
        'description': 'Letter I → digit 1',
        'example': 'SW1I → SW11',
        'frequency': 'High (5.4%)'
    },
    {
        'name': '1_to_I',
        'category': 'OCR Errors',
        'description': 'Digit 1 → letter I',
        'example': '123 → I23',
        'frequency': 'Medium (3.4%)'
    },
    {
        'name': 'remove_last_part',
        'category': 'Truncation',
        'description': 'Remove final comma part',
        'example': 'A, B, C → A, B',
        'frequency': 'Medium (3.2%)'
    },
    {
        'name': 'add_england',
        'category': 'Additions',
        'description': 'Add ", England"',
        'example': 'London, UK → London, UK, England',
        'frequency': 'Medium (3.8%)'
    },
    {
        'name': 'expand_uk',
        'category': 'Country',
        'description': 'UK → United Kingdom',
        'example': 'London, UK → London, United Kingdom',
        'frequency': 'Medium (5.1%)'
    },
    {
        'name': 'abbreviate_uk',
        'category': 'Country',
        'description': 'United Kingdom → UK',
        'example': 'London, United Kingdom → London, UK',
        'frequency': 'Medium (3.1%)'
    }
]

# Enhanced Field-Specific Corruptions (60+ types)  
print("\\n🎯 NEW FIELD-SPECIFIC CORRUPTIONS (60+ types)")
print("=" * 50)

enhanced_corruptions = [
    {
        'field': 'Place Names',
        'types': [
            'London → Londan, Londin, Londn',
            'Manchester → Machester, Manchestor',  
            'Birmingham → Birmingam, Burmingham',
            'Liverpool → Liverpol, Liverpul',
            'Edinburgh → Edinburg, Edinborough',
            'Glasgow → Glasgo, Glascow',
            'Cardiff → Cardif, Carrdiff',
            'Belfast → Belast, Belfest',
            'Newcastle → Newcastel, New Castle',
            'Leeds → Leds, Leedes',
            'Sheffield → Shefield, Sheffild', 
            'Bristol → Bristal, Bristoll',
            '+ 15 more major UK cities/areas'
        ],
        'coverage': '17.2% of dirty records',
        'impact': 'Tests location name recognition resilience'
    },
    {
        'field': 'Card Numbers (PAN)',
        'types': [
            'Alpha chars: 1234 → 123O, 123l, 123I',
            'Truncation: 1234567890123456 → 123456789012345',
            'Extra digits: 1234567890123456 → 12345678901234569',
            'Formatting: 1234567890123456 → 1234 5678 9012 3456',
            'Hyphens: 1234567890123456 → 1234-5678-9012-3456',
            'OCR errors: 1234567890123456 → I234567890I23456',
            'Missing zeros: 0123456789012345 → 123456789012345'
        ],
        'coverage': '100% of dirty records',
        'impact': 'Tests financial data validation'
    },
    {
        'field': 'Personal Names',
        'types': [
            'Common misspellings: John → Jon, Jhon',
            'Michael → Micheal, Mikael',
            'David → Davd, Daivd',
            'James → Jmes, Jamies',
            'Sarah → Sara, Sarrah',
            'Jennifer → Jenifer, Jeniffer',
            'Doubled letters: Amy → Ammy',
            'Missing letters: Robert → Robrt',
            'Swapped letters: Mark → Mrak',
            'i/y substitution: Emily → Emyly',
            'ph/f substitution: Christopher → Cristopher',
            '+ 50+ common name variations'
        ],
        'coverage': '83.5% of dirty records', 
        'impact': 'Tests name recognition flexibility'
    },
    {
        'field': 'Email Domains',
        'types': [
            'gmail.com → gmai.com, gmial.com',
            'yahoo.com → yaho.com, yahooo.com',
            'hotmail.com → hotmai.com, htomail.com',
            'outlook.com → outlok.com, outloook.com',
            '.com → .co (missing m)',
            '.com → .cm (missing o)',
            'Missing dot: domain.com → domaincom',
            'Wrong TLD: .co.uk → .com'
        ],
        'coverage': '74.6% of dirty records',
        'impact': 'Tests email validation robustness'
    },
    {
        'field': 'Phone Numbers',
        'types': [
            'Wrong country code: +44 → +441',
            'Duplicate: +447123456789 → +44+447123456789',
            'Missing country: +447123456789 → 7123456789',
            'Truncation: 07123456789 → 0712345678',
            'Extra digits: 07123456789 → 071234567891',
            'Formatting: 07123456789 → 07-123-456-789',
            'OCR: 07123456789 → O7I23456789',
            'Spacing: 07123456789 → 07 123 456 789'
        ],
        'coverage': '92.3% of dirty records',
        'impact': 'Tests phone number parsing flexibility'
    },
    {
        'field': 'Date Formats',
        'types': [
            'Separators: 12/03/1990 → 12-03-1990',
            'No separators: 12/03/1990 → 12031990',
            'Reversed: 12/03/1990 → 1990/03/12',
            'Missing zeros: 02/03/1990 → 2/3/1990',
            'Wrong format: DD/MM/YYYY → MM/DD/YYYY',
            'Two-digit year: 12/03/1990 → 12/03/90',
            'Dots: 12/03/1990 → 12.03.1990'
        ],
        'coverage': '90.9% of dirty records',
        'impact': 'Tests date parsing robustness'
    },
    {
        'field': 'National Insurance Numbers',
        'types': [
            'No spaces: AB 12 34 56 C → AB123456C',
            'Wrong spacing: AB123456C → AB 12 3456 C',  
            'Missing suffix: AB123456C → AB123456',
            'Case changes: AB123456C → ab123456c',
            'OCR errors: AB123456C → A8123456C',
            'Hyphens: AB 12 34 56 C → AB-12-34-56-C'
        ],
        'coverage': '0% (not corrupted in current run)',
        'impact': 'Tests NINO format validation'
    }
]

# Display original corruptions by category
categories = {}
for corruption in original_corruptions:
    cat = corruption['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(corruption)

for category, items in categories.items():
    print(f"\\n🏷️ {category.upper()}:")
    for item in items:
        print(f"  • {item['name']:<20}: {item['description']}")
        print(f"    Example: {item['example']}")
        print(f"    Frequency: {item['frequency']}")

# Display enhanced corruptions
print("\\n\\n🎯 ENHANCED FIELD-SPECIFIC CORRUPTIONS:")
print("=" * 45)

for corruption_set in enhanced_corruptions:
    print(f"\\n🔧 {corruption_set['field'].upper()}:")
    print(f"   Coverage: {corruption_set['coverage']}")
    print(f"   Impact: {corruption_set['impact']}")
    print("   Types:")
    for corruption_type in corruption_set['types'][:5]:  # Show first 5
        print(f"     • {corruption_type}")
    if len(corruption_set['types']) > 5:
        print(f"     • ... and {len(corruption_set['types']) - 5} more types")

# Summary statistics
print(f"\\n📊 COMPREHENSIVE CORRUPTION STATISTICS:")
print("=" * 45)

total_original = len(original_corruptions)
total_enhanced_fields = len(enhanced_corruptions)
total_enhanced_types = sum(len(c['types']) for c in enhanced_corruptions)

print(f"📍 Original Address Corruptions:")
print(f"   • Total Types: {total_original}")
print(f"   • Categories: {len(categories)}")
print(f"   • Focus: Address formatting and structure")

print(f"\\n🎯 Enhanced Field-Specific Corruptions:")
print(f"   • Fields Covered: {total_enhanced_fields}")
print(f"   • Total Corruption Types: {total_enhanced_types}+")
print(f"   • Focus: Realistic data quality issues per field")

print(f"\\n🏆 TOTAL CORRUPTION CAPABILITY:")
print(f"   • Combined Types: {total_original + total_enhanced_types}+ different corruptions")
print(f"   • Coverage: 8 different PII field types")
print(f"   • Tracking: Individual corruption type per field per record")
print(f"   • Realism: Based on real-world data quality issues")

print(f"\\n💡 CORRUPTION STRATEGY BENEFITS:")
print("=" * 35)
print("✅ Simulates OCR scanning errors")
print("✅ Models user input mistakes") 
print("✅ Replicates system integration issues")
print("✅ Tests format variation tolerance")
print("✅ Validates parsing robustness")
print("✅ Benchmarks detection accuracy")
print("✅ Enables algorithmic comparison")
print("✅ Provides quantitative corruption impact analysis")

print(f"\\n🎯 READY FOR PRODUCTION PII TESTING")
print("🔍 Test suite capable of comprehensive corruption analysis")
print("📊 Granular tracking enables detailed performance metrics") 
print("🚀 Realistic data quality simulation for robust validation")
