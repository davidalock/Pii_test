#!/usr/bin/env python3
"""
Simple Presidio Anonymizer Script
=================================

A standalone script that uses only base Presidio to:
1. Read a CSV file
2. Analyze the 'source' field for PII entities
3. Anonymize detected PII
4. Save results to a new CSV file

Dependencies: presidio-analyzer, presidio-anonymizer, pandas
"""

import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Presidio imports
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

def setup_presidio():
    """Initialize Presidio analyzer and anonymizer engines"""
    print("Setting up Presidio engines...")
    
    # Initialize analyzer with default recognizers - cache to avoid repeated fetching
    analyzer = AnalyzerEngine()
    
    # Initialize anonymizer
    anonymizer = AnonymizerEngine()
    
    print(f"Available recognizers: {[recognizer.name for recognizer in analyzer.get_recognizers()]}")
    print(f"Supported entities: {analyzer.get_supported_entities()}")
    
    return analyzer, anonymizer

# Global cache for analyzer engines to prevent repeated initialization
_analyzer_cache = None
_anonymizer_cache = None

def get_cached_engines():
    """Get cached analyzer and anonymizer engines"""
    global _analyzer_cache, _anonymizer_cache
    
    if _analyzer_cache is None or _anonymizer_cache is None:
        _analyzer_cache, _anonymizer_cache = setup_presidio()
    
    return _analyzer_cache, _anonymizer_cache

def analyze_text(analyzer: AnalyzerEngine, text: str, language: str = "en") -> List[RecognizerResult]:
    """Analyze text for PII entities"""
    if not text or pd.isna(text):
        return []
    
    # Analyze text
    results = analyzer.analyze(
        text=str(text),
        language=language,
        score_threshold=0.35  # Minimum confidence threshold
    )
    
    return results

def anonymize_text(anonymizer: AnonymizerEngine, text: str, analyzer_results: List[RecognizerResult]) -> str:
    """Anonymize text based on analyzer results"""
    if not text or pd.isna(text) or not analyzer_results:
        return str(text) if text else ""
    
    # Anonymize the text
    anonymized_result = anonymizer.anonymize(
        text=str(text),
        analyzer_results=analyzer_results
    )
    
    return anonymized_result.text

def process_csv(input_file: str, output_file: str = None, source_field: str = "source", max_records: int = None):
    """Process CSV file and anonymize the specified field"""
    
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Set output file name if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.stem + f"_anonymized_{timestamp}.csv"
    
    print(f"Processing CSV file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target field: {source_field}")
    if max_records:
        print(f"Max records to process: {max_records}")
    
    # Setup Presidio using cached engines
    analyzer, anonymizer = get_cached_engines()
    
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        # Limit records if specified
        if max_records and max_records < len(df):
            df = df.head(max_records)
            print(f"Limited to first {max_records} records (original file had {original_count} records)")
        
        print(f"Processing {len(df)} records from CSV")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")
    
    # Validate source field exists
    if source_field not in df.columns:
        raise ValueError(f"Field '{source_field}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Process records
    results = []
    total_entities_found = 0
    
    print(f"\nProcessing {len(df)} records...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx} records...")
        
        # Get original text
        original_text = row[source_field]
        
        if pd.isna(original_text) or not str(original_text).strip():
            # Handle empty/null values
            result = {
                'record_id': idx,
                'original_text': "",
                'anonymized_text': "",
                'entities_found': 0,
                'entities': []
            }
        else:
            # Analyze for PII
            pii_results = analyze_text(analyzer, str(original_text))
            
            # Anonymize
            anonymized_text = anonymize_text(anonymizer, str(original_text), pii_results)
            
            # Prepare entity information
            entities_info = []
            for entity in pii_results:
                entities_info.append({
                    'entity_type': entity.entity_type,
                    'text': original_text[entity.start:entity.end],
                    'start': entity.start,
                    'end': entity.end,
                    'confidence': entity.score
                })
            
            result = {
                'record_id': idx,
                'original_text': str(original_text),
                'anonymized_text': anonymized_text,
                'entities_found': len(pii_results),
                'entities': entities_info
            }
            
            total_entities_found += len(pii_results)
        
        results.append(result)
    
    # Create output DataFrame
    output_data = []
    
    for result in results:
        # Start with original row data
        output_row = df.iloc[result['record_id']].to_dict()
        
        # Add anonymized field
        output_row[f'{source_field}_anonymized'] = result['anonymized_text']
        
        # Add entity information
        output_row['entities_found'] = result['entities_found']
        output_row['entity_types'] = ', '.join(set(e['entity_type'] for e in result['entities']))
        
        # Add detailed entity info as JSON string
        if result['entities']:
            entity_details = []
            for entity in result['entities']:
                entity_details.append(f"{entity['entity_type']}:{entity['text']}({entity['confidence']:.2f})")
            output_row['entity_details'] = '; '.join(entity_details)
        else:
            output_row['entity_details'] = ""
        
        output_data.append(output_row)
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    
    # Print summary
    records_with_pii = sum(1 for r in results if r['entities_found'] > 0)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total records processed: {len(results)}")
    print(f"Records with PII found: {records_with_pii}")
    print(f"Total entities detected: {total_entities_found}")
    print(f"Output saved to: {output_file}")
    
    # Entity type summary
    entity_types = {}
    for result in results:
        for entity in result['entities']:
            entity_type = entity['entity_type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    if entity_types:
        print(f"\nEntity types found:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"  {entity_type}: {count}")
    
    return output_file, results

def main():
    """Main function to handle command line execution"""
    
    print("Simple Presidio Anonymizer")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python simple_presidio_anonymizer.py <input_csv> [output_csv] [source_field] [max_records]")
        print("\nArguments:")
        print("  input_csv    : Path to input CSV file")
        print("  output_csv   : Path to output CSV file (optional)")
        print("  source_field : Name of field to anonymize (default: 'source')")
        print("  max_records  : Maximum number of records to process (optional)")
        print("\nExample:")
        print("  python simple_presidio_anonymizer.py data.csv")
        print("  python simple_presidio_anonymizer.py data.csv anonymized_data.csv")
        print("  python simple_presidio_anonymizer.py data.csv anonymized_data.csv content")
        print("  python simple_presidio_anonymizer.py data.csv anonymized_data.csv source 100")
        sys.exit(1)
    
    # Parse arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    source_field = sys.argv[3] if len(sys.argv) > 3 else "source"
    max_records = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    try:
        output_path, results = process_csv(input_file, output_file, source_field, max_records)
        print(f"\nSuccess! Anonymized data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
