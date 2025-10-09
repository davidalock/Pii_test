#!/usr/bin/env python3
"""
Script to analyze the first column of merchant_frame for ATM.csv using Presidio
to determine data types and write results to a new CSV file.
"""

import pandas as pd
import csv
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def analyze_addresses():
    # Initialize Presidio analyzer
    analyzer = AnalyzerEngine()
    
    # Read the CSV file
    input_file = '/Users/davidlock/Downloads/soccer data python/testing poe/merchant_frame for ATM.csv'
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/address_analysis_results.csv'
    
    print("Reading CSV file...")
    df = pd.read_csv(input_file)
    
    # Get the first column (formatted_address)
    first_column = df.iloc[:, 0]  # Get first column regardless of name
    column_name = df.columns[0]
    
    print(f"Analyzing column: '{column_name}'")
    print(f"Total rows to analyze: {len(first_column)}")
    
    results = []
    
    # Analyze each address in the first column
    for idx, address in enumerate(first_column):
        if pd.isna(address):
            # Handle NaN values
            results.append({
                'row_index': idx,
                'original_text': '',
                'detected_entities': 'NaN - No text to analyze',
                'entity_types': '',
                'confidence_scores': ''
            })
            continue
            
        # Convert to string if not already
        address_str = str(address)
        
        # Analyze the address text
        analysis_results = analyzer.analyze(text=address_str, language='en')
        
        # Extract entity information
        entity_types = []
        confidence_scores = []
        detected_entities = []
        
        for result in analysis_results:
            entity_types.append(result.entity_type)
            confidence_scores.append(f"{result.score:.3f}")
            # Extract the actual detected text
            detected_text = address_str[result.start:result.end]
            detected_entities.append(f"{result.entity_type}:{detected_text}")
        
        # Prepare result row
        result_row = {
            'row_index': idx,
            'original_text': address_str,
            'detected_entities': '; '.join(detected_entities) if detected_entities else 'No PII detected',
            'entity_types': '; '.join(entity_types) if entity_types else 'None',
            'confidence_scores': '; '.join(confidence_scores) if confidence_scores else 'N/A'
        }
        
        results.append(result_row)
        
        # Print progress every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    print(f"Saving results to: {output_file}")
    results_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total rows analyzed: {len(results)}")
    
    # Count different entity types found
    all_entity_types = []
    rows_with_pii = 0
    
    for result in results:
        if result['entity_types'] != 'None' and result['entity_types'] != '':
            rows_with_pii += 1
            if result['entity_types'] != 'None':
                all_entity_types.extend(result['entity_types'].split('; '))
    
    print(f"Rows with PII detected: {rows_with_pii}")
    print(f"Rows without PII: {len(results) - rows_with_pii}")
    
    if all_entity_types:
        entity_counts = {}
        for entity in all_entity_types:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        print("\nEntity types found:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type}: {count} occurrences")
    
    print(f"\nResults saved to: {output_file}")
    return results_df

if __name__ == "__main__":
    results = analyze_addresses()
