#!/usr/bin/env python3
"""
Batch Processor for PII Analysis
Provides multithreaded processing capability for large datasets
"""

import concurrent.futures
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional

# Shared core functions to ensure sync with interactive flow
try:
    from pii_core import enhanced_pii_analysis as core_enhanced_pii_analysis, enhanced_anonymization as core_enhanced_anonymization
except Exception:
    core_enhanced_pii_analysis = None
    core_enhanced_anonymization = None

def process_record(record_data, analyzer, selected_columns, selected_entities, 
                  use_anonymization=True, use_verification=True, 
                  batch_single_type_anon=True, pii_analysis_func=None,
                  anonymization_func=None, verification_func=None, config=None):
    """Process a single record for PII detection and anonymization
    
    This function can be called directly or by the multithreaded processor
    """
    if not pii_analysis_func or not anonymization_func or not verification_func:
        raise ValueError("Analysis, anonymization, and verification functions must be provided")
        
    if analyzer is None:
        raise ValueError("Presidio analyzer engine must be provided")
        
    idx, row = record_data
    
    record_results = {
        'record_id': row.get('record_id', row.get('customer_id', f'record_{idx}')),
        'record_index': idx,
        'columns_analyzed': {},
        'summary': {
            'total_presidio': 0,
            'total_uk_patterns': 0,
            'total_missed': 0,
            'total_all': 0
        }
    }
    
    # Analyze selected columns only
    for col in selected_columns:
        if pd.notna(row[col]):
            text_content = str(row[col])
            if len(text_content.strip()) > 0:
                
                # Enhanced analysis with selected entity types
                # Support both injected func (from UI) and shared core default
                if pii_analysis_func is not None:
                    analysis_results = pii_analysis_func(
                        text_content, 
                        analyzer, 
                        config,
                        selected_entities
                    )
                else:
                    if core_enhanced_pii_analysis is None:
                        raise ValueError("No analysis function provided and core_enhanced_pii_analysis unavailable")
                    # Note: core_enhanced_pii_analysis signature: (text, analyzer, selected_entities)
                    analysis_results = core_enhanced_pii_analysis(text_content, analyzer, selected_entities)
                    analysis_results['original_text'] = text_content
                
                # Store the original text in the analysis results
                analysis_results['original_text'] = text_content
                
                # Calculate column totals
                total_presidio = len(analysis_results['presidio_findings'])
                total_uk = len(analysis_results['uk_pattern_findings'])
                total_missed = len(analysis_results['missed_findings'])
                total_all = len(analysis_results['all_findings'])
                
                # Save to record results
                record_results['columns_analyzed'][col] = {
                    'analysis': analysis_results
                }
                
                # Anonymization
                if use_anonymization:
                    if anonymization_func is not None:
                        anonymized_content = anonymization_func(
                            text_content,
                            analysis_results['all_findings'],
                            batch_single_type_anon
                        )
                    else:
                        if core_enhanced_anonymization is None:
                            raise ValueError("No anonymization function provided and core_enhanced_anonymization unavailable")
                        anonymized_content = core_enhanced_anonymization(
                            text_content,
                            analysis_results['all_findings'],
                            batch_single_type_anon
                        )
                    
                    record_results['columns_analyzed'][col]['anonymization'] = {
                        'anonymized_content': anonymized_content
                    }
                    
                    # Verification of anonymization
                    if use_verification:
                        verification_results = verification_func(
                            original_text=text_content,
                            anonymized_text=anonymized_content,
                            column=col
                        )
                        
                        record_results['columns_analyzed'][col]['verification'] = verification_results
                
                # Update record totals
                record_results['summary']['total_presidio'] += total_presidio
                record_results['summary']['total_uk_patterns'] += total_uk
                record_results['summary']['total_missed'] += total_missed
                record_results['summary']['total_all'] += total_all
    
    return record_results

def batch_process_records(df, analyzer, selected_columns, selected_entities, 
                         pii_analysis_func, anonymization_func, verification_func,
                         max_records=None, use_threading=True, num_threads=4,
                         use_anonymization=True, use_verification=True, 
                         batch_single_type_anon=True, config=None,
                         progress_callback=None, status_callback=None):
    """Process multiple records for PII detection and anonymization"""
    all_results = []
    
    # Limit records if specified
    if max_records is not None and max_records > 0:
        df_to_process = df.head(max_records)
    else:
        df_to_process = df
        
    records_to_process = len(df_to_process)
    
    if status_callback:
        status_callback(f"Processing {records_to_process} records...")
    
    # Use multithreading if enabled
    if use_threading and records_to_process > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit processing tasks
            futures = []
            for idx, row in df_to_process.iterrows():
                future = executor.submit(
                    process_record,
                    (idx, row),
                    analyzer,
                    selected_columns,
                    selected_entities,
                    use_anonymization,
                    use_verification,
                    batch_single_type_anon,
                    pii_analysis_func,
                    anonymization_func,
                    verification_func,
                    config
                )
                futures.append(future)
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # Update progress
                    if progress_callback:
                        progress = (i + 1) / records_to_process
                        progress_callback(progress)
                    
                    # Update status
                    if status_callback:
                        status_callback(f"Processed {i + 1}/{records_to_process} records")
                        
                except Exception as e:
                    print(f"Error processing record: {str(e)}")
    else:
        # Process records sequentially
        for idx, row in df_to_process.iterrows():
            try:
                result = process_record(
                    (idx, row),
                    analyzer,
                    selected_columns,
                    selected_entities,
                    use_anonymization,
                    use_verification,
                    batch_single_type_anon,
                    pii_analysis_func,
                    anonymization_func,
                    verification_func,
                    config
                )
                all_results.append(result)
                
                # Update progress
                if progress_callback:
                    progress = (idx + 1) / records_to_process
                    progress_callback(progress)
                
                # Update status
                if status_callback:
                    status_callback(f"Processing record {idx + 1}/{records_to_process}")
                    
            except Exception as e:
                print(f"Error processing record {idx}: {str(e)}")
    
    # Sort results by record index to maintain order
    all_results.sort(key=lambda x: x['record_index'])
    
    return all_results

def export_results(results, filename, format="CSV"):
    """Export analysis results to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{filename}_analysis_{timestamp}"
    
    # Check if results is a dict with 'detailed_results' or just the list of records
    if isinstance(results, dict) and 'detailed_results' in results:
        records = results['detailed_results']
    else:
        records = results
    
    if format == "JSON":
        export_file = f"{base_filename}.json"
        with open(export_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "CSV":
        export_file = f"{base_filename}.csv"
        
        # Flatten results for CSV format
        flattened_data = []
        for record in records:
            for col_name, col_data in record['columns_analyzed'].items():
                # Extract PII findings
                findings = col_data['analysis']['all_findings']
                
                # Get anonymized content if available
                anonymized_text = ''
                if 'anonymization' in col_data and 'anonymized_content' in col_data['anonymization']:
                    anonymized_text = col_data['anonymization']['anonymized_content']
                
                # Get original text
                original_text = col_data['analysis'].get('original_text', '')
                
                if findings:
                    for finding in findings:
                        # For potentially missed PII, check if it appears in anonymized text
                        is_in_anonymized = ''
                        if finding['entity_type'] == 'POTENTIALLY_MISSED_PII' and anonymized_text:
                            is_in_anonymized = 'Yes' if finding['text'] in anonymized_text else 'No'
                        
                        flattened_data.append({
                            'record_id': record['record_id'],
                            'column': col_name,
                            'entity_type': finding['entity_type'],
                            'text': finding['text'],
                            'confidence': finding['confidence'],
                            'source': finding['source'],
                            'start': finding['start'],
                            'end': finding['end'],
                            'original_text': original_text,
                            'anonymized': anonymized_text,
                            'in_anonymized': is_in_anonymized
                        })
                else:
                    # Include record with no findings
                    flattened_data.append({
                        'record_id': record['record_id'],
                        'column': col_name,
                        'entity_type': 'NONE',
                        'text': '',
                        'confidence': 0,
                        'source': '',
                        'start': 0,
                        'end': 0,
                        'original_text': original_text,
                        'anonymized': anonymized_text,
                        'in_anonymized': ''
                    })
        
        # Convert to DataFrame and export
        df = pd.DataFrame(flattened_data)
        df.to_csv(export_file, index=False)
    elif format == "Excel":
        export_file = f"{base_filename}.xlsx"
        
        # Flatten results like for CSV
        flattened_data = []
        for record in records:
            for col_name, col_data in record['columns_analyzed'].items():
                findings = col_data['analysis']['all_findings']
                
                # Get anonymized content if available
                anonymized_text = ''
                if 'anonymization' in col_data and 'anonymized_content' in col_data['anonymization']:
                    anonymized_text = col_data['anonymization']['anonymized_content']
                
                # Get original text
                original_text = col_data['analysis'].get('original_text', '')
                
                if findings:
                    for finding in findings:
                        # For potentially missed PII, check if it appears in anonymized text
                        is_in_anonymized = ''
                        if finding['entity_type'] == 'POTENTIALLY_MISSED_PII' and anonymized_text:
                            is_in_anonymized = 'Yes' if finding['text'] in anonymized_text else 'No'
                        
                        flattened_data.append({
                            'record_id': record['record_id'],
                            'column': col_name,
                            'entity_type': finding['entity_type'],
                            'text': finding['text'],
                            'confidence': finding['confidence'],
                            'source': finding['source'],
                            'start': finding['start'],
                            'end': finding['end'],
                            'original_text': original_text,
                            'anonymized': anonymized_text,
                            'in_anonymized': is_in_anonymized
                        })
                else:
                    flattened_data.append({
                        'record_id': record['record_id'],
                        'column': col_name,
                        'entity_type': 'NONE',
                        'text': '',
                        'confidence': 0,
                        'source': '',
                        'start': 0,
                        'end': 0,
                        'original_text': original_text,
                        'anonymized': anonymized_text,
                        'in_anonymized': ''
                    })
        
        # Export to Excel
        df = pd.DataFrame(flattened_data)
        df.to_excel(export_file, index=False)
    
    return export_file
