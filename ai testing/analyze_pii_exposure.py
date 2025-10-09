#!/usr/bin/env python3
"""
PII Exposure Analysis Tool
This script analyzes anonymized results to determine if any raw PII data is exposed (partially or fully).
"""

import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Any, Set
import argparse
from datetime import datetime

class PIIExposureAnalyzer:
    def __init__(self):
        self.exposure_levels = {
            'NO_EXPOSURE': 'No PII data exposed',
            'PARTIAL_EXPOSURE': 'Partial PII data exposed',
            'FULL_EXPOSURE': 'Full PII data exposed'
        }
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra spaces and converting to lowercase"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', str(text).strip().lower())
    
    def tokenize_text(self, text: str) -> Set[str]:
        """Tokenize text into words and numbers for partial matching"""
        if not text:
            return set()
        # Extract words, numbers, and special patterns
        tokens = set()
        text_lower = text.lower()
        
        # Extract words (2+ characters)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
        tokens.update(words)
        
        # Extract numbers (2+ digits)
        numbers = re.findall(r'\b\d{2,}\b', text_lower)
        tokens.update(numbers)
        
        # Extract email-like patterns
        emails = re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text_lower)
        tokens.update(emails)
        
        # Extract date-like patterns
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text_lower)
        tokens.update(dates)
        
        # Extract UK postcode-like patterns
        postcodes = re.findall(r'\b[a-zA-Z]{1,2}\d{1,2}[a-zA-Z]?\s?\d[a-zA-Z]{2}\b', text_lower)
        tokens.update([pc.replace(' ', '') for pc in postcodes])
        
        return tokens
    
    def check_exposure_level(self, raw_data: str, anonymized_text: str) -> Tuple[str, List[str]]:
        """Check the exposure level of PII data in anonymized text"""
        if not raw_data or not anonymized_text:
            return 'NO_EXPOSURE', []
        
        raw_normalized = self.normalize_text(raw_data)
        anon_normalized = self.normalize_text(anonymized_text)
        
        exposed_data = []
        
        # Check for full exposure (exact match)
        if raw_normalized in anon_normalized:
            return 'FULL_EXPOSURE', [raw_data]
        
        # Check for partial exposure using tokenization
        raw_tokens = self.tokenize_text(raw_data)
        anon_tokens = self.tokenize_text(anonymized_text)
        
        # Find overlapping tokens
        exposed_tokens = raw_tokens.intersection(anon_tokens)
        
        if exposed_tokens:
            # Filter out very common/short tokens that might be coincidental
            significant_tokens = {
                token for token in exposed_tokens 
                if len(token) >= 3 and not token.isdigit() or (token.isdigit() and len(token) >= 4)
            }
            
            if significant_tokens:
                exposed_data = list(significant_tokens)
                return 'PARTIAL_EXPOSURE', exposed_data
        
        return 'NO_EXPOSURE', []
    
    def analyze_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single record for PII exposure"""
        analysis = {
            'record_id': record.get('record_id', 'unknown'),
            'overall_exposure_level': 'NO_EXPOSURE',
            'exposed_fields': {},
            'exposed_data_summary': [],
            'anonymized_text': '',
            'original_text': '',
            'total_pii_fields': 0,
            'exposed_pii_count': 0
        }
        
        # Get the anonymized text from the record structure
        anonymized_text = ''
        original_text = ''
        
        # Try different possible locations for anonymized text
        if 'fields' in record and 'source' in record['fields']:
            anonymized_text = record['fields']['source'].get('anonymized_text', '')
            original_text = record['fields']['source'].get('original_text', '')
        elif 'analysis' in record:
            anonymized_text = record['analysis'].get('anonymized_text', '')
        
        analysis['anonymized_text'] = anonymized_text
        analysis['original_text'] = original_text
        
        # Get the original PII data from ground_truth_pii
        ground_truth = record.get('ground_truth_pii', {})
        pii_fields = ground_truth.get('fields_used', [])
        pii_values = ground_truth.get('raw_values', [])
        
        if not pii_fields or not pii_values or not anonymized_text:
            return analysis
        
        # Create a dictionary mapping field names to values
        pii_data = dict(zip(pii_fields, pii_values))
        analysis['total_pii_fields'] = len(pii_data)
        max_exposure_level = 'NO_EXPOSURE'
        
        # Check each PII field for exposure
        for field_name, field_value in pii_data.items():
            if field_value and str(field_value).strip():
                exposure_level, exposed_data = self.check_exposure_level(
                    str(field_value), anonymized_text
                )
                
                if exposure_level != 'NO_EXPOSURE':
                    analysis['exposed_fields'][field_name] = {
                        'original_value': str(field_value),
                        'exposure_level': exposure_level,
                        'exposed_data': exposed_data
                    }
                    analysis['exposed_pii_count'] += 1
                    
                    # Update overall exposure level
                    if exposure_level == 'FULL_EXPOSURE':
                        max_exposure_level = 'FULL_EXPOSURE'
                    elif exposure_level == 'PARTIAL_EXPOSURE' and max_exposure_level != 'FULL_EXPOSURE':
                        max_exposure_level = 'PARTIAL_EXPOSURE'
        
        analysis['overall_exposure_level'] = max_exposure_level
        
        # Create summary of all exposed data
        all_exposed = []
        for field_info in analysis['exposed_fields'].values():
            all_exposed.extend(field_info['exposed_data'])
        analysis['exposed_data_summary'] = list(set(all_exposed))
        
        return analysis
    
    def load_results_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load and parse the PII detection results file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different file formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'results' in data:
                return data['results']
            else:
                print(f"Warning: Unexpected file format in {filepath}")
                return []
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []
    
    def analyze_all_records(self, results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all records for PII exposure"""
        summary = {
            'total_records': len(results_data),
            'no_exposure': 0,
            'partial_exposure': 0,
            'full_exposure': 0,
            'exposure_details': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print(f"Analyzing {len(results_data)} records for PII exposure...")
        
        for i, record in enumerate(results_data):
            if i % 100 == 0:
                print(f"Processing record {i+1}/{len(results_data)}")
            
            analysis = self.analyze_record(record)
            summary['exposure_details'].append(analysis)
            
            # Update counters
            exposure_level = analysis['overall_exposure_level']
            if exposure_level == 'NO_EXPOSURE':
                summary['no_exposure'] += 1
            elif exposure_level == 'PARTIAL_EXPOSURE':
                summary['partial_exposure'] += 1
            elif exposure_level == 'FULL_EXPOSURE':
                summary['full_exposure'] += 1
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any]) -> str:
        """Generate a human-readable report"""
        report = []
        report.append("=" * 60)
        report.append("PII EXPOSURE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {summary['analysis_timestamp']}")
        report.append(f"Total Records Analyzed: {summary['total_records']}")
        report.append("")
        
        # Summary statistics
        report.append("EXPOSURE SUMMARY:")
        report.append(f"  No PII Exposed: {summary['no_exposure']} ({summary['no_exposure']/summary['total_records']*100:.1f}%)")
        report.append(f"  Partial PII Exposed: {summary['partial_exposure']} ({summary['partial_exposure']/summary['total_records']*100:.1f}%)")
        report.append(f"  Full PII Exposed: {summary['full_exposure']} ({summary['full_exposure']/summary['total_records']*100:.1f}%)")
        report.append("")
        
        # Detailed exposure analysis
        if summary['partial_exposure'] > 0 or summary['full_exposure'] > 0:
            report.append("DETAILED EXPOSURE ANALYSIS:")
            report.append("-" * 40)
            
            exposure_records = [
                record for record in summary['exposure_details'] 
                if record['overall_exposure_level'] != 'NO_EXPOSURE'
            ]
            
            for record in exposure_records[:10]:  # Show first 10 exposed records
                report.append(f"Record ID: {record['record_id']}")
                report.append(f"  Exposure Level: {record['overall_exposure_level']}")
                report.append(f"  Exposed Fields: {len(record['exposed_fields'])}/{record['total_pii_fields']}")
                
                for field_name, field_info in record['exposed_fields'].items():
                    report.append(f"    - {field_name}: {field_info['exposure_level']}")
                    report.append(f"      Original: {field_info['original_value']}")
                    report.append(f"      Exposed: {field_info['exposed_data']}")
                
                report.append(f"  Anonymized Text: {record['anonymized_text'][:100]}...")
                report.append("")
            
            if len(exposure_records) > 10:
                report.append(f"... and {len(exposure_records) - 10} more records with exposure")
        
        return "\n".join(report)
    
    def save_detailed_analysis(self, summary: Dict[str, Any], output_file: str):
        """Save detailed analysis to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Detailed analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze PII exposure in anonymized results')
    parser.add_argument('--input-file', required=True, help='Path to PII detection results JSON file')
    parser.add_argument('--output-file', help='Path to save detailed analysis JSON (optional)')
    parser.add_argument('--report-file', help='Path to save human-readable report (optional)')
    
    args = parser.parse_args()
    
    analyzer = PIIExposureAnalyzer()
    
    # Load results
    print(f"Loading results from: {args.input_file}")
    results_data = analyzer.load_results_file(args.input_file)
    
    if not results_data:
        print("No data found in the input file.")
        return
    
    # Analyze for exposure
    summary = analyzer.analyze_all_records(results_data)
    
    # Generate and display report
    report = analyzer.generate_report(summary)
    print("\n" + report)
    
    # Save detailed analysis if requested
    if args.output_file:
        analyzer.save_detailed_analysis(summary, args.output_file)
    else:
        # Default output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"pii_exposure_analysis_{timestamp}.json"
        analyzer.save_detailed_analysis(summary, default_output)
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.report_file}")

if __name__ == "__main__":
    main()
