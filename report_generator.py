
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def generate_exposure_report_data(analysis_results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Analyzes PII exposure from batch results and categorizes PII entities.
    - Not Detected: PII in original is not found in anonymized.
    - Partially Exposed: PII is found, but anonymized text is not a simple placeholder (e.g., contains parts of the original).
    - Fully Protected: PII is found, and anonymized text is a standard placeholder (e.g., <PERSON>).
    """
    rows = []
    for record in analysis_results:
        original_pii = record.get('original_pii', [])
        anonymized_pii = record.get('anonymized_pii', [])
        
        # Create a map of anonymized entities for quick lookup
        anon_map = {(p['entity_type'], p['text']): p for p in anonymized_pii}

        for pii in original_pii:
            entity_type = pii['entity_type']
            original_text = pii['text']
            
            # Find a matching anonymized entity
            match = anon_map.get((entity_type, original_text))

            status = ""
            anonymized_text = ""
            if not match:
                status = "Not Detected"
            else:
                anonymized_text = match.get('anonymized_text', '')
                # Simple heuristic for full protection: placeholder is just the entity type in angle brackets
                if anonymized_text.strip() == f"<{entity_type}>":
                    status = "Fully Protected"
                else:
                    status = "Partially Exposed"
            
            rows.append({
                'entity_type': entity_type,
                'original_text': original_text,
                'anonymized_text': anonymized_text,
                'status': status,
                'source_record_id': record.get('id', 'N/A')
            })

    if not rows:
        return {
            'fully_protected': pd.DataFrame(),
            'partially_exposed': pd.DataFrame(),
            'not_detected': pd.DataFrame()
        }

    df = pd.DataFrame(rows)
    
    report_data = {
        'fully_protected': df[df['status'] == 'Fully Protected'],
        'partially_exposed': df[df['status'] == 'Partially Exposed'],
        'not_detected': df[df['status'] == 'Not Detected']
    }
    
    return report_data

def get_latest_results_file(results_path: Path) -> Optional[Path]:
    """Finds the most recently modified JSON results file."""
    json_files = list(results_path.glob("pii_detection_results_*.json"))
    if not json_files:
        return None
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return json_files[0]

def display_exposure_report(results_path: Path):
    """
    Loads the latest batch results, generates the PII exposure report,
    and displays it in the Streamlit UI.
    """
    st.header("PII Exposure Analysis")

    # Find all available result files
    json_files = list(results_path.glob("pii_detection_results_*.json"))
    
    if not json_files:
        st.warning("No PII analysis result files found in the 'results' directory.")
        st.info("Please run a batch analysis first from the 'Interactive Search' page.")
        return

    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Let user select a file
    file_options = [f.name for f in json_files]
    selected_file_name = st.selectbox(
        "Select a results file to analyze:",
        options=file_options,
        index=0,
        help="Choose which batch analysis results to analyze"
    )
    
    # Add a button to trigger the analysis
    if st.button("Generate Report", type="primary"):
        selected_file = results_path / selected_file_name
        
        st.info(f"Loading results from: `{selected_file_name}`")
        

        try:
            with open(selected_file, 'r') as f:
                analysis_results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading or parsing {selected_file_name}: {e}")
            return

        # Accept either a list of records, or a dict with a 'results' key
        if isinstance(analysis_results, dict) and 'results' in analysis_results:
            analysis_results = analysis_results['results']

        if not isinstance(analysis_results, list):
            st.error(f"Loaded data is not a list. Type: {type(analysis_results)}. Check the contents of the file.")
            return
        if not analysis_results or not all(isinstance(r, dict) for r in analysis_results):
            st.warning("Result file is empty or not in the expected format (a list of records).")
            return

        with st.spinner("Generating exposure report..."):
            report_data = generate_exposure_report_data(analysis_results)

        st.success("Report generated successfully!")
        
        st.subheader("Exposure Summary")
        summary_df = pd.DataFrame({
            'Category': ['Fully Protected', 'Partially Exposed', 'Not Detected'],
            'Count': [
                len(report_data['fully_protected']),
                len(report_data['partially_exposed']),
                len(report_data['not_detected'])
            ]
        })
        st.dataframe(summary_df, use_container_width=True)

        # Display details for each category
        st.subheader("Not Detected PII")
        st.dataframe(report_data['not_detected'], use_container_width=True)

        st.subheader("Partially Exposed PII")
        st.dataframe(report_data['partially_exposed'], use_container_width=True)

        st.subheader("Fully Protected PII")
        st.dataframe(report_data['fully_protected'], use_container_width=True)

        # Provide a download link for the not detected PII
        if not report_data['not_detected'].empty:
            csv = report_data['not_detected'][['entity_type', 'original_text', 'source_record_id']].to_csv(index=False)
            st.download_button(
                label="Download Not Detected PII as CSV",
                data=csv,
                file_name='not_detected_pii.csv',
                mime='text/csv',
            )

