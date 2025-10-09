#!/usr/bin/env python3
"""
Enhanced Recognizer Analysis - Show overlapping detections from all engines
This tool runs all recognizers separately and shows which ones detect the same entities
"""

import streamlit as st
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

@dataclass
class EntityDetection:
    """Single entity detection with full metadata"""
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    source: str
    recognizer_name: str = ""
    
    def overlaps_with(self, other: 'EntityDetection') -> bool:
        """Check if this detection overlaps with another"""
        return not (self.end <= other.start or other.end <= self.start)
    
    def get_key(self) -> str:
        """Get a key for grouping overlapping entities"""
        return f"{self.entity_type}_{self.start}_{self.end}_{self.text.lower()}"

class EnhancedRecognizerAnalysis:
    """Enhanced analysis that runs each recognizer separately"""
    
    def __init__(self):
        self.detections = []
        
    def analyze_with_all_recognizers(self, text: str, config: Dict = None) -> Dict:
        """Run analysis with all recognizers separately to show overlaps"""
        
        results = {
            'text': text,
            'detections': [],
            'overlapping_groups': [],
            'recognizer_summary': {},
            'entity_coverage_matrix': {}
        }
        
        try:
            # Import required modules
            from pii_analysis_cli import load_presidio_engines
            import tempfile
            
            # Load configuration
            if not config:
                config = {'use_presidio': True, 'use_transformers': True, 'use_ollama': False, 'use_uk_patterns': True}
            
            # Initialize engines (allow None when Presidio is disabled)
            analyzer, anonymizer = load_presidio_engines(config)
            if not analyzer and config.get('use_presidio', True):
                st.error("❌ Failed to initialize PII analysis engines")
                return results
            
            all_detections = []
            
            # 1. Run INDIVIDUAL Presidio recognizers
            if config.get('use_presidio', True) and analyzer is not None:
                presidio_detections = self._run_presidio_recognizers(text, analyzer)
                all_detections.extend(presidio_detections)
            
            # 2. Run Transformer-based recognizers 
            if config.get('use_transformers', False):
                # If Presidio is enabled and transformer_models provided, optionally rebuild analyzer;
                # when Presidio is disabled, skip rebuilding and use standalone HF recognizer path.
                try:
                    tm = config.get('transformer_models')
                    if tm and config.get('use_presidio', True):
                        from transformer_integration import get_analyzer_with_transformers
                        from uk_recognizers import register_uk_recognizers
                        analyzer = get_analyzer_with_transformers(
                            sensitivity=config.get('sensitivity', 0.35),
                            uk_specific=True,
                            transformer_model=tm[0],
                            transformer_models=tm,
                            add_custom_recognizers=register_uk_recognizers,
                            use_custom_datetime=config.get('use_custom_datetime', True)
                        )
                except Exception as e:
                    logger.warning(f"Could not rebuild analyzer with transformer models for enhanced analysis: {e}")
                transformer_detections = self._run_transformer_recognizers(text, analyzer)
                all_detections.extend(transformer_detections)
            
            # 3. Run Ollama analysis
            if config.get('use_ollama', False):
                ollama_detections = self._run_ollama_recognizers(text, config)
                all_detections.extend(ollama_detections)
            
            # 4. Run UK Pattern recognizers
            if config.get('use_uk_patterns', True):
                uk_detections = self._run_uk_pattern_recognizers(text)
                all_detections.extend(uk_detections)
            
            # Process results
            results['detections'] = all_detections
            results['overlapping_groups'] = self._find_overlapping_groups(all_detections)
            results['recognizer_summary'] = self._create_recognizer_summary(all_detections)
            results['entity_coverage_matrix'] = self._create_coverage_matrix(all_detections)
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            st.error(f"❌ Analysis failed: {str(e)}")
        
        return results
    
    def _run_presidio_recognizers(self, text: str, analyzer) -> List[EntityDetection]:
        """Run Presidio analysis with all built-in recognizers"""
        detections = []
        
        try:
            # Get allow_list from analyzer config if available
            allow_list = None
            if hasattr(analyzer, '_config') and analyzer._config.get('allow_list'):
                allow_list = analyzer._config['allow_list']
            
            # Run full Presidio analysis to get all results
            results = analyzer.analyze(
                text=text, 
                language='en',
                allow_list=allow_list,
                score_threshold=getattr(analyzer, '_sensitivity', 0.35)
            )
            
            # Post-process to filter out allow_listed terms if Presidio's allow_list didn't work
            if allow_list and results:
                filtered_results = []
                allow_list_lower = [term.lower() for term in allow_list]
                logger.info(f"Enhanced analysis: Applying allow_list filter with {len(allow_list)} terms")
                
                for result in results:
                    detected_text = text[result.start:result.end].lower()
                    # Check if the detected text matches any term in the allow_list
                    if detected_text not in allow_list_lower:
                        filtered_results.append(result)
                    else:
                        logger.info(f"Enhanced analysis: Filtered out '{detected_text}' from {result.entity_type} detection (in allow_list)")
                
                logger.info(f"Enhanced analysis: Filtered results: {len(results)} -> {len(filtered_results)}")
                results = filtered_results
            
            # Convert each result to our EntityDetection format
            for result in results:
                # For now, just use a generic recognizer name since we're getting all results together
                # The goal is to show that Presidio detected these entities
                recognizer_name = f"Presidio-{result.entity_type}"
                
                detection = EntityDetection(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    text=text[result.start:result.end],
                    confidence=result.score,
                    source="Presidio",
                    recognizer_name=recognizer_name
                )
                detections.append(detection)
                    
        except Exception as e:
            logger.error(f"Failed to run Presidio recognizers: {e}")
        
        return detections
    
    def _run_transformer_recognizers(self, text: str, analyzer) -> List[EntityDetection]:
        """Run transformer-based analysis if available"""
        detections = []
        
        try:
            logger.info("Starting transformer recognizers...")
            
            # Check if transformers are available
            try:
                from hf_transformer_recognizer import create_transformer_recognizer
                import transformers
                logger.info(f"Transformers version: {transformers.__version__}")
            except ImportError as e:
                logger.info(f"Transformers not available: {e}")
                return detections
            
            logger.info("Creating transformer recognizer...")
            
            # Create a transformer recognizer
            transformer_recognizer = create_transformer_recognizer()
            
            if not transformer_recognizer:
                logger.error("Failed to create transformer recognizer")
                return detections
                
            logger.info(f"Created transformer recognizer: {transformer_recognizer.name}")
            logger.info(f"Analyzing text: '{text}'")
            
            # Run the transformer analysis
            transformer_results = transformer_recognizer.analyze(text)
            
            logger.info(f"Transformer returned {len(transformer_results)} results")
            
            # Convert transformer results to our EntityDetection format
            for result in transformer_results:
                detection = EntityDetection(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    text=text[result.start:result.end],
                    confidence=result.score,
                    source="Transformers",
                    recognizer_name=f"Transformer-{result.entity_type}"
                )
                detections.append(detection)
                logger.info(f"Added detection: {result.entity_type} = '{text[result.start:result.end]}' (score: {result.score})")
                    
        except ImportError as e:
            logger.info(f"Transformers not available: {e}")
        except Exception as e:
            logger.error(f"Failed to run Transformer recognizers: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Transformer analysis complete, returning {len(detections)} detections")
        return detections
    
    def _run_ollama_recognizers(self, text: str, config: Dict) -> List[EntityDetection]:
        """Run Ollama-based analysis"""
        detections = []
        
        try:
            from ollama_integration import analyze_text_with_ollama
            
            ollama_config = config.get('ollama_extractor_config', {})
            
            ollama_findings = analyze_text_with_ollama(
                text=text,
                model_name=ollama_config.get('model_name', 'mistral'),
                prompt_template=ollama_config.get('prompt_template', None),
                entity_types=ollama_config.get('entity_types', None),
                _ui_debug=True
            )
            
            for finding in ollama_findings:
                detection = EntityDetection(
                    entity_type=finding.get('entity_type', 'UNKNOWN'),
                    start=finding.get('start', 0),
                    end=finding.get('end', 0),
                    text=finding.get('text', ''),
                    confidence=float(finding.get('confidence', 0)),
                    source="Ollama",
                    recognizer_name=f"Ollama-{ollama_config.get('model_name', 'mistral')}"
                )
                detections.append(detection)
                
        except Exception as e:
            logger.error(f"Failed to run Ollama recognizers: {e}")
        
        return detections
    
    def _run_uk_pattern_recognizers(self, text: str) -> List[EntityDetection]:
        """Run UK-specific pattern recognizers"""
        detections = []
        
        try:
            from pii_analysis_cli import detect_uk_specific_pii
            
            uk_findings = detect_uk_specific_pii(text)
            
            for finding in uk_findings:
                detection = EntityDetection(
                    entity_type=finding.get('entity_type', 'UNKNOWN'),
                    start=finding.get('start', 0),
                    end=finding.get('end', 0),
                    text=finding.get('text', ''),
                    confidence=float(finding.get('confidence', 0)),
                    source="UK Patterns",
                    recognizer_name="UK Pattern Recognizer"
                )
                detections.append(detection)
                
        except Exception as e:
            logger.error(f"Failed to run UK pattern recognizers: {e}")
        
        return detections
    
    def _find_overlapping_groups(self, detections: List[EntityDetection]) -> List[Dict]:
        """Find groups of overlapping entity detections"""
        overlapping_groups = []
        processed_indices = set()
        
        for i, detection1 in enumerate(detections):
            if i in processed_indices:
                continue
                
            # Find all detections that overlap with this one
            group = [detection1]
            group_indices = {i}
            
            for j, detection2 in enumerate(detections):
                if j <= i or j in processed_indices:
                    continue
                    
                # Check if detection2 overlaps with any detection in the current group
                if any(detection2.overlaps_with(d) for d in group):
                    group.append(detection2)
                    group_indices.add(j)
            
            if len(group) > 1:
                # Create overlapping group info
                group_info = {
                    'entity_type': group[0].entity_type,  # Use first detection's type
                    'text': group[0].text,  # Use first detection's text 
                    'start': min(d.start for d in group),
                    'end': max(d.end for d in group),
                    'detections': group,
                    'recognizer_count': len(group),
                    'recognizers': [d.source for d in group],
                    'recognizer_names': [d.recognizer_name for d in group]
                }
                overlapping_groups.append(group_info)
                processed_indices.update(group_indices)
        
        return overlapping_groups
    
    def _create_recognizer_summary(self, detections: List[EntityDetection]) -> Dict:
        """Create summary of recognizer performance"""
        summary = defaultdict(lambda: {'count': 0, 'entity_types': set(), 'confidence_avg': 0})
        
        for detection in detections:
            source = detection.source
            summary[source]['count'] += 1
            summary[source]['entity_types'].add(detection.entity_type)
            summary[source]['confidence_avg'] += detection.confidence
        
        # Calculate averages
        for source in summary:
            if summary[source]['count'] > 0:
                summary[source]['confidence_avg'] /= summary[source]['count']
            summary[source]['entity_types'] = list(summary[source]['entity_types'])
        
        return dict(summary)
    
    def _create_coverage_matrix(self, detections: List[EntityDetection]) -> Dict:
        """Create matrix showing which recognizers detect which entity types"""
        matrix = defaultdict(lambda: defaultdict(int))
        
        for detection in detections:
            matrix[detection.entity_type][detection.source] += 1
        
        return {entity_type: dict(sources) for entity_type, sources in matrix.items()}

def display_enhanced_analysis_results(results: Dict):
    """Display comprehensive analysis results in Streamlit"""
    
    if not results or not results.get('detections'):
        st.info("No PII entities detected.")
        return
    
    st.subheader("🔍 Comprehensive Recognizer Analysis")
    
    # Summary metrics
    detections = results['detections']
    overlapping_groups = results['overlapping_groups']
    recognizer_summary = results['recognizer_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Detections", len(detections))
    with col2:
        st.metric("Overlapping Groups", len(overlapping_groups))
    with col3:
        st.metric("Unique Entities", len(set(d.get_key() for d in detections)))
    with col4:
        recognizer_count = len(recognizer_summary)
        st.metric("Active Recognizers", recognizer_count)
    
    # Recognizer performance summary
    st.subheader("📊 Recognizer Performance")
    summary_data = []
    for source, stats in recognizer_summary.items():
        summary_data.append({
            'Recognizer': source,
            'Detections': stats['count'],
            'Entity Types': len(stats['entity_types']),
            'Avg Confidence': f"{stats['confidence_avg']:.3f}",
            'Detected Types': ', '.join(stats['entity_types'][:3]) + ('...' if len(stats['entity_types']) > 3 else '')
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, width="stretch", hide_index=True)
    
    # Overlapping detections (most important!)
    if overlapping_groups:
        with st.expander("🎯 Overlapping Entity Detections", expanded=False):
            st.markdown("*These entities were detected by **multiple recognizers***")

            for i, group in enumerate(overlapping_groups):
                with st.expander(f"🔄 {group['entity_type']}: '{group['text']}' - Detected by {group['recognizer_count']} recognizers"):

                    # Show which recognizers detected this entity
                    recognizer_data = []
                    for detection in group['detections']:
                        recognizer_data.append({
                            'Recognizer': detection.source,
                            'Specific Name': detection.recognizer_name,
                            'Confidence': f"{detection.confidence:.3f}",
                            'Position': f"{detection.start}-{detection.end}",
                            'Entity Type': detection.entity_type
                        })

                    df_group = pd.DataFrame(recognizer_data)
                    st.dataframe(df_group, width="stretch", hide_index=True)
    
    # All detections table
    st.subheader("📋 All Detections")
    detection_data = []
    for detection in detections:
        detection_data.append({
            'Entity Type': detection.entity_type,
            'Text': detection.text,
            'Source': detection.source,
            'Recognizer': detection.recognizer_name,
            'Confidence': f"{detection.confidence:.3f}",
            'Position': f"{detection.start}-{detection.end}"
        })
    
    if detection_data:
        df_all = pd.DataFrame(detection_data)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            selected_sources = st.multiselect(
                "Filter by Source:",
                options=df_all['Source'].unique(),
                default=df_all['Source'].unique()
            )
        with col2:
            selected_entities = st.multiselect(
                "Filter by Entity Type:",
                options=df_all['Entity Type'].unique(),
                default=df_all['Entity Type'].unique()
            )
        
        # Apply filters
        filtered_df = df_all[
            (df_all['Source'].isin(selected_sources)) &
            (df_all['Entity Type'].isin(selected_entities))
        ]
        
        st.dataframe(filtered_df, width="stretch", hide_index=True)
    
    # Entity coverage matrix
    if results['entity_coverage_matrix']:
        with st.expander("🗂️ Entity Type Coverage Matrix", expanded=False):
            matrix_data = []
            for entity_type, sources in results['entity_coverage_matrix'].items():
                row = {'Entity Type': entity_type}
                for source in recognizer_summary.keys():
                    row[source] = sources.get(source, 0)
                matrix_data.append(row)

            if matrix_data:
                df_matrix = pd.DataFrame(matrix_data)
                st.dataframe(df_matrix, width="stretch", hide_index=True)
