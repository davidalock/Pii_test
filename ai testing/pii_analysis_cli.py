#!/usr/bin/env python3
"""
PII Analysis CLI
A command-line interface for PII detection and anonymization using Microsoft Presidio
Features:
- YAML configuration file support
- Command-line parameter overrides
- Multithreaded processing for large datasets
- Support for various input/output formats
- Comprehensive PII detection including UK-specific patterns
"""

import argparse
import yaml
import os
import sys
import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pii_analysis')

# Check for presidio availability
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer import RecognizerResult
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.error("Microsoft Presidio is not available. Install with: pip install presidio-analyzer presidio-anonymizer")

# Check for transformers availability
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    logger.info(f"Hugging Face Transformers available (version: {transformers.__version__})")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Hugging Face Transformers not available. Install with: pip install transformers torch")

# Import batch processor functionality
from batch_processor import batch_process_records, export_results

# Import UK-specific recognizers
from uk_recognizers import register_uk_recognizers

# Define enhanced PII functions directly here instead of importing from UI scripts
def detect_uk_specific_pii(text: str) -> List[Dict]:
    """Detect UK-specific PII patterns that Presidio might miss"""
    uk_patterns = []
    
    # UK postcodes (AB12 3CD or AB1 2CD format)
    uk_postcode_pattern = r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b'
    for match in re.finditer(uk_postcode_pattern, text, re.IGNORECASE):
        uk_patterns.append({
            'entity_type': 'UK_POSTCODE',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK sort codes (XX-XX-XX format)
    sort_code_pattern = r'\b\d{2}-\d{2}-\d{2}\b'
    for match in re.finditer(sort_code_pattern, text):
        uk_patterns.append({
            'entity_type': 'UK_SORT_CODE',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK NHS numbers (XXX XXX XXXX format)
    nhs_pattern = r'\b\d{3}\s\d{3}\s\d{4}\b'
    for match in re.finditer(nhs_pattern, text):
        uk_patterns.append({
            'entity_type': 'UK_NHS_NUMBER',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK Monetary values (£XX,XXX.XX format)
    money_pattern = r'£\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
    for match in re.finditer(money_pattern, text):
        uk_patterns.append({
            'entity_type': 'UK_MONETARY_VALUE',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK Bank account numbers (8 digits)
    bank_pattern = r'\b\d{8}\b'
    for match in re.finditer(bank_pattern, text):
        uk_patterns.append({
            'entity_type': 'UK_BANK_ACCOUNT',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK Passport numbers (9 digits)
    passport_pattern = r'\b\d{9}\b'
    for match in re.finditer(passport_pattern, text):
        uk_patterns.append({
            'entity_type': 'UK_PASSPORT_NUMBER',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    # UK National Insurance numbers (XX XX XX XX X format)
    ni_pattern = r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b'
    for match in re.finditer(ni_pattern, text, re.IGNORECASE):
        uk_patterns.append({
            'entity_type': 'UK_NI_NUMBER',
            'start': match.start(),
            'end': match.end(),
            'text': match.group(),
            'confidence': 0.9,
            'source': 'Custom UK Pattern'
        })
    
    return uk_patterns

def detect_potentially_missed_pii(text: str) -> List[Dict]:
    """Detect potentially missed PII patterns using simple heuristics"""
    
    # Basic word-by-word analysis for potential PII
    words = re.findall(r'\b\w+\b', text)
    
    # Word context patterns (list of patterns to check for potential PII)
    context_patterns = [
        r'\b(?:username|user name|login)\b.{1,20}',
        r'\b(?:password|pwd|passcode)\b.{1,20}',
        r'\b(?:account|acct).{1,20}\d',
        r'\b(?:ref|reference).{1,20}\d',
        r'\b(?:confirm|verification).{1,30}',
        r'\b(?:code|pin).{1,15}\d',
        r'\b(?:id|identification).{1,20}'
    ]
    
    findings = []
    
    # Check for potential missed PII
    for pattern in context_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Extract the value after the context keyword
            found_text = match.group().strip()
            
            # Only add if not obviously a sentence
            if len(found_text.split()) < 4:
                findings.append({
                    'entity_type': 'POTENTIALLY_MISSED_PII',
                    'start': match.start(),
                    'end': match.end(),
                    'text': found_text,
                    'confidence': 0.6,
                    'source': 'Missed Pattern Detection'
                })
    
    # Check each word for potential PII based on simple heuristics
    for word in words:
        # Skip very short words and common stopwords
        if len(word) < 4 or word.lower() in {'this', 'that', 'with', 'from', 'have', 'your'}:
            continue
            
        # Check for potential proper nouns (capitalized words not at beginning of sentence)
        if word[0].isupper() and len(word) > 4:
            findings.append({
                'entity_type': 'POTENTIALLY_MISSED_PII',
                'start': text.find(word),
                'end': text.find(word) + len(word),
                'text': word,
                'confidence': 0.6,
                'source': 'Missed Pattern Detection'
            })
    
    return findings

def enhanced_pii_analysis(text: str, analyzer, config=None, selected_entities=None) -> Dict:
    """Perform enhanced PII analysis using selected engines only.
    Works even when Presidio is disabled (analyzer can be None) by running other enabled engines.
    """

    # Always initialize result buckets; do NOT early-return when analyzer is None.
    results = {}
    
    # Initialize all result categories
    results['presidio_findings'] = []
    results['transformer_findings'] = []
    results['uk_pattern_findings'] = []
    results['missed_findings'] = []
    results['ollama_findings'] = []
    
    # Track execution time for each analyzer component
    timings: Dict[str, float] = {
        'ollama': 0.0,
        'core_analyzer': 0.0,
        'uk_patterns': 0.0,
        'missed_patterns': 0.0,
    }

    # Determine which engines to use based on configuration
    use_presidio = config.get('use_presidio', True)  # Default to True if not specified
    use_transformers = config.get('use_transformers', False)
    use_ollama = config.get('use_ollama', False)
    use_uk_patterns = config.get('use_uk_patterns', True)  # Default to True if not specified
    
    # Note: Each engine (Presidio, Transformers, Ollama, UK patterns) can be independently enabled/disabled
    # They are additive - enabling one doesn't disable others unless explicitly configured
    
    # Use Ollama for entity extraction if enabled
    if use_ollama:
        try:
            from ollama_integration import analyze_text_with_ollama

            ollama_config = config.get('ollama_extractor_config', {})

            # Enable UI debug mode if we're in interactive mode
            ui_debug = config.get('__interactive_mode__', False)

            _t0 = time.perf_counter()
            ollama_findings = analyze_text_with_ollama(
                text=text,
                model_name=ollama_config.get('model_name', 'mistral'),
                prompt_template=ollama_config.get('prompt_template', None),
                entity_types=ollama_config.get('entity_types', None),
                _ui_debug=ui_debug
            )
            timings['ollama'] = time.perf_counter() - _t0

            results['ollama_findings'] = ollama_findings

            # If only Ollama is requested, return just Ollama results
            if not use_presidio and not use_transformers and not use_uk_patterns:
                results['timings'] = timings
                results['all_findings'] = ollama_findings
                return results

        except Exception as e:
            logger.error(f"Failed to use Ollama for entity extraction: {e}")
            if not use_presidio and not use_transformers:
                # If only Ollama was requested and it failed, return empty results
                logger.warning("Only Ollama was requested but it failed. Returning empty results.")
                results['all_findings'] = []
                results['timings'] = timings
                return results
            logger.warning("Ollama failed, continuing with other enabled engines")

    # Run analyzer-backed analysis only if we have an analyzer and it's requested
    analyzer_results = []
    if analyzer is not None and (use_presidio or use_transformers):
        # Get allow_list from analyzer config if available
        allow_list = None
        try:
            if config and hasattr(analyzer, '_config') and analyzer._config.get('allow_list'):
                allow_list = analyzer._config['allow_list']
        except Exception:
            allow_list = None

        try:
            _t0 = time.perf_counter()
            analyzer_results = analyzer.analyze(
                text=text,
                language='en',
                entities=selected_entities if selected_entities else None,
                allow_list=allow_list,
                score_threshold=getattr(analyzer, '_sensitivity', 0.35)  # Use stored sensitivity or default
            )
            timings['core_analyzer'] = time.perf_counter() - _t0
        except Exception as e:
            logger.warning(f"Analyzer analyze() failed: {e}")
            analyzer_results = []
            timings['core_analyzer'] = 0.0
        
        # Post-process to filter out allow_listed terms if Presidio's allow_list didn't work
        if allow_list and analyzer_results:
            filtered_results = []
            allow_list_lower = [term.lower() for term in allow_list]
            logger.info(f"Applying post-processing filter with {len(allow_list)} terms: {allow_list[:5]}...")
            
            for result in analyzer_results:
                detected_text = text[result.start:result.end].lower()
                # Check if the detected text matches any term in the allow_list
                if detected_text not in allow_list_lower:
                    filtered_results.append(result)
                else:
                    logger.info(f"Filtered out '{detected_text}' from {result.entity_type} detection (in allow_list)")
            
            logger.info(f"Filtered results: {len(analyzer_results)} -> {len(filtered_results)}")
            analyzer_results = filtered_results
    
    # Convert Presidio/transformer analyzer results to our format
    for finding in analyzer_results:
        # Check if this is a transformer-based finding
        recognition_source = "Presidio"
        if hasattr(finding, 'recognition_metadata') and finding.recognition_metadata:
            source = finding.recognition_metadata.get('source', '')
            if source and 'transformer' in source:
                recognition_source = source
        # Try to extract recognizer name and pattern details when available
        recognizer_name = None
        pattern_name = None
        
        # First check if this is a transformer detection - DON'T set default yet
        # Determine the recognizer name based on the NLP provider only as a fallback
        default_recognizer_name = None
        nlp_provider = getattr(analyzer, 'nlp_provider', 'spacy')
        default_recognizer_name = 'SpacyRecognizer' if nlp_provider == 'spacy' else nlp_provider
            
        try:
            ex = getattr(finding, 'analysis_explanation', None)
            if ex:
                # Attempt direct attributes first
                recognizer_name_from_ex = getattr(ex, 'recognizer', None) or getattr(ex, 'recognizer_name', None)
                if recognizer_name_from_ex:
                    recognizer_name = recognizer_name_from_ex
                pattern_name = getattr(ex, 'pattern_name', None)
                # If explanation can be converted to dict, try that
                if hasattr(ex, 'to_dict'):
                    try:
                        exd = ex.to_dict()
                        recognizer_name_from_ex = exd.get('recognizer') or exd.get('recognizer_name')
                        if recognizer_name_from_ex:
                            recognizer_name = recognizer_name_from_ex
                        pattern_name = pattern_name or exd.get('pattern_name')
                    except Exception:
                        pass
                elif isinstance(ex, dict):
                    recognizer_name_from_ex = ex.get('recognizer') or ex.get('recognizer_name')
                    if recognizer_name_from_ex:
                        recognizer_name = recognizer_name_from_ex
                    pattern_name = pattern_name or ex.get('pattern_name')
        except Exception:
            pass
        
        # Fallbacks - check for transformer FIRST before using default
        if not recognizer_name:
            # Use recognition source if it's a transformer hit
            if recognition_source and 'transformer' in recognition_source:
                recognizer_name = recognition_source
            else:
                # Check metadata for recognizer info
                try:
                    if hasattr(finding, 'recognition_metadata') and finding.recognition_metadata:
                        recognizer_name = finding.recognition_metadata.get('recognizer_name') or finding.recognition_metadata.get('recognizer')
                except Exception:
                    pass
                # Final fallback to default
                if not recognizer_name:
                    recognizer_name = default_recognizer_name
                
        result_dict = {
            'entity_type': finding.entity_type,
            'start': finding.start,
            'end': finding.end,
            'text': text[finding.start:finding.end],
            'confidence': finding.score,
            'source': recognition_source,
            'recognizer': recognizer_name or ''
        }
        if pattern_name:
            result_dict['pattern'] = pattern_name
        
        # Add to appropriate category based on source and configuration
        if 'transformer' in recognition_source and use_transformers:
            results['transformer_findings'].append(result_dict)
        elif 'transformer' not in recognition_source and use_presidio:
            results['presidio_findings'].append(result_dict)
    
    # UK-specific PII detection (only if enabled)
    if use_uk_patterns:
        _t0 = time.perf_counter()
        results['uk_pattern_findings'] = detect_uk_specific_pii(text)
        timings['uk_patterns'] = time.perf_counter() - _t0
    
    # Potentially missed PII detection (only if enabled)
    if use_uk_patterns:  # Group with UK patterns for now
        _t0 = time.perf_counter()
        results['missed_findings'] = detect_potentially_missed_pii(text)
        timings['missed_patterns'] = time.perf_counter() - _t0
    
    # Combine findings from enabled engines only
    all_findings = []
    if use_presidio:
        all_findings.extend(results['presidio_findings'])
    if use_transformers:
        all_findings.extend(results['transformer_findings'])
    if use_ollama:
        all_findings.extend(results['ollama_findings'])
    if use_uk_patterns:
        all_findings.extend(results['uk_pattern_findings'])
        all_findings.extend(results['missed_findings'])
    
    results['all_findings'] = all_findings
    
    # Add summary statistics
    results['summary'] = {
        'total_presidio': len(results['presidio_findings']),
        'total_uk_patterns': len(results['uk_pattern_findings']),
        'total_missed': len(results['missed_findings']),
        'total_transformer': len(results['transformer_findings']),
        'total_ollama': len(results['ollama_findings']),
        'total_all': len(results['all_findings'])
    }

    results['timings'] = timings
    
    return results

def enhanced_anonymization(text: str, findings: List[Dict], single_type_overlaps=True) -> str:
    """Perform enhanced anonymization using only the highest confidence PII entity"""
    
    if not findings:
        return text
    
    # Only process the entity with the highest confidence
    highest_conf_finding = max(findings, key=lambda x: x['confidence']) if findings else None
    
    if highest_conf_finding:
        start = highest_conf_finding['start']
        end = highest_conf_finding['end']
        entity_type = highest_conf_finding['entity_type']
        
        # Generate a suitable replacement based on entity type
        if entity_type == 'PERSON' or entity_type == 'POTENTIALLY_MISSED_PII':
            replacement = '[PERSON]'
        elif entity_type == 'EMAIL_ADDRESS':
            replacement = '[EMAIL]'
        elif entity_type == 'PHONE_NUMBER':
            replacement = '[PHONE]'
        elif entity_type == 'URL':
            replacement = '[URL]'
        elif entity_type == 'LOCATION' or entity_type == 'ADDRESS':
            replacement = '[ADDRESS]'
        elif entity_type == 'UK_NHS_NUMBER':
            replacement = '[NHS_NUMBER]'
        elif entity_type == 'UK_NATIONAL_INSURANCE_NUMBER' or entity_type == 'UK_NI_NUMBER':
            replacement = '[NI_NUMBER]'
        elif entity_type == 'CREDIT_CARD':
            replacement = '[CREDIT_CARD]'
        elif entity_type == 'DATE_TIME':
            replacement = '[DATE]'
        elif entity_type == 'US_SSN':
            replacement = '[SSN]'
        elif entity_type == 'IBAN_CODE':
            replacement = '[IBAN]'
        elif entity_type == 'IP_ADDRESS':
            replacement = '[IP]'
        elif entity_type == 'UK_BANK_ACCOUNT':
            replacement = '[BANK_ACCOUNT]'
        elif entity_type == 'UK_PASSPORT_NUMBER':
            replacement = '[PASSPORT]'
        elif entity_type == 'UK_POSTCODE':
            replacement = '[POSTCODE]'
        elif entity_type == 'UK_SORT_CODE':
            replacement = '[SORT_CODE]'
        elif entity_type == 'UK_MONETARY_VALUE':
            replacement = '[AMOUNT]'
        else:
            replacement = f'[{entity_type}]'
        
        # Replace the finding with the anonymized version
        anonymized_text = text[:start] + replacement + text[end:]
        
        return anonymized_text
    
    return text

def verify_pii_removal(original_text: str, anonymized_text: str, column: str = '') -> Dict:
    """Verify that PII has been properly anonymized"""
    original_record = {'text': original_text}
    if column:
        original_record[column] = original_text
        
    verification_results = {
        'success': True,
        'risk_level': 'LOW',
        'recommendations': []
    }
    
    # Basic verification - check if the anonymized text still contains parts of the original text
    if column:
        if column in original_record and original_record[column]:
            original_text = str(original_record[column])
            
            # Skip very short values
            if len(original_text) < 4:
                return verification_results
                
            # Check for exact matches of words or phrases
            words = original_text.split()
            for word in words:
                if len(word) > 3 and word in anonymized_text:
                    verification_results['success'] = False
                    verification_results['risk_level'] = 'HIGH'
                    verification_results['recommendations'].append(
                        f"PII data from column '{column}' was not properly anonymized"
                    )
                    break
    
    return verification_results

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def load_presidio_engines(config: Optional[Dict[str, Any]] = None):
    """Initialize Presidio engines, prioritizing UI/session config over YAML."""
    if not PRESIDIO_AVAILABLE:
        logger.error("Microsoft Presidio is not installed. Please install with: pip install presidio-analyzer presidio-anonymizer")
        return None, None

    if config and config.get('use_presidio') is False:
        logger.info("use_presidio=False -> Skipping Presidio engine initialization")
        return None, None

    try:
        # Determine NLP engine configuration, with clear priority.
        # Priority 1: Direct command from UI/session config.
        # Priority 2: Fallback to analyzers_config.yaml.
        # Priority 3: Default to spaCy.
        
        nlp_settings: Dict[str, Any] = {
            'auto_download': True,
            'spacy_model': 'en_core_web_sm',
            'spacy_lang_code': 'en',
        }

        # 1. Check for UI/session config first
        if config:
            requested_provider = str(config.get('__presidio_nlp_provider__', 'spacy')).lower()
            if requested_provider != 'spacy':
                logger.info(f"UI requested unsupported NLP provider '{requested_provider}'. Using spaCy instead.")
            nlp_settings['auto_download'] = bool(config.get('__presidio_nlp_auto_download__', nlp_settings['auto_download']))
            nlp_settings['spacy_model'] = str(config.get('__presidio_spacy_model__', nlp_settings['spacy_model']))
            nlp_settings['spacy_lang_code'] = str(config.get('__presidio_spacy_lang__', nlp_settings['spacy_lang_code']))
        else:
            # 2. Fallback to YAML configuration
            if os.path.exists('analyzers_config.yaml'):
                try:
                    with open('analyzers_config.yaml', 'r') as f:
                        full_config = yaml.safe_load(f)
                        analyzer_config = full_config.get('analyzers', {}).get('presidio', {})
                        if isinstance(analyzer_config.get('nlp_engine'), dict):
                            yaml_engine = analyzer_config['nlp_engine']
                            requested_provider = str(yaml_engine.get('provider') or yaml_engine.get('nlp_engine_name') or 'spacy').lower()
                            if requested_provider != 'spacy':
                                logger.info(f"Ignoring unsupported NLP provider '{requested_provider}' from analyzers_config.yaml. Using spaCy.")
                            nlp_settings['auto_download'] = bool(yaml_engine.get('auto_download', nlp_settings['auto_download']))
                            if 'spacy_model' in yaml_engine:
                                nlp_settings['spacy_model'] = str(yaml_engine.get('spacy_model') or nlp_settings['spacy_model'])
                            if 'spacy_lang_code' in yaml_engine:
                                nlp_settings['spacy_lang_code'] = str(yaml_engine.get('spacy_lang_code') or nlp_settings['spacy_lang_code'])
                            yaml_models = yaml_engine.get('models')
                            if isinstance(yaml_models, list) and yaml_models:
                                first_model = yaml_models[0]
                                if isinstance(first_model, dict):
                                    nlp_settings['spacy_model'] = str(first_model.get('model_name') or nlp_settings['spacy_model'])
                                    nlp_settings['spacy_lang_code'] = str(first_model.get('lang_code') or nlp_settings['spacy_lang_code'])
                except Exception as e:
                    logger.warning(f"Could not load or parse analyzers_config.yaml, will use defaults. Error: {e}")
            else:
                logger.info("No UI config or YAML file found. Using default spaCy NLP engine.")

        auto_download = bool(nlp_settings.get('auto_download', True))
        spacy_model = nlp_settings.get('spacy_model', 'en_core_web_sm')
        spacy_lang_code = nlp_settings.get('spacy_lang_code', 'en')

        if auto_download:
            try:
                import spacy
                try:
                    spacy.load(spacy_model)
                except OSError:
                    from spacy.cli import download as spacy_download
                    logger.info(f"Downloading spaCy model '{spacy_model}'...")
                    spacy_download(spacy_model)
            except Exception as e:
                logger.warning(f"Failed to ensure spaCy model '{spacy_model}' is available: {e}")

        configuration: Dict[str, Any] = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": spacy_lang_code, "model_name": spacy_model}]
        }

        # Create NLP engine provider
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        analyzer.nlp_provider = 'spacy'  # Store the provider name for downstream logic
        
        # --- The rest of the function for loading recognizers, etc. ---
        
        # Load general analyzer settings from YAML if available
        analyzer_config = {}
        if os.path.exists('analyzers_config.yaml'):
            try:
                with open('analyzers_config.yaml', 'r') as f:
                    full_config = yaml.safe_load(f)
                    analyzer_config = full_config.get('analyzers', {}).get('presidio', {})
            except Exception:
                pass # Ignore errors, just won't have the config

        # Store sensitivity and allow_list for later use
        sensitivity = analyzer_config.get('sensitivity_threshold', 0.35)
        if config and 'sensitivity' in config:
            sensitivity = config.get('sensitivity')
            
        analyzer._sensitivity = sensitivity
        
        # Store allow_list configuration 
        analyzer._config = {}
        
        # Look for allow_list in multiple locations in the config
        allow_list = None
        if analyzer_config.get('allow_list'):
            allow_list = analyzer_config['allow_list']
        elif config and config.get('allow_list'):
            allow_list = config['allow_list']
        
        if allow_list:
            analyzer._config['allow_list'] = allow_list
            logger.info(f"Loaded allow_list with {len(allow_list)} terms.")
        else:
            logger.warning("No allow_list found in configuration")
        
        # Determine feature flags from config/session
        built_in_only = bool(config.get('__built_in_only__')) if config else False
        use_yaml_patterns = False if built_in_only else bool(config.get('__use_yaml_patterns__', True))
        use_code_recognizers = False if built_in_only else bool(config.get('__use_code_recognizers__', True))
        yaml_entity_subset = set(config.get('__yaml_entity_subset__', []) or [])
        code_entity_subset = set(config.get('__code_entity_subset__', []) or [])

        # Optional: fine-grained control to disable/allow built-in entities BEFORE adding YAML/code recognizers
        try:
            disable_builtin_entities = set((config or {}).get('__disable_builtin_entities__', []) or [])
            enable_only_builtin_entities = set((config or {}).get('__enable_only_builtin_entities__', []) or [])
            if disable_builtin_entities or enable_only_builtin_entities:
                new_recognizers = []
                for r in list(analyzer.registry.recognizers):
                    ents = list(getattr(r, 'supported_entities', []) or [])
                    # If enable-only list provided, intersect; drop recognizer if empty
                    if enable_only_builtin_entities:
                        keep = [e for e in ents if e in enable_only_builtin_entities]
                        if keep:
                            try:
                                r.supported_entities = keep
                            except Exception:
                                pass
                            # Also apply disable list after intersection
                            if disable_builtin_entities:
                                keep = [e for e in keep if e not in disable_builtin_entities]
                                if keep:
                                    try:
                                        r.supported_entities = keep
                                    except Exception:
                                        pass
                                    new_recognizers.append(r)
                                # else: drop as no entities left
                            else:
                                new_recognizers.append(r)
                        # else: drop as no intersection with allow-only set
                    else:
                        # No allow-only; apply disable list only
                        if disable_builtin_entities:
                            keep = [e for e in ents if e not in disable_builtin_entities]
                            if keep:
                                try:
                                    r.supported_entities = keep
                                except Exception:
                                    pass
                                new_recognizers.append(r)
                            # else: drop recognizer entirely
                        else:
                            new_recognizers.append(r)
                analyzer.registry.recognizers = new_recognizers
                try:
                    logger.info(
                        f"Applied built-in entity controls. Disabled: {sorted(disable_builtin_entities) if disable_builtin_entities else []}; "
                        f"Allow-only: {sorted(enable_only_builtin_entities) if enable_only_builtin_entities else []}. "
                        f"Remaining built-in recognizers: {len(new_recognizers)}"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to apply built-in recognizer controls: {e}")

        # Load YAML-based recognizers (unless disabled)
        # Allow overriding the recognizers YAML path through config
        recognizers_file = str(config.get('__yaml_recognizers_path__', "presidio_recognizers_config.yaml")) if config else "presidio_recognizers_config.yaml"
        if use_yaml_patterns:
            try:
                use_custom_datetime = config.get('use_custom_datetime', True) if config else True
                if use_custom_datetime:
                    existing_recognizers = [r for r in analyzer.registry.recognizers if r.__class__.__name__ != 'DateRecognizer']
                    analyzer.registry.recognizers = existing_recognizers
                    logger.info("Removed built-in DateRecognizer to use custom version")
                else:
                    logger.info("Keeping built-in DateRecognizer (custom version disabled)")

                analyzer.registry.add_recognizers_from_yaml(yml_path=recognizers_file)
                logger.info(f"✅ Successfully loaded no-code recognizers from {recognizers_file}")

                # If user limited YAML entities, filter recognizers whose supported_entities are fully outside subset
                if yaml_entity_subset:
                    filtered = []
                    for r in analyzer.registry.recognizers:
                        if hasattr(r, 'supported_entities'):
                            if any(ent in yaml_entity_subset for ent in r.supported_entities):
                                # Optionally narrow recognizer entities to intersection
                                r.supported_entities = [ent for ent in r.supported_entities if ent in yaml_entity_subset]
                                filtered.append(r)
                        else:
                            filtered.append(r)
                    analyzer.registry.recognizers = filtered
                    logger.info(f"Applied YAML entity subset filter: {sorted(yaml_entity_subset)}")

                # Debug
                recognizers = analyzer.registry.recognizers
                supported_entities = set()
                for recognizer in recognizers:
                    supported_entities.update(getattr(recognizer, 'supported_entities', []))
                logger.info(f"📊 Total recognizers loaded (post-YAML filtering): {len(recognizers)}")
                logger.info(f"🏷️  Supported entity types: {sorted(supported_entities)}")
            except Exception as e:
                logger.warning(f"Could not load YAML recognizers: {e}")
                logger.info("Falling back to built-in recognizers")
        else:
            logger.info("Skipping YAML pattern recognizers (disabled)")

        # Register UK / custom code recognizers unless disabled or built-in only
        if use_code_recognizers:
            register_uk_recognizers(analyzer)
            # Filter code recognizers if subset provided
            if code_entity_subset:
                for r in analyzer.registry.recognizers:
                    if hasattr(r, 'supported_entities'):
                        if any(ent in code_entity_subset for ent in r.supported_entities):
                            r.supported_entities = [ent for ent in r.supported_entities if ent in code_entity_subset]
                logger.info(f"Applied code recognizer entity subset: {sorted(code_entity_subset)}")
        else:
            logger.info("Skipping custom code recognizers (disabled or built-in-only mode)")
        
        # Add transformer-based recognizer if available
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to import the transformer recognizer
                from hf_transformer_recognizer import TransformerEntityRecognizer
                # Import is successful, continue with logic
                logger.info("Transformer recognizer module available")
            except ImportError:
                logger.warning("Transformer recognizer module not available")
        
        anonymizer = AnonymizerEngine()
        
        return analyzer, anonymizer
    except Exception as e:
        logger.error(f"Error initializing Presidio engines: {e}", exc_info=True)
        return None, None

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="PII Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python pii_analysis_cli.py --input-file customer_data.csv
  
  # Analyze only specific fields and limit to 100 records
  python pii_analysis_cli.py --input-file customer_data.csv --fields name email address --max-records 100
  
  # Use a configuration file
  python pii_analysis_cli.py --config config.yaml
  
  # Override configuration settings
  python pii_analysis_cli.py --config config.yaml --threads 8 --format csv
        """
    )
    
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--input-file', help='Path to input file')
    parser.add_argument('--fields', nargs='+', help='Fields to process')
    parser.add_argument('--entities', nargs='+', 
                        help='Entities to detect (PERSON, EMAIL_ADDRESS, etc.)')
    parser.add_argument('--max-records', type=int,
                        help='Maximum number of records to process')
    parser.add_argument('--threads', type=int,
                        help='Number of threads to use')
    parser.add_argument('--no-threading', action='store_true',
                        help='Disable multithreading')
    parser.add_argument('--no-anonymization', action='store_true',
                        help='Disable anonymization')
    parser.add_argument('--no-verification', action='store_true',
                        help='Disable verification of anonymization')
    parser.add_argument('--format', choices=['json', 'csv', 'excel'], default='csv',
                        help='Output format (json, csv, or excel). Default is csv.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--transformer-model', 
                        help='Hugging Face transformer model to use for enhanced entity recognition')
    parser.add_argument('--transformer-models', nargs='+',
                        help='Space-separated list of Hugging Face transformer models to use (multi-transformer)')
    parser.add_argument('--no-transformers', action='store_true',
                        help='Disable transformer-based entity recognition even if available')
    
    # Ollama integration options
    parser.add_argument('--use-ollama', action='store_true',
                        help='Enable Ollama-based entity recognition')
    parser.add_argument('--ollama-model', default='mistral:7b-instruct',
                        help='Ollama model to use (default: mistral:7b-instruct)')
    parser.add_argument('--ollama-url', default='http://localhost:11434/api/generate',
                        help='Ollama API URL')
    
    return parser.parse_args()

def main():
    """Main CLI function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.input_file:
        config['input_file'] = args.input_file
    if args.fields:
        config['fields'] = args.fields
    if args.entities:
        config['entities'] = args.entities
    if args.max_records is not None:
        config['max_records'] = args.max_records
    if args.threads:
        config['num_threads'] = args.threads
    if args.no_threading:
        config['use_threading'] = False
    if args.no_anonymization:
        config['perform_anonymization'] = False
    if args.no_verification:
        config['verify_anonymization'] = False
    if args.format:
        config['export_format'] = args.format
    if args.transformer_model:
        config['transformer_model'] = args.transformer_model
    if args.transformer_models:
        # Multi-model takes precedence over single
        config['transformer_models'] = args.transformer_models
        config['use_transformers'] = TRANSFORMERS_AVAILABLE
    if args.no_transformers:
        config['use_transformers'] = False
    else:
        config['use_transformers'] = TRANSFORMERS_AVAILABLE
        
    # Ollama options
    if args.use_ollama:
        config['use_ollama'] = True
        config['ollama_extractor_config'] = {
            'model_name': args.ollama_model,
            'api_url': args.ollama_url,
            'temperature': 0.1,
            'entity_types': ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'LOCATION', 'ORGANIZATION'],
            'prompt_template': None  # Use default
        }
    else:
        config['use_ollama'] = False
    
    # Validate required parameters
    if 'input_file' not in config:
        logger.error("Input file is required. Use --input-file or specify in config file.")
        sys.exit(1)
    
    # Set default values if not specified
    config.setdefault('fields', [])
    config.setdefault('entities', [])
    config.setdefault('max_records', None)
    config.setdefault('use_threading', True)
    config.setdefault('num_threads', 4)
    config.setdefault('perform_anonymization', True)
    config.setdefault('verify_anonymization', True)
    config.setdefault('single_type_anonymization', True)
    config.setdefault('export_format', 'csv')  # Default to CSV
    
    try:
        # Initialize Presidio engines
        logger.info("Initializing Presidio engines...")
        analyzer, anonymizer = load_presidio_engines(config)
        
        if not analyzer or not anonymizer:
            logger.error("Failed to initialize Presidio engines.")
            sys.exit(1)
            
        # Setup transformer recognizer if enabled
        if config.get('use_transformers', False) and TRANSFORMERS_AVAILABLE:
            try:
                from transformer_integration import get_analyzer_with_transformers
                
                transformer_model = config.get('transformer_model', 'dslim/bert-base-NER')
                transformer_models = config.get('transformer_models')
                if transformer_models:
                    logger.info(f"Initializing transformer-based entity recognition with models: {transformer_models}")
                else:
                    logger.info(f"Initializing transformer-based entity recognition with model: {transformer_model}")
                
                # Create an enhanced analyzer with transformer support
                analyzer = get_analyzer_with_transformers(
                    sensitivity=config.get('sensitivity', 0.35),  # Get from config directly
                    uk_specific=True,
                    transformer_model=transformer_model,
                    transformer_models=transformer_models,
                    add_custom_recognizers=register_uk_recognizers,
                    use_custom_datetime=config.get('use_custom_datetime', True)
                )
                
                logger.info("Successfully initialized transformer-based entity recognition")
            except Exception as e:
                logger.error(f"Failed to initialize transformer-based entity recognition: {e}")
                logger.warning("Continuing with standard Presidio analyzer")
        
    # Setup Ollama integration if enabled
        elif config.get('use_ollama', False):
            try:
                
                ollama_model = config.get('ollama_model', 'mistral')
                ollama_url = config.get('ollama_url', 'http://localhost:11434/api/generate')
                
                logger.info(f"Initializing Ollama-based entity recognition with model: {ollama_model}")
                
                # Get prompt template from config if available
                prompt_template = config.get('ollama_prompt_template', None)
                entity_types = config.get('ollama_entity_types', None)
                
                # Store Ollama configuration in the config dictionary for later use
                config['ollama_extractor_config'] = {
                    'model_name': ollama_model,
                    'api_url': ollama_url,
                    'prompt_template': prompt_template,
                    'entity_types': entity_types
                }
                
                logger.info("Successfully initialized Ollama-based entity recognition")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama-based entity recognition: {e}")
                logger.warning("Continuing with standard Presidio analyzer")
        
        # Load data
        input_file = config['input_file']
        logger.info(f"Loading data from {input_file}...")
        
        # Determine file type
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = pd.read_excel(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            logger.error("Unsupported file format. Use CSV, Excel, or JSON.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(df)} records")
        
        # Process parameters
        fields_to_process = config['fields']
        selected_entities = config['entities']
        max_records = config['max_records']
        use_threading = config['use_threading']
        num_threads = config['num_threads']
        perform_anonymization = config['perform_anonymization']
        verify_anonymization = config['verify_anonymization']
        single_type_anonymization = config['single_type_anonymization']
        
        # Auto-detect fields if not specified
        if not fields_to_process:
            logger.info("No fields specified, auto-detecting text fields...")
            # Auto-detect text fields (string/object columns)
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            fields_to_process = text_columns
            logger.info(f"Auto-detected fields: {', '.join(fields_to_process)}")
        
        # Define progress callback
        def update_progress(progress):
            """Update progress indicator"""
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + ' ' * (bar_length - filled_length)
            sys.stdout.write(f"\rProgress: {progress * 100:.1f}%")
            sys.stdout.flush()
        
        # Define status callback
        def update_status(status):
            """Update status message"""
            logger.debug(status)
        
        # Logging information about the processing
        logger.info(f"Processing {max_records if max_records else len(df)} records with {len(fields_to_process)} fields")
        if use_threading:
            logger.info(f"Using multithreaded processing with {num_threads} threads")
        else:
            logger.info("Using single-threaded processing")
        
        # Start timing
        start_time = datetime.now()
        
        # Process the data
        results = batch_process_records(
            df=df,
            max_records=max_records,
            selected_columns=fields_to_process,
            analyzer=analyzer,
            selected_entities=selected_entities,
            pii_analysis_func=enhanced_pii_analysis,
            anonymization_func=enhanced_anonymization,
            verification_func=verify_pii_removal,
            use_threading=use_threading,
            num_threads=num_threads,
            use_anonymization=perform_anonymization,
            use_verification=verify_anonymization,
            batch_single_type_anon=single_type_anonymization,
            config=config,
            progress_callback=update_progress,
            status_callback=update_status
        )
        
        processing_time = datetime.now() - start_time
        logger.info(f"\nProcessed {len(results)} records in {processing_time.total_seconds():.2f} seconds")
        
        # Calculate summary statistics
        total_presidio = sum(r['summary']['total_presidio'] for r in results)
        total_uk_patterns = sum(r['summary']['total_uk_patterns'] for r in results)
        total_missed = sum(r['summary']['total_missed'] for r in results)
        total_transformer = sum(r['summary'].get('total_transformer', 0) for r in results)
        total_all = sum(r['summary']['total_all'] for r in results)
        
        logger.info(f"Total PII entities found: {total_all}")
        logger.info(f"  - Standard Presidio entities: {total_presidio}")
        logger.info(f"  - UK-specific patterns: {total_uk_patterns}")
        logger.info(f"  - Potentially missed PII: {total_missed}")
        if total_transformer > 0:
            logger.info(f"  - Transformer-based entities: {total_transformer}")
        
        # Export results
        output_file = config.get('output_file', f"pii_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Check if results directory exists, and if so, save there
        import os
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        if os.path.exists(results_dir):
            output_file = os.path.join(results_dir, os.path.basename(output_file))
        
        export_format = config.get('export_format', 'csv').lower()
        
        # Prepare export data
        export_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'file_analyzed': config['input_file'],
                'records_processed': len(results),
                'fields_analyzed': fields_to_process,
                'total_presidio_entities': total_presidio,
                'total_uk_patterns': total_uk_patterns,
                'total_missed_entities': total_missed,
                'total_transformer_entities': total_transformer,
                'total_all_entities': total_all,
                'analysis_options': {
                    'use_threading': use_threading,
                    'num_threads': num_threads,
                    'perform_anonymization': perform_anonymization,
                    'verify_anonymization': verify_anonymization,
                    'use_transformers': config.get('use_transformers', False),
                    'transformer_model': config.get('transformer_model', ''),
                    'transformer_models': config.get('transformer_models', [])
                }
            },
            'detailed_results': results
        }
        
        # Export the results
        if export_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif export_format == 'csv':
            # Create a flattened dataframe for CSV export
            export_results(export_data['detailed_results'], output_file, format='CSV')
        elif export_format == 'excel':
            # Create a flattened dataframe for Excel export
            export_results(export_data['detailed_results'], output_file, format='Excel')
        
        logger.info(f"Results exported to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
