#!/usr/bin/env python3
"""
Comprehensive PII detection test with all recognizers.
This script runs PII detection on 20 records using all available recognizers.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import time
import yaml
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc

# Configure logging (will be updated by analyzer manager)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pii-test-all-recognizers")

# Import the configurable analyzer manager
try:
    from configurable_analyzers import ConfigurableAnalyzerManager, initialize_all_analyzers
    CONFIGURABLE_ANALYZERS_AVAILABLE = True
    logger.info("Configurable analyzer system available")
except ImportError as e:
    logger.warning(f"Configurable analyzer system not available: {e}")
    CONFIGURABLE_ANALYZERS_AVAILABLE = False

def load_config(config_file='config.yaml'):
    """Load configuration from YAML file with fallback defaults"""
    default_config = {
        'target_field': 'source',
        'max_records': 1000,
        'num_threads': 4,
        'use_threading': True
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                logger.info(f"Loaded configuration from {config_file}")
                return config
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
            logger.info("Using default configuration")
            return default_config
    else:
        logger.info(f"Config file {config_file} not found, using defaults")
        return default_config

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def load_presidio_engines():
    """Initialize Presidio engines"""
    logger.info("Initializing standard Presidio engines...")
    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception as e:
        logger.error(f"Error initializing Presidio engines: {e}")
        return None, None

def enhance_analyzer_with_uk_recognizers(analyzer):
    """Add UK-specific recognizers to the analyzer"""
    logger.info("Adding UK-specific recognizers...")
    try:
        from uk_recognizers import register_uk_recognizers
        register_uk_recognizers(analyzer)
        logger.info("Successfully added UK-specific recognizers")
        return True
    except Exception as e:
        logger.error(f"Error adding UK-specific recognizers: {e}")
        return False

def enhance_analyzer_with_transformers(analyzer):
    """Add transformer-based recognizers to the analyzer"""
    logger.info("Adding transformer-based recognizers...")
    try:
        from transformer_integration import get_analyzer_with_transformers
        from uk_recognizers import register_uk_recognizers
        
        # Create a function to add custom recognizers
        def add_custom_recognizers(analyzer_obj):
            register_uk_recognizers(analyzer_obj)
            return True
        
        analyzer = get_analyzer_with_transformers(
            sensitivity=0.35,
            uk_specific=True,
            transformer_model="dslim/bert-base-NER",
            add_custom_recognizers=add_custom_recognizers
        )
        
        logger.info("Successfully added transformer-based recognizers")
        return analyzer
    except Exception as e:
        logger.error(f"Error adding transformer-based recognizers: {e}")
        return None

def analyze_text_with_presidio(text, analyzer, entities=None):
    """Analyze text using Presidio analyzer"""
    if not text or len(text.strip()) == 0:
        return []
    
    # Set default threshold
    threshold = getattr(analyzer, '_sensitivity', 0.35)
    
    # Analyze the text
    results = analyzer.analyze(
        text=text,
        language='en',
        entities=entities,
        score_threshold=threshold
    )
    
    # Convert to a standard format
    entities = []
    for result in results:
        entity = {
            'entity_type': result.entity_type,
            'text': text[result.start:result.end],
            'start': result.start,
            'end': result.end,
            'confidence': result.score
        }
        
        # Add source information if available
        if hasattr(result, 'recognition_metadata') and result.recognition_metadata:
            source = result.recognition_metadata.get('source', 'presidio')
            entity['source'] = source
            
            # Add detailed recognizer information
            if 'recognizer_name' in result.recognition_metadata:
                entity['recognizer_details'] = {
                    'recognizer_name': result.recognition_metadata.get('recognizer_name', 'Unknown'),
                    'recognizer_type': 'presidio_builtin' if source == 'presidio' else 'custom',
                    'recognition_metadata': result.recognition_metadata
                }
        else:
            entity['source'] = 'presidio'
            # Try to extract recognizer information from the result object
            if hasattr(result, 'recognizer_name'):
                entity['recognizer_details'] = {
                    'recognizer_name': result.recognizer_name,
                    'recognizer_type': 'presidio_builtin',
                    'score_explanation': getattr(result, 'analysis_explanation', 'No explanation available')
                }
        
        entities.append(entity)
    
    return entities

def analyze_text_with_ollama(text, ollama_extractor=None):
    """Analyze text using pre-initialized Ollama extractor with validation"""
    if not text or len(text.strip()) == 0:
        return []
    
    if not ollama_extractor:
        logger.warning("No Ollama extractor provided - skipping Ollama analysis")
        return []
    
    try:
        entities = ollama_extractor.extract_entities(text)
        
        # Add source information and convert to expected format
        formatted_entities = []
        for entity in entities:
            entity_type = entity.get('entity_type', 'UNKNOWN')
            entity_text = entity.get('text', '')
            
            # Validate UK_NI_NUMBER entities to prevent false positives
            if entity_type == 'UK_NI_NUMBER':
                if not validate_uk_ni_number(entity_text):
                    logger.warning(f"Ollama false positive: '{entity_text}' is not a valid UK NI number pattern - skipping")
                    continue
            
            # Validate other common false positives
            if entity_type == 'CREDIT_CARD' and len(entity_text.replace(' ', '').replace('-', '')) != 16:
                logger.warning(f"Ollama validation: Credit card '{entity_text}' has invalid length - adjusting confidence")
                # Reduce confidence for suspicious credit card detections
                confidence = min(entity.get('confidence', 0.5), 0.3)
            else:
                confidence = entity.get('confidence', 0.5)
            
            formatted_entity = {
                'entity_type': entity_type,
                'text': entity_text,
                'start': entity.get('start', 0),
                'end': entity.get('end', 0),
                'confidence': confidence,
                'source': 'ollama-mistral',
                'recognizer_details': {
                    'recognizer_name': 'OllamaEntityExtractor',
                    'recognizer_type': 'llm_based',
                    'model_name': 'mistral:7b-instruct',
                    'extraction_method': 'natural_language_understanding',
                    'validation_applied': entity_type in ['UK_NI_NUMBER', 'CREDIT_CARD']
                }
            }
            formatted_entities.append(formatted_entity)
        
        return formatted_entities
    except Exception as e:
        logger.error(f"Error analyzing text with Ollama: {e}")
        return []

def validate_uk_ni_number(text):
    """Validate if text matches UK National Insurance number pattern"""
    import re
    if not text:
        return False
    
    # Check against valid UK NI number patterns
    patterns = [
        r'^[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]$',  # With or without spaces
        r'^[A-Z]{2}\d{6}[A-Z]?$'  # Without spaces, optional suffix
    ]
    
    for pattern in patterns:
        if re.match(pattern, text.strip()):
            return True
    
    return False

def analyze_text_with_patterns(text):
    """Analyze text using simple regex patterns"""
    if not text or len(text.strip()) == 0:
        return []
    
    import re
    
    # Simple pattern-based PII detectors
    PII_PATTERNS = {
        "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE_NUMBER": r'\b(?:\+\d{1,3}\s?)?\(?\d{3,5}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "UK_POSTCODE": r'\b[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "PERSON": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
        "UK_NI_NUMBER": r'\b[A-Z]{2}\d{6}[A-Z]?\b',
        "UK_NHS_NUMBER": r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
        "ADDRESS": r'\b\d+\s+[A-Za-z]+ (?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl|Terrace|Ter)\b',
    }
    
    entities = []
    
    for entity_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            entity = {
                "entity_type": entity_type,
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85,  # Mock confidence score
                "source": "pattern-matching",
                "recognizer_details": {
                    "recognizer_name": f"Pattern_{entity_type}",
                    "recognizer_type": "regex-based",
                    "pattern_used": pattern,
                    "pattern_source": "hardcoded"
                }
            }
            entities.append(entity)
    
    return entities

def resolve_overlapping_entities(entities, prioritize_person=True):
    """
    Resolve overlapping entities with improved logic for PERSON entities
    
    Args:
        entities: List of entities to resolve
        prioritize_person: If True, prioritize PERSON entities in conflicts
    
    Returns:
        List of resolved entities
    """
    if not entities:
        return entities
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    resolved_entities = []
    
    for current_entity in sorted_entities:
        # Check if current entity overlaps with any in resolved list
        overlaps = False
        for i, resolved_entity in enumerate(resolved_entities):
            # Check for overlap (entities overlap if they share any character positions)
            if (current_entity['start'] < resolved_entity['end'] and 
                current_entity['end'] > resolved_entity['start']):
                
                overlaps = True
                
                # Enhanced resolution logic with PERSON prioritization
                should_replace = False
                
                if prioritize_person:
                    # Priority 1: PERSON entities get preference
                    if current_entity['entity_type'] == 'PERSON' and resolved_entity['entity_type'] != 'PERSON':
                        should_replace = True
                        logger.info(f"Prioritizing PERSON entity: Replacing '{resolved_entity['text']}' "
                                  f"({resolved_entity['entity_type']}, {resolved_entity['confidence']:.3f}) "
                                  f"with '{current_entity['text']}' (PERSON, {current_entity['confidence']:.3f})")
                    elif current_entity['entity_type'] != 'PERSON' and resolved_entity['entity_type'] == 'PERSON':
                        should_replace = False
                        logger.info(f"Keeping PERSON entity: '{resolved_entity['text']}' "
                                  f"(PERSON, {resolved_entity['confidence']:.3f}) "
                                  f"over '{current_entity['text']}' ({current_entity['entity_type']}, {current_entity['confidence']:.3f})")
                    else:
                        # Both are PERSON or neither is PERSON - use confidence
                        should_replace = current_entity['confidence'] > resolved_entity['confidence']
                else:
                    # Standard confidence-based resolution
                    should_replace = current_entity['confidence'] > resolved_entity['confidence']
                
                if should_replace:
                    if not prioritize_person or (current_entity['entity_type'] != 'PERSON' and resolved_entity['entity_type'] != 'PERSON'):
                        logger.info(f"Replacing '{resolved_entity['text']}' ({resolved_entity['confidence']:.3f}) "
                                  f"with '{current_entity['text']}' ({current_entity['confidence']:.3f}) due to higher confidence")
                    resolved_entities[i] = current_entity
                else:
                    if not prioritize_person or (current_entity['entity_type'] != 'PERSON' and resolved_entity['entity_type'] != 'PERSON'):
                        logger.info(f"Keeping '{resolved_entity['text']}' ({resolved_entity['confidence']:.3f}) "
                                  f"over '{current_entity['text']}' ({current_entity['confidence']:.3f}) due to higher confidence")
                break
        
        if not overlaps:
            resolved_entities.append(current_entity)
    
    return resolved_entities

def analyze_text_comprehensive(text, analyzers, resolve_overlaps=True, prioritize_person=True):
    """
    Analyze text using all available recognizers with overlap resolution
    
    Args:
        text: Text to analyze
        analyzers: Dictionary of pre-initialized analyzers
        resolve_overlaps: Whether to resolve overlapping entities (default: True)
        prioritize_person: Whether to prioritize PERSON entities in overlap resolution (default: True)
    
    Returns:
        Tuple[List[Dict], Dict[str, float]] where the dict contains per-analyzer timings in seconds
    """
    # Check if this is built-in only mode
    if analyzers.get('builtin_only', False):
        logger.debug("Using Presidio-only analysis (built-in only mode)")
        t0 = time.perf_counter()
        ents = analyze_text_presidio_only(text, analyzers, resolve_overlaps, prioritize_person)
        timings = {'presidio': time.perf_counter() - t0, 'transformer': 0.0, 'patterns': 0.0, 'ollama': 0.0}
        return ents, timings
    
    all_entities = []
    timings = {'presidio': 0.0, 'transformer': 0.0, 'patterns': 0.0, 'ollama': 0.0}
    
    # Use pre-initialized Presidio analyzer
    if analyzers.get('presidio'):
        _t0 = time.perf_counter()
        presidio_entities = analyze_text_with_presidio(text, analyzers['presidio'])
        timings['presidio'] = time.perf_counter() - _t0
        all_entities.extend(presidio_entities)
    
    # Use pre-initialized transformer analyzer
    if analyzers.get('transformer'):
        _t0 = time.perf_counter()
        transformer_entities = analyze_text_with_presidio(text, analyzers['transformer'])
        timings['transformer'] = time.perf_counter() - _t0
        all_entities.extend(transformer_entities)
    
    # Add pattern-based entities
    _t0 = time.perf_counter()
    pattern_entities = analyze_text_with_patterns(text)
    timings['patterns'] = time.perf_counter() - _t0
    all_entities.extend(pattern_entities)
    
    # Add Ollama-based entities (if available)
    try:
        if analyzers.get('ollama'):
            _t0 = time.perf_counter()
            ollama_entities = analyze_text_with_ollama(text, analyzers['ollama'])
            timings['ollama'] = time.perf_counter() - _t0
            all_entities.extend(ollama_entities)
        else:
            logger.debug("Ollama analyzer not available - skipping")
    except Exception as e:
        logger.error(f"Error with Ollama analysis: {e}")
    
    # Remove duplicates (same entity type, start, and end)
    unique_entities = {}
    for entity in all_entities:
        key = f"{entity['entity_type']}:{entity['start']}:{entity['end']}"
        # Keep the entity with the highest confidence if there's a duplicate
        if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
            unique_entities[key] = entity
    
    entities_list = list(unique_entities.values())
    
    # Resolve overlaps if requested
    if resolve_overlaps:
        entities_list = resolve_overlapping_entities(entities_list, prioritize_person=prioritize_person)
    
    return entities_list, timings

def analyze_text_presidio_only(text, analyzers, resolve_overlaps=True, prioritize_person=True):
    """
    Analyze text using ONLY the Presidio analyzer, no pattern matching or other analyzers
    
    Args:
        text: Text to analyze
        analyzers: Dictionary of pre-initialized analyzers
        resolve_overlaps: Whether to resolve overlapping entities (default: True)
        prioritize_person: Whether to prioritize PERSON entities in overlap resolution (default: True)
    
    Returns:
        List of detected entities from Presidio only
    """
    all_entities = []
    
    # Use ONLY the pre-initialized Presidio analyzer
    if analyzers.get('presidio'):
        presidio_entities = analyze_text_with_presidio(text, analyzers['presidio'])
        all_entities.extend(presidio_entities)
    else:
        logger.warning("No Presidio analyzer available for built-in only mode")
        return []
    
    # Remove duplicates (same entity type, start, and end)
    unique_entities = {}
    for entity in all_entities:
        key = f"{entity['entity_type']}:{entity['start']}:{entity['end']}"
        # Keep the entity with the highest confidence if there's a duplicate
        if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
            unique_entities[key] = entity
    
    entities_list = list(unique_entities.values())
    
    # Resolve overlaps if requested
    if resolve_overlaps:
        entities_list = resolve_overlapping_entities(entities_list, prioritize_person=prioritize_person)
    
    return entities_list

def filter_entities_by_config(entities, config):
    """Filter entities based on configuration settings"""
    if not config or 'pii_entities' not in config or not config['pii_entities']:
        # If no entity filtering configured, return all entities
        return entities
    
    allowed_entities = config['pii_entities']
    logger.info(f"Filtering entities to allowed types: {allowed_entities}")
    
    filtered_entities = []
    for entity in entities:
        if entity['entity_type'] in allowed_entities:
            filtered_entities.append(entity)
        else:
            logger.debug(f"Filtered out entity type '{entity['entity_type']}': {entity['text']}")
    
    logger.info(f"Entity filtering: {len(entities)} -> {len(filtered_entities)} entities")
    return filtered_entities

def initialize_all_analyzers_legacy(use_ollama=True, ollama_max_concurrency: int | None = None, ollama_max_tokens: int | None = None):
    """Legacy analyzer initialization - kept for backwards compatibility"""
    analyzers = {
        'presidio': None,
        'transformer': None,
        'ollama': None  # Add Ollama instance
    }
    
    logger.info(f"Using legacy analyzer initialization, use_ollama={use_ollama}...")
    
    # Initialize standard Presidio
    logger.info("Initializing standard Presidio engines...")
    analyzer, _ = load_presidio_engines()
    if analyzer:
        # Add UK recognizers
        logger.info("Adding UK-specific recognizers...")
        enhance_analyzer_with_uk_recognizers(analyzer)
        logger.info("Successfully added UK-specific recognizers")
        analyzers['presidio'] = analyzer
        
        # Try to create transformer-enhanced analyzer
        try:
            logger.info("Adding transformer-based recognizers...")
            transformer_analyzer = enhance_analyzer_with_transformers(analyzer)
            if transformer_analyzer:
                analyzers['transformer'] = transformer_analyzer
                logger.info("Successfully added transformer-based recognizers")
        except Exception as e:
            logger.error(f"Error initializing transformer analyzer: {e}")
    
    # Initialize Ollama extractor with persistent HTTP connection
    if use_ollama:
        try:
            logger.info("Initializing Ollama extractor...")
            from ollama_integration import OllamaEntityExtractor
            # Align Ollama concurrency with worker count if provided
            _max_conc = max(1, int(ollama_max_concurrency)) if ollama_max_concurrency else 2
            _max_tokens = int(ollama_max_tokens) if ollama_max_tokens is not None else 128
            ollama_extractor = OllamaEntityExtractor(
                model_name="mistral:7b-instruct",
                temperature=0.1,
                max_tokens=_max_tokens,
                http_pool_maxsize=max(8, _max_conc),
                request_timeout_seconds=30,
                keep_alive="10m",
                max_concurrent_requests=_max_conc,
            )
            logger.info(f"Ollama configured: max_concurrent_requests={_max_conc}, max_tokens={_max_tokens}")
            analyzers['ollama'] = ollama_extractor
            logger.info("Successfully initialized Ollama extractor")
        except Exception as e:
            logger.error(f"Error initializing Ollama extractor: {e}")
            analyzers['ollama'] = None
    else:
        logger.info("Ollama disabled by configuration")
        analyzers['ollama'] = None
    
    logger.info("Legacy analyzer initialization complete")
    return analyzers

def initialize_all_analyzers(use_ollama=True, config_file='analyzers_config.yaml', ollama_max_concurrency: int | None = None, ollama_max_tokens: int | None = None):
    """Initialize all analyzers using configurable system or fall back to legacy"""
    if CONFIGURABLE_ANALYZERS_AVAILABLE and os.path.exists(config_file):
        logger.info(f"Using configurable analyzer system with {config_file}")
        try:
            # Use the configurable analyzer manager
            from configurable_analyzers import initialize_all_analyzers as init_configurable
            return init_configurable(use_ollama, config_file)
        except Exception as e:
            logger.error(f"Error using configurable analyzers: {e}")
            logger.info("Falling back to legacy initialization")
            return initialize_all_analyzers_legacy(use_ollama, ollama_max_concurrency, ollama_max_tokens)
    else:
        if not os.path.exists(config_file):
            logger.info(f"Configuration file {config_file} not found, using legacy initialization")
        else:
            logger.info("Configurable analyzers not available, using legacy initialization")
    return initialize_all_analyzers_legacy(use_ollama, ollama_max_concurrency, ollama_max_tokens)

def get_recognizer_information(analyzers, builtin_only=False):
    """
    Collect detailed information about all available recognizers and their mappings.
    
    Args:
        analyzers: Dictionary of analyzer instances
        builtin_only: Whether only built-in analyzers are used
        
    Returns:
        Dictionary containing recognizer information and mappings
    """
    recognizer_info = {
        'recognizer_categories': {},
        'entity_type_mappings': {},
        'pattern_definitions': {},
        'total_recognizers': 0
    }
    
    # Built-in Presidio recognizers
    if analyzers.get('presidio'):
        presidio_recognizers = []
        for recognizer in analyzers['presidio'].registry.recognizers:
            recognizer_name = recognizer.__class__.__name__
            supported_entities = getattr(recognizer, 'supported_entities', [])
            presidio_recognizers.append({
                'name': recognizer_name,
                'supported_entities': supported_entities,
                'type': 'built-in'
            })
        
        recognizer_info['recognizer_categories']['presidio_builtin'] = presidio_recognizers
        recognizer_info['total_recognizers'] += len(presidio_recognizers)
        
        # Map entity types to recognizers
        for recognizer_data in presidio_recognizers:
            for entity_type in recognizer_data['supported_entities']:
                if entity_type not in recognizer_info['entity_type_mappings']:
                    recognizer_info['entity_type_mappings'][entity_type] = []
                recognizer_info['entity_type_mappings'][entity_type].append({
                    'recognizer': recognizer_data['name'],
                    'source': 'presidio',
                    'type': 'built-in'
                })
    
    # Custom UK recognizers (only if not built-in only mode)
    if not builtin_only:
        try:
            from uk_recognizers import create_uk_recognizers
            uk_recognizers = create_uk_recognizers()
            uk_recognizer_list = []
            
            for recognizer in uk_recognizers:
                recognizer_name = recognizer.__class__.__name__
                supported_entity = getattr(recognizer, 'supported_entity', 'UNKNOWN')
                patterns = []
                if hasattr(recognizer, 'patterns'):
                    patterns = [{'name': p.name, 'regex': p.regex, 'score': p.score} for p in recognizer.patterns]
                
                uk_recognizer_list.append({
                    'name': recognizer_name,
                    'supported_entity': supported_entity,
                    'type': 'custom_uk',
                    'patterns': patterns
                })
                
                # Add to entity type mappings
                if supported_entity not in recognizer_info['entity_type_mappings']:
                    recognizer_info['entity_type_mappings'][supported_entity] = []
                recognizer_info['entity_type_mappings'][supported_entity].append({
                    'recognizer': recognizer_name,
                    'source': 'custom_uk',
                    'type': 'pattern-based'
                })
                
                # Store pattern definitions
                recognizer_info['pattern_definitions'][supported_entity] = patterns
            
            recognizer_info['recognizer_categories']['custom_uk'] = uk_recognizer_list
            recognizer_info['total_recognizers'] += len(uk_recognizer_list)
            
        except ImportError:
            logger.warning("UK recognizers not available")
        
        # Pattern-matching recognizers (hardcoded patterns)
        PII_PATTERNS = {
            "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_NUMBER": r'\b(?:\+\d{1,3}\s?)?\(?\d{3,5}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "UK_POSTCODE": r'\b[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "PERSON": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "UK_NI_NUMBER": r'\b[A-Z]{2}\d{6}[A-Z]?\b',
            "UK_NHS_NUMBER": r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
            "ADDRESS": r'\b\d+\s+[A-Za-z]+ (?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl|Terrace|Ter)\b',
        }
        
        pattern_recognizers = []
        for entity_type, pattern in PII_PATTERNS.items():
            pattern_recognizers.append({
                'name': f'Pattern_{entity_type}',
                'supported_entity': entity_type,
                'type': 'pattern_matching',
                'regex': pattern
            })
            
            # Add to entity type mappings
            if entity_type not in recognizer_info['entity_type_mappings']:
                recognizer_info['entity_type_mappings'][entity_type] = []
            recognizer_info['entity_type_mappings'][entity_type].append({
                'recognizer': f'Pattern_{entity_type}',
                'source': 'pattern-matching',
                'type': 'regex-based'
            })
            
            # Store pattern definition
            if entity_type not in recognizer_info['pattern_definitions']:
                recognizer_info['pattern_definitions'][entity_type] = []
            recognizer_info['pattern_definitions'][entity_type].append({
                'name': f'hardcoded_{entity_type.lower()}',
                'regex': pattern,
                'score': 0.8  # Default score for pattern matching
            })
        
        recognizer_info['recognizer_categories']['pattern_matching'] = pattern_recognizers
        recognizer_info['total_recognizers'] += len(pattern_recognizers)
    
    # Transformer-based recognizers (if available)
    if analyzers.get('transformer') and not builtin_only:
        # This would be similar to presidio but enhanced with transformers
        # For now, we'll note it's available but defer detailed analysis
        recognizer_info['recognizer_categories']['transformer_enhanced'] = {
            'note': 'Transformer-enhanced Presidio analyzer available but details not extracted',
            'type': 'enhanced'
        }
    
    # Ollama LLM-based extraction
    if analyzers.get('ollama') and not builtin_only:
        recognizer_info['recognizer_categories']['ollama_llm'] = {
            'name': 'OllamaEntityExtractor',
            'model': 'mistral:7b-instruct',
            'type': 'llm_based',
            'supported_entities': 'dynamic_extraction'
        }
        recognizer_info['total_recognizers'] += 1
    
    return recognizer_info

def initialize_builtin_presidio_only():
    """
    Initialize only built-in Presidio recognizers, no custom recognizers.
    
    Returns:
        Dictionary with only the 'presidio' analyzer containing built-in recognizers
    """
    logger.info("Initializing built-in Presidio recognizers only (no custom recognizers)")
    
    analyzers = {}
    
    try:
        from presidio_analyzer import AnalyzerEngine
        
        # Create a standard Presidio analyzer with only built-in recognizers
        presidio_analyzer = AnalyzerEngine()
        
        logger.info("Successfully initialized Presidio with built-in recognizers only")
        logger.info(f"Built-in recognizers loaded: {len(presidio_analyzer.registry.recognizers)}")
        
        # Log the recognizer names for verification
        recognizer_names = [r.__class__.__name__ for r in presidio_analyzer.registry.recognizers]
        logger.info(f"Built-in recognizers: {', '.join(recognizer_names)}")
        
        analyzers['presidio'] = presidio_analyzer
        analyzers['ollama'] = None  # No Ollama in built-in only mode
        
        # Add a flag to indicate this is built-in only mode
        analyzers['builtin_only'] = True
        
    except Exception as e:
        logger.error(f"Error initializing built-in Presidio analyzer: {e}")
        analyzers['presidio'] = None
        analyzers['ollama'] = None
    
    logger.info("Built-in only analyzer initialization complete")
    return analyzers

def get_fields_to_process(target_field):
    """Determine which fields to process based on target field configuration"""
    if target_field == "source":
        # Process the concatenated source field only
        return ["source"]
    elif target_field in ["customer_message", "agent_response"]:
        # Process the specified individual field
        return [target_field]
    else:
        # Default to both conversation fields for backwards compatibility
        logger.warning(f"Unknown target field '{target_field}', defaulting to customer_message and agent_response")
        return ["customer_message", "agent_response"]

def process_record_thread_safe(record_data, analyzers, resolve_overlaps=True, min_confidence=0.0, target_field="source", config=None, prioritize_person=True):
    """Process a single record in a thread-safe manner"""
    idx, row = record_data
    fields_to_process = get_fields_to_process(target_field)
    
    # Parse PII fields from the dataset
    pii_fields_used = []
    pii_raw_values = []
    
    try:
        if pd.notna(row['pii_fields_used']) and row['pii_fields_used'] != '[]':
            import ast
            pii_fields_used = ast.literal_eval(row['pii_fields_used'])
            pii_raw_values = ast.literal_eval(row['pii_raw_values'])
    except Exception as e:
        logger.warning(f"Could not parse PII fields for record {idx}: {e}")
    
    record_result = {
        'record_id': f'record_{idx}',
        'record_index': idx,
        'conversation_id': row.get('conversation_id', ''),
        'customer_id': row.get('customer_id', ''),
        'template_category': row.get('template_category', ''),
        'template_sensitivity': row.get('template_sensitivity', ''),
        'ground_truth_pii': {
            'fields_used': pii_fields_used,
            'raw_values': pii_raw_values,
            'count': len(pii_fields_used)
        },
        'fields': {}
    }
    
    record_entities = 0
    # Aggregate timings per analyzer at record level
    record_timings = {'presidio': 0.0, 'transformer': 0.0, 'patterns': 0.0, 'ollama': 0.0}
    record_entity_sources = {}
    
    # Process each field in the record
    for field in fields_to_process:
        if pd.notna(row[field]) and len(str(row[field]).strip()) > 0:
            field_text = str(row[field])
            
            # Process with all recognizers (using pre-initialized analyzers)
            logger.info(f"Processing record {idx}, field '{field}'...")
            entities, timings = analyze_text_comprehensive(field_text, analyzers, resolve_overlaps=resolve_overlaps, prioritize_person=prioritize_person)
            
            # Filter by configuration if specified
            if config:
                entities = filter_entities_by_config(entities, config)
            
            # Filter by confidence if specified
            if min_confidence > 0:
                entities = [e for e in entities if e['confidence'] >= min_confidence]
                logger.info(f"Filtered to {len(entities)} entities with confidence >= {min_confidence}")
            
            # Count entities by source
            for entity in entities:
                source = entity.get('source', 'unknown')
                record_entity_sources[source] = record_entity_sources.get(source, 0) + 1
            
            # Anonymize the text
            anonymized_text = anonymize_text(field_text, entities)
            
            # Add to results
            record_result['fields'][field] = {
                'original_text': field_text,
                'entities': entities,
                'anonymized_text': anonymized_text,
                'entity_count': len(entities),
                'timings': timings
            }

            record_entities += len(entities)
            # Accumulate timings
            try:
                record_timings['presidio'] += float(timings.get('presidio', 0.0))
                record_timings['transformer'] += float(timings.get('transformer', 0.0))
                record_timings['patterns'] += float(timings.get('patterns', 0.0))
                record_timings['ollama'] += float(timings.get('ollama', 0.0))
            except Exception:
                pass
    
    # Log completion
    logger.info(f"Processed record {idx}: found {record_entities} entities")
    
    # Log comparison with ground truth
    ground_truth_count = len(pii_fields_used)
    logger.info(f"Ground truth PII count: {ground_truth_count}, Detected: {record_entities}")
    if pii_fields_used:
        logger.info(f"Ground truth PII fields: {pii_fields_used}")
        logger.info(f"Ground truth PII values: {pii_raw_values}")
    
    # Attach per-record aggregated timings
    record_result['timings'] = record_timings
    return record_result, record_entities, record_entity_sources

def process_records_multithreaded(df, analyzers, resolve_overlaps=True, min_confidence=0.0, max_workers=4, target_field="source", config=None, prioritize_person=True):
    """Process records using multithreading for improved performance"""
    logger.info(f"Starting multithreaded processing with {max_workers} workers...")
    
    results = []
    total_entities = 0
    entity_sources = {}
    
    # Create thread pool and submit all records for processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all records to the thread pool
        future_to_record = {}
        for idx, row in df.iterrows():
            future = executor.submit(
                process_record_thread_safe,
                (idx, row),
                analyzers,
                resolve_overlaps,
                min_confidence,
                target_field,
                config,
                prioritize_person
            )
            future_to_record[future] = idx
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_record):
            record_idx = future_to_record[future]
            try:
                record_result, record_entities, record_entity_sources = future.result()
                results.append(record_result)
                total_entities += record_entities
                
                # Merge entity sources
                for source, count in record_entity_sources.items():
                    entity_sources[source] = entity_sources.get(source, 0) + count
                
                completed_count += 1
                logger.info(f"Completed {completed_count}/{len(df)} records")
                
            except Exception as exc:
                logger.error(f'Record {record_idx} generated an exception: {exc}')
    
    # Sort results by record index to maintain original order
    results.sort(key=lambda x: x['record_index'])
    
    return results, total_entities, entity_sources

def process_records_sequential(df, analyzers, resolve_overlaps=True, min_confidence=0.0, target_field="source", config=None, prioritize_person=True):
    """Process records sequentially (original implementation)"""
    logger.info("Starting sequential processing...")
    
    results = []
    total_entities = 0
    entity_sources = {}
    fields_to_process = get_fields_to_process(target_field)
    
    for idx, row in df.iterrows():
        # Parse PII fields from the dataset
        pii_fields_used = []
        pii_raw_values = []
        
        try:
            if pd.notna(row['pii_fields_used']) and row['pii_fields_used'] != '[]':
                import ast
                pii_fields_used = ast.literal_eval(row['pii_fields_used'])
                pii_raw_values = ast.literal_eval(row['pii_raw_values'])
        except Exception as e:
            logger.warning(f"Could not parse PII fields for record {idx}: {e}")
        
        record_result = {
            'record_id': f'record_{idx}',
            'record_index': idx,
            'conversation_id': row.get('conversation_id', ''),
            'customer_id': row.get('customer_id', ''),
            'template_category': row.get('template_category', ''),
            'template_sensitivity': row.get('template_sensitivity', ''),
            'ground_truth_pii': {
                'fields_used': pii_fields_used,
                'raw_values': pii_raw_values,
                'count': len(pii_fields_used)
            },
            'fields': {}
        }
        
        record_entities = 0
        
        # Process each field in the record
        for field in fields_to_process:
            if pd.notna(row[field]) and len(str(row[field]).strip()) > 0:
                field_text = str(row[field])
                
                # Process with all recognizers (using pre-initialized analyzers)
                logger.info(f"Processing record {idx}, field '{field}'...")
                entities, timings = analyze_text_comprehensive(field_text, analyzers, resolve_overlaps=resolve_overlaps, prioritize_person=prioritize_person)
                
                # Filter by confidence if specified
                if min_confidence > 0:
                    entities = [e for e in entities if e['confidence'] >= min_confidence]
                    logger.info(f"Filtered to {len(entities)} entities with confidence >= {min_confidence}")
                
                # Count entities by source
                for entity in entities:
                    source = entity.get('source', 'unknown')
                    entity_sources[source] = entity_sources.get(source, 0) + 1
                
                # Anonymize the text
                anonymized_text = anonymize_text(field_text, entities)
                
                # Add to results
                record_result['fields'][field] = {
                    'original_text': field_text,
                    'entities': entities,
                    'anonymized_text': anonymized_text,
                    'entity_count': len(entities),
                    'timings': timings
                }
                
                record_entities += len(entities)
                total_entities += len(entities)
        
        results.append(record_result)
        
        # Log progress
        logger.info(f"Processed record {idx}: found {record_entities} entities")
        
        # Log comparison with ground truth
        ground_truth_count = len(pii_fields_used)
        logger.info(f"Ground truth PII count: {ground_truth_count}, Detected: {record_entities}")
        if pii_fields_used:
            logger.info(f"Ground truth PII fields: {pii_fields_used}")
            logger.info(f"Ground truth PII values: {pii_raw_values}")
    
    return results, total_entities, entity_sources

def anonymize_text(text, entities):
    """Anonymize text by replacing entities with markers"""
    if not text or not entities:
        return text
    
    # Sort entities by start position (descending) to avoid offset issues
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Create a copy of the text to modify
    modified_text = text
    
    # Replace each entity with a redacted marker
    for entity in sorted_entities:
        entity_type = entity['entity_type']
        start = entity['start']
        end = entity['end']
        
        # Skip if indices are invalid
        if start < 0 or end <= start or end > len(modified_text):
            continue
        
        # Replace with redacted marker
        modified_text = modified_text[:start] + f"[{entity_type}]" + modified_text[end:]
    
    return modified_text

def main():
    """Main function with multithreading support"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test PII detection with all recognizers')
    parser.add_argument('--records', type=int, default=100, help='Number of records to process')
    parser.add_argument('--no-resolve-overlaps', action='store_true', 
                       help='Disable overlap resolution (keep all duplicate entities)')
    parser.add_argument('--no-prioritize-person', action='store_true',
                       help='Disable PERSON entity prioritization in overlap resolution')
    parser.add_argument('--min-confidence', type=float, default=0.0, 
                       help='Minimum confidence threshold for entities')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--use-ollama', action='store_true', help='Enable Ollama for LLM-based analysis')
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing instead of multithreading')
    parser.add_argument('--workers', type=int, default=4, help='Number of threads for multithreaded processing')
    parser.add_argument('--ollama-max-concurrency', type=int, default=None,
                       help='Max concurrent Ollama requests (defaults to --workers when multithreading)')
    parser.add_argument('--ollama-max-tokens', type=int, default=None,
                       help='Limit Ollama num_predict tokens (smaller is faster; default 128)')
    parser.add_argument('--input-file', type=str, help='Custom input CSV file to process')
    parser.add_argument('--target-field', type=str, help='Field to process for PII detection (default: from config.yaml or "source")')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--analyzer-config', type=str, default='analyzers_config.yaml', 
                       help='Path to analyzer configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save results files')
    parser.add_argument('--builtin-only', action='store_true', 
                       help='Use only built-in Presidio recognizers, no custom recognizers (UK, transformers, etc.)')
    
    args = parser.parse_args()
    
    # Set overlap resolution flags
    resolve_overlaps = not args.no_resolve_overlaps
    prioritize_person = not args.no_prioritize_person
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Configuration - Ollama is disabled by default now
    use_ollama = args.use_ollama
    use_multithreading = not args.sequential
    max_workers = args.workers
    # Derive Ollama concurrency from workers if not provided
    derived_ollama_conc = args.ollama_max_concurrency if args.ollama_max_concurrency is not None else (max_workers if use_multithreading else 1)
    derived_ollama_tokens = args.ollama_max_tokens
    
    # Determine target field (CLI overrides config)
    target_field = args.target_field if args.target_field else config.get('target_field', 'source')
    
    logger.info("Starting PII Detection Analysis with Multithreading Support")
    logger.info(f"Configuration: use_ollama={use_ollama}, use_multithreading={use_multithreading}")
    logger.info(f"Max workers: {max_workers}, Max records: {args.records}")
    logger.info(f"Target field for processing: '{target_field}'")
    
    try:
        # Load the dataset - check custom file first, then default locations
        df = None
        
        if args.input_file:
            # Use custom input file if specified
            if os.path.exists(args.input_file):
                try:
                    if args.input_file.endswith('.json'):
                        df = pd.read_json(args.input_file, lines=True)
                    else:
                        df = pd.read_csv(args.input_file)
                    logger.info(f"Loaded dataset from custom file: {args.input_file}")
                except Exception as e:
                    logger.error(f"Failed to load custom file {args.input_file}: {e}")
                    return
            else:
                logger.error(f"Custom input file not found: {args.input_file}")
                return
        else:
            # Check multiple possible default file locations (prioritizing test_datasets folder)
            possible_files = [
                "test_datasets/unified_test_analyzer_format.csv",
                "test_chat_conversations_full_dataset.csv",
                "/Users/davidlock/Downloads/soccer data python/testing poe/uk_pii_conversations_test.json",
                "uk_pii_conversations_test.json"
            ]
            
            for data_path in possible_files:
                if os.path.exists(data_path):
                    try:
                        if data_path.endswith('.json'):
                            df = pd.read_json(data_path, lines=True)
                        else:
                            df = pd.read_csv(data_path)
                        logger.info(f"Loaded dataset from: {data_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {data_path}: {e}")
                        continue
        
        if df is None:
            logger.error("Could not find or load any dataset file")
            return
        
        # Validate target field exists in dataset
        fields_to_process = get_fields_to_process(target_field)
        missing_fields = [field for field in fields_to_process if field not in df.columns]
        if missing_fields:
            logger.error(f"Target field(s) not found in dataset: {missing_fields}")
            logger.info(f"Available columns: {list(df.columns)}")
            return
        
        # Limit to first N records for testing
        df = df.head(args.records)
        logger.info(f"Processing {len(df)} records (limited to first {args.records})")
        
        # Record timing
        start_time = time.time()
        
        # Initialize all analyzers once (performance optimization)
        logger.info("Initializing analyzers...")
        init_start = time.time()
        
        if args.builtin_only:
            logger.info("Using built-in Presidio recognizers only (--builtin-only flag)")
            analyzers = initialize_builtin_presidio_only()
        else:
            analyzers = initialize_all_analyzers(
                use_ollama=use_ollama,
                config_file=args.analyzer_config,
                ollama_max_concurrency=derived_ollama_conc,
                ollama_max_tokens=derived_ollama_tokens,
            )
            
        init_time = time.time() - init_start
        logger.info(f"Analyzer initialization completed in {init_time:.2f} seconds")
        
        # Process records using multithreading or sequential processing
        processing_start = time.time()
        
        if use_multithreading:
            logger.info(f"Using multithreaded processing with {max_workers} workers...")
            results, total_entities, entity_sources = process_records_multithreaded(
                df, analyzers, resolve_overlaps, args.min_confidence, max_workers, target_field, config, prioritize_person
            )
        else:
            logger.info("Using sequential processing...")
            results, total_entities, entity_sources = process_records_sequential(
                df, analyzers, resolve_overlaps, args.min_confidence, target_field, config, prioritize_person
            )
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        # Log summary statistics
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING COMPLETE - PERFORMANCE SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Processing mode: {'Multithreaded' if use_multithreading else 'Sequential'}")
        logger.info(f"Records processed: {len(results)}")
        logger.info(f"Total entities found: {total_entities}")
        logger.info(f"Average entities per record: {total_entities/len(results):.2f}")
        logger.info(f"Initialization time: {init_time:.2f} seconds")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Log entity distribution by source
        logger.info(f"\nEntity distribution by source:")
        for source, count in sorted(entity_sources.items()):
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            logger.info(f"  {source}: {count} ({percentage:.1f}%)")
        
        # Calculate ground truth comparison
        total_ground_truth = sum(len(result['ground_truth_pii']['fields_used']) for result in results)
        logger.info(f"\nGround truth comparison:")
        logger.info(f"  Total ground truth PII fields: {total_ground_truth}")
        logger.info(f"  Total detected entities: {total_entities}")
        logger.info(f"  Detection ratio: {total_entities/total_ground_truth:.2f}x" if total_ground_truth > 0 else "N/A")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "multithreaded" if use_multithreading else "sequential"
        ollama_suffix = "with_ollama" if use_ollama else "no_ollama"
        builtin_suffix = "builtin_only" if args.builtin_only else "all_analyzers"
        output_filename = f"pii_detection_results_{mode_suffix}_{builtin_suffix}_{ollama_suffix}_{timestamp}.json"
        
        # Ensure output directory exists
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_filename)
        
        # Collect recognizer information for analysis
        logger.info("Collecting recognizer information for metadata...")
        recognizer_info = get_recognizer_information(analyzers, builtin_only=args.builtin_only)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'processing_mode': 'multithreaded' if use_multithreading else 'sequential',
                    'max_workers': max_workers if use_multithreading else 1,
                    'records_processed': len(results),
                    'total_entities_found': total_entities,
                    'processing_time_seconds': processing_time,
                    'total_time_seconds': total_time,
                    'initialization_time_seconds': init_time,
                    'use_ollama': use_ollama,
                    'builtin_only': args.builtin_only,
                    'resolve_overlaps': resolve_overlaps,
                    'min_confidence': args.min_confidence,
                    'entity_sources': entity_sources,
                    'ground_truth_total': total_ground_truth,
                    'configuration': config,
                    'recognizer_analysis': recognizer_info
                },
                'results': results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        logger.info("Analysis complete!")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)
    finally:
        # Cleanup resources
        if 'analyzers' in locals() and analyzers.get('ollama'):
            try:
                analyzers['ollama'].close()
                logger.info("Ollama HTTP session closed")
            except Exception as e:
                logger.warning(f"Error closing Ollama session: {e}")
if __name__ == "__main__":
    main()
