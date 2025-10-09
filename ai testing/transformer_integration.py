"""
Integration of Hugging Face transformer-based recognizer with PII analysis CLI.
This module provides functions to enhance the existing PII analysis with
transformer-based entity recognition.
"""

import logging
from typing import Dict, List, Optional, Set, Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Import our custom recognizer
from hf_transformer_recognizer import TransformerEntityRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presidio-transformer-integration")

def get_analyzer_with_transformers(
    sensitivity: float = 0.35,
    uk_specific: bool = True,
    transformer_model: str = "dslim/bert-base-NER",
    transformer_models: Optional[List[str]] = None,
    add_custom_recognizers: Optional[callable] = None,
    use_custom_datetime: bool = True
) -> AnalyzerEngine:
    """
    Initialize and return the analyzer with both custom recognizers and
    the transformer-based recognizer.
    
    Args:
        sensitivity: Sensitivity threshold for detection
        uk_specific: Whether to add UK-specific recognizers
        transformer_model: Hugging Face model to use for entity recognition
        add_custom_recognizers: Function to add custom recognizers to the registry
        use_custom_datetime: Whether to use custom DATE_TIME recognizer instead of built-in
        
    Returns:
        AnalyzerEngine instance
    """
    # Create a new recognizer registry
    registry = RecognizerRegistry()
    
    # Add built-in recognizers
    registry.load_predefined_recognizers()
    
    # Remove built-in DateRecognizer if using custom version
    if use_custom_datetime:
        registry.recognizers = [r for r in registry.recognizers if r.__class__.__name__ != 'DateRecognizer']
        logger.info("Removed built-in DateRecognizer from transformer analyzer registry")
    else:
        logger.info("Keeping built-in DateRecognizer in transformer analyzer registry")
    
    # Load YAML-based recognizers (including our custom DATE_TIME recognizer)
    try:
        registry.add_recognizers_from_yaml(yml_path="presidio_recognizers_config.yaml")
        logger.info("✅ Successfully loaded YAML recognizers in transformer analyzer")
    except Exception as e:
        logger.warning(f"Could not load YAML recognizers in transformer analyzer: {e}")
    
    # Add custom recognizers if provided
    if add_custom_recognizers and callable(add_custom_recognizers):
        # Create a temporary analyzer to register UK recognizers if that's what we're using
        # since register_uk_recognizers expects an analyzer object, not just a registry
        temp_analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["en"]
        )
        if uk_specific:
            add_custom_recognizers(temp_analyzer)
        # For other custom recognizers that might use the registry directly
        # We don't need to do anything as they were already added to registry
    
    # Add transformer-based recognizer(s)
    models_to_add: List[str] = []
    # Back-compat: if caller passed transformer_models, use that; else fall back to single transformer_model
    if transformer_models and isinstance(transformer_models, list) and len(transformer_models) > 0:
        models_to_add = transformer_models
    elif transformer_model:
        models_to_add = [transformer_model]

    for idx, model in enumerate(models_to_add):
        try:
            tr = TransformerEntityRecognizer(
                model_name=model,
                confidence_threshold=sensitivity
            )
            # Ensure unique recognizer name to avoid collisions in registry listings/logs
            try:
                tr.name = f"transformer_entity_recognizer_{idx}_{model.replace('/', '_')}"
            except Exception:
                pass
            registry.add_recognizer(tr)
            logger.info(f"Added transformer recognizer with model: {model}")
        except Exception as e:
            logger.error(f"Failed to add transformer recognizer for model '{model}': {e}")
            logger.error("Continuing without this transformer recognizer")
    
    # Create analyzer with custom NLP engine
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
    }).create_engine()
    
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"]
    )
    
    # Store sensitivity threshold for later use
    # (Presidio no longer has a direct threshold attribute on AnalyzerEngine)
    analyzer._sensitivity = sensitivity
    
    return analyzer

def enhance_pii_analysis_results(
    results: List[Dict], 
    transformer_results: List[Dict]
) -> List[Dict]:
    """
    Enhance existing PII analysis results with transformer-based results.
    
    Args:
        results: Original PII analysis results
        transformer_results: Results from transformer-based recognition
        
    Returns:
        Enhanced results
    """
    # Create a deep copy of the original results
    enhanced_results = results.copy()
    
    # Add a source field to the original results if not present
    for result in enhanced_results:
        if "source" not in result:
            result["source"] = "presidio_standard"
    
    # Add transformer results with source field
    for tr_result in transformer_results:
        tr_result["source"] = f"transformer_{tr_result.get('model_entity', 'unknown')}"
        enhanced_results.append(tr_result)
    
    return enhanced_results

"""
Example usage in pii_analysis_cli.py:

# In your existing analyze_text function:

from transformer_integration import get_analyzer_with_transformers, enhance_pii_analysis_results

def analyze_text_with_transformers(text, analyzer, selected_entities=None, 
                                  min_chars=3, analyze_anonymized=True,
                                  transformer_model="dslim/bert-base-NER",
                                  sensitivity=0.35):
    # Get standard results
    standard_results = analyze_text(text, analyzer, selected_entities, 
                                    min_chars, analyze_anonymized)
    
    # If transformers are not being used, return standard results
    if not transformer_model:
        return standard_results
    
    # Get transformer analyzer (can be cached)
    transformer_analyzer = get_analyzer_with_transformers(
        sensitivity=sensitivity,
        transformer_model=transformer_model,
        add_custom_recognizers=add_custom_recognizers  # Your existing function
    )
    
    # Analyze with transformer
    transformer_results = transformer_analyzer.analyze(
        text=text, 
        language="en",
        entities=selected_entities if selected_entities else None,
        score_threshold=sensitivity  # Pass sensitivity as score_threshold
    )
    
    # Convert transformer results to the same format as standard results
    formatted_transformer_results = []
    for result in transformer_results:
        formatted_result = {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "confidence": result.score,
            "text": text[result.start:result.end],
            "field_category": entity_field_map.get(result.entity_type, "Other"),
            "source": "transformer"
        }
        formatted_transformer_results.append(formatted_result)
    
    # Enhance entities_found with transformer results
    standard_results["entities_found"] = enhance_pii_analysis_results(
        standard_results["entities_found"],
        formatted_transformer_results
    )
    
    return standard_results
"""
