"""
Hugging Face Transformer-based Entity Recognizer for Presidio
This module provides a custom recognizer that leverages Hugging Face transformer models
for enhanced entity recognition within the Microsoft Presidio framework.
"""

import logging
from typing import List, Optional, Tuple, Set, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

# Configure logging
logger = logging.getLogger("presidio-analyzer")

class TransformerEntityRecognizer(EntityRecognizer):
    """
    A custom entity recognizer using Hugging Face transformers for NER.
    
    This recognizer uses a pre-trained transformer model for token classification
    to identify entities in text, then maps those entities to Presidio entity types.
    """
    
    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        supported_entities: Optional[List[str]] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize the TransformerEntityRecognizer with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            supported_entities: List of entity types this recognizer can detect
            entity_mapping: Mapping from model's entity types to Presidio entity types
            confidence_threshold: Minimum confidence score to return entities
            device: Device to run the model on ("cpu" or "cuda")
        """
        if not supported_entities:
            supported_entities = ["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", 
                               "EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "CREDIT_CARD"]
            
        # Default entity mapping for common NER models
        self.entity_mapping = entity_mapping or {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "I-PER": "PERSON",
            "B-PER": "PERSON",
            "LOC": "LOCATION",
            "LOCATION": "LOCATION",
            "I-LOC": "LOCATION",
            "B-LOC": "LOCATION",
            "GPE": "LOCATION",
            "ORG": "ORGANIZATION",
            "ORGANIZATION": "ORGANIZATION",
            "I-ORG": "ORGANIZATION",
            "B-ORG": "ORGANIZATION",
            "DATE": "DATE_TIME",
            "TIME": "DATE_TIME",
            "MONEY": "FINANCIAL",
            "CARDINAL": "NUMBER",
            "I-MISC": "MISC",
            "B-MISC": "MISC",
            # Add more mappings as needed
        }
        
        super().__init__(
            supported_entities=supported_entities,
            name="transformer_entity_recognizer",
        )
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        logger.info(f"Loading transformer model: {model_name}")
        try:
            # Initialize the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Set up NER pipeline
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Merge tokens with same entity
                device=0 if device == "cuda" and torch.cuda.is_available() else -1
            )
            logger.info(f"Successfully loaded model {model_name}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise
    
    def load(self) -> None:
        """Load the recognizer, already done in __init__."""
        pass
    
    def analyze(
        self, 
        text: str, 
        entities: List[str] = None, 
        nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using the transformer model and return recognized entities.
        
        Args:
            text: The text to analyze
            entities: Entity types to look for
            nlp_artifacts: NLP artifacts from NLP engine
            
        Returns:
            List of RecognizerResult objects
        """
        if not text:
            return []
        
        # Use only supported entities
        if entities:
            entities = [entity for entity in entities if entity in self.supported_entities]
        else:
            entities = self.supported_entities
            
        results = []
        
        try:
            # Try analysis with original text first
            ner_results = self.ner_pipeline(text)
            analysis_text = text
            
            # If no results with original text, try with smart capitalization
            if not ner_results:
                # Create a version with likely names capitalized
                analysis_text = self._apply_smart_capitalization(text)
                logger.info(f"No results with original text, trying smart capitalization: '{analysis_text}'")
                ner_results = self.ner_pipeline(analysis_text)
            
            # Convert NER results to Presidio RecognizerResult objects
            for item in ner_results:
                logger.info(f"Processing NER result: {item}")
                
                # Get the entity type and ensure it exists
                if 'entity' not in item and 'entity_group' in item:
                    # Some models use entity_group instead of entity
                    raw_entity = item["entity_group"]
                    entity_type = self.entity_mapping.get(item["entity_group"], item["entity_group"])
                elif 'entity' in item:
                    raw_entity = item["entity"]
                    entity_type = self.entity_mapping.get(item["entity"], item["entity"])
                else:
                    # Skip if no entity type found
                    logger.warning(f"No entity type found in result: {item}")
                    continue
                
                logger.info(f"Raw entity: {raw_entity} -> Mapped entity: {entity_type}")
                
                # Skip if the entity type is not in the requested entities
                if entity_type not in entities:
                    logger.info(f"Entity {entity_type} not in requested entities: {entities}")
                    continue
                    
                # Skip if confidence is below threshold
                if item["score"] < self.confidence_threshold:
                    logger.info(f"Score {item['score']} below threshold {self.confidence_threshold}")
                    continue
                
                logger.info(f"Entity {entity_type} passed all filters, creating result")
                
                # Map positions back to original text if we used modified text
                start_pos = item["start"]
                end_pos = item["end"]
                
                if analysis_text != text:
                    # Find the corresponding position in original text
                    detected_word = analysis_text[start_pos:end_pos]
                    logger.info(f"Mapping '{detected_word}' back to original text")
                    # Look for this word in the original text (case insensitive)
                    original_start = text.lower().find(detected_word.lower())
                    if original_start != -1:
                        start_pos = original_start
                        end_pos = original_start + len(detected_word)
                        logger.info(f"Mapped to position {start_pos}-{end_pos} in original text")
                    else:
                        # If we can't find it, skip this result
                        logger.warning(f"Could not map '{detected_word}' back to original text")
                        continue
                
                # Create a RecognizerResult
                result = RecognizerResult(
                    entity_type=entity_type,
                    start=start_pos,
                    end=end_pos,
                    score=item["score"],
                    recognition_metadata={
                        "source": f"transformer-{self.model_name}",
                        "model_entity": entity_type,
                    }
                )
                results.append(result)
                logger.info(f"Added result: {entity_type} at {start_pos}-{end_pos}")
                
                
        except Exception as e:
            logger.error(f"Error during transformer analysis: {e}")
            
        return results
    
    def _apply_smart_capitalization(self, text: str) -> str:
        """
        Apply smart capitalization to improve NER detection.
        This capitalizes words that are likely to be names or places.
        """
        import re
        
        # Common patterns that suggest names/places
        name_patterns = [
            r'\bmy name is ([a-z]+ [a-z]+)',  # "my name is john smith"
            r'\bi am ([a-z]+ [a-z]+)',       # "i am john smith"
            r'\bcalled ([a-z]+ [a-z]+)',     # "called john smith"
            r'\blive in ([a-z]+)',           # "live in london"
            r'\bfrom ([a-z]+)',              # "from london"
            r'\bin ([a-z]+)',                # "in london"
        ]
        
        result = text
        
        # Capitalize likely names and places
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                captured = match.group(1)
                capitalized = captured.title()
                result = result.replace(captured, capitalized)
        
        return result

# Example function to create an instance with a specific model
def create_transformer_recognizer(model_name: str = "dslim/bert-base-NER") -> TransformerEntityRecognizer:
    """
    Create a transformer-based entity recognizer with the specified model.
    
    Args:
        model_name: Name of the Hugging Face model to use
        
    Returns:
        TransformerEntityRecognizer instance
    """
    # You can customize the entity mapping based on the specific model used
    if "bert-base-NER" in model_name:
        entity_mapping = {
            # Simple entity labels from aggregated results
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "MISC": "MISC",
            # BIO-tagged entity labels from non-aggregated results
            "B-PER": "PERSON",
            "I-PER": "PERSON",
            "B-LOC": "LOCATION",
            "I-LOC": "LOCATION",
            "B-ORG": "ORGANIZATION",
            "I-ORG": "ORGANIZATION",
            "B-MISC": "MISC",
            "I-MISC": "MISC",
        }
    else:
        # Default mapping
        entity_mapping = None
        
    return TransformerEntityRecognizer(
        model_name=model_name,
        entity_mapping=entity_mapping,
        confidence_threshold=0.5,
    )
