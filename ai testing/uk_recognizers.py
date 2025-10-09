#!/usr/bin/env python3
"""
UK-specific Recognizers for Presidio
This module contains custom pattern recognizers for UK-specific PII entities.
"""

from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional, Tuple

def create_uk_recognizers() -> List[PatternRecognizer]:
    """Create and return a list of UK-specific pattern recognizers"""
    recognizers = []
    
    # UK Postcode Recognizer
    uk_postcode_patterns = [
        Pattern(
            name="uk_postcode_standard",
            regex=r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b",
            score=0.9
        ),
        # Handle postcodes without space
        Pattern(
            name="uk_postcode_no_space",
            regex=r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\d[A-Z]{2}\b",
            score=0.85
        )
    ]
    uk_postcode_recognizer = PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=uk_postcode_patterns,
        context=["post", "code", "postcode", "address", "mail", "zip", "postal"]
    )
    recognizers.append(uk_postcode_recognizer)
    
    # UK Sort Code Recognizer
    uk_sort_code_patterns = [
        Pattern(
            name="uk_sort_code_standard",
            regex=r"\b\d{2}-\d{2}-\d{2}\b",
            score=0.9
        ),
        # Handle sort codes without hyphens
        Pattern(
            name="uk_sort_code_no_hyphens",
            regex=r"\b\d{6}\b",
            score=0.7
        )
    ]
    uk_sort_code_recognizer = PatternRecognizer(
        supported_entity="UK_SORT_CODE",
        patterns=uk_sort_code_patterns,
        context=["sort", "code", "bank", "account", "routing"]
    )
    recognizers.append(uk_sort_code_recognizer)
    
    # UK NHS Number Recognizer
    uk_nhs_patterns = [
        Pattern(
            name="uk_nhs_standard",
            regex=r"\b\d{3}\s\d{3}\s\d{4}\b",
            score=0.9
        ),
        # Handle NHS numbers without spaces
        Pattern(
            name="uk_nhs_no_spaces",
            regex=r"\b\d{10}\b",
            score=0.7
        )
    ]
    uk_nhs_recognizer = PatternRecognizer(
        supported_entity="UK_NHS_NUMBER",
        patterns=uk_nhs_patterns,
        context=["nhs", "health", "service", "medical", "patient"]
    )
    recognizers.append(uk_nhs_recognizer)
    
    # UK National Insurance Number Recognizer
    uk_ni_patterns = [
        Pattern(
            name="uk_ni_standard",
            regex=r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b",
            score=0.9
        ),
        # Handle NI numbers with or without spaces
        Pattern(
            name="uk_ni_no_spaces",
            regex=r"\b[A-Z]{2}\d{6}[A-Z]\b",
            score=0.85
        )
    ]
    uk_ni_recognizer = PatternRecognizer(
        supported_entity="UK_NI_NUMBER",
        patterns=uk_ni_patterns,
        context=["national", "insurance", "ni", "number", "social"]
    )
    recognizers.append(uk_ni_recognizer)
    
    # UK Bank Account Number Recognizer
    uk_bank_account_patterns = [
        Pattern(
            name="uk_bank_account_standard",
            regex=r"\b\d{8}\b",
            score=0.7
        ),
        # Handle account numbers with spaces
        Pattern(
            name="uk_bank_account_with_spaces",
            regex=r"\b\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\b",
            score=0.7
        )
    ]
    uk_bank_account_recognizer = PatternRecognizer(
        supported_entity="UK_BANK_ACCOUNT",
        patterns=uk_bank_account_patterns,
        context=["bank", "account", "number", "banking", "current", "savings"]
    )
    recognizers.append(uk_bank_account_recognizer)
    
    # UK Monetary Value Recognizer
    uk_monetary_patterns = [
        Pattern(
            name="uk_monetary_standard",
            regex=r"£\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            score=0.9
        ),
        # Handle monetary values with GBP prefix
        Pattern(
            name="uk_monetary_gbp",
            regex=r"GBP\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            score=0.85
        )
    ]
    uk_monetary_recognizer = PatternRecognizer(
        supported_entity="UK_MONETARY_VALUE",
        patterns=uk_monetary_patterns,
        context=["pounds", "sterling", "gbp", "£", "price", "cost", "payment"]
    )
    recognizers.append(uk_monetary_recognizer)
    
    # UK Passport Number Recognizer
    uk_passport_patterns = [
        Pattern(
            name="uk_passport_standard",
            regex=r"\b\d{9}\b",
            score=0.7
        ),
        # Handle passport numbers with context
        Pattern(
            name="uk_passport_with_prefix",
            regex=r"\b[Pp]assport\s+[Nn]o\.?\s*\d{9}\b",
            score=0.9
        )
    ]
    uk_passport_recognizer = PatternRecognizer(
        supported_entity="UK_PASSPORT",
        patterns=uk_passport_patterns,
        context=["passport", "travel", "document", "id", "identification"]
    )
    recognizers.append(uk_passport_recognizer)
    
    # Customer ID Recognizer
    customer_id_patterns = [
        Pattern(
            name="customer_id_standard",
            regex=r"\bCUS\d{6,8}\b",
            score=0.9
        ),
        # Handle customer IDs with different prefixes
        Pattern(
            name="customer_id_generic",
            regex=r"\b[A-Z]{2,4}\d{6,10}\b",
            score=0.7
        )
    ]
    customer_id_recognizer = PatternRecognizer(
        supported_entity="CUSTOMER_ID",
        patterns=customer_id_patterns,
        context=["customer", "client", "id", "number", "account", "reference"]
    )
    recognizers.append(customer_id_recognizer)
    
    # UK Phone Number Recognizer
    uk_phone_patterns = [
        Pattern(
            name="uk_phone_mobile",
            regex=r"\b07\d{3}\s?\d{3}\s?\d{3}\b",  # Mobile format: 07XXX XXX XXX
            score=0.9
        ),
        Pattern(
            name="uk_phone_landline", 
            regex=r"\b0[1-9]\d{2,4}\s?\d{3,7}\b",  # Landline format: 0XXXX XXXXXX
            score=0.85
        )
    ]
    uk_phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=uk_phone_patterns,
        context=["phone", "tel", "mobile", "call", "number", "contact"]
    )
    recognizers.append(uk_phone_recognizer)
    
    return recognizers

def register_uk_recognizers(analyzer) -> None:
    """Register UK-specific recognizers with the Presidio analyzer"""
    recognizers = create_uk_recognizers()
    for recognizer in recognizers:
        analyzer.registry.add_recognizer(recognizer)
    
    return
