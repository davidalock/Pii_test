#!/usr/bin/env python3
"""
Quick script to list all supported Presidio entities.
"""

from presidio_analyzer import AnalyzerEngine

def list_entities():
    analyzer = AnalyzerEngine()
    entities = analyzer.get_supported_entities()
    
    print("PRESIDIO SUPPORTED ENTITIES:")
    print("=" * 40)
    for i, entity in enumerate(sorted(entities), 1):
        print(f"{i:2d}. {entity}")
    
    print(f"\nTotal: {len(entities)} entity types")
    
    return entities

if __name__ == "__main__":
    list_entities()
