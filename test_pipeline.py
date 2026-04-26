"""
Test script for Agent Vision Pipeline.
"""
import logging
from pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize pipeline
pipeline = Pipeline()

# Test with the actual test image
result = pipeline.analyze("test.jpg", mode="fast")

# Print results
print("\n=== Analysis Results ===")
print(f"Mode: {result.mode}")
print(f"Summary: {result.summary}")
print(f"Objects: {result.objects}")
print(f"Text Visible: {result.text_visible}")
print(f"Processing Time: {result.processing_sec}s")
print(f"Providers: {result.providers}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")