#!/usr/bin/env python3
"""
Quick demo script to verify web chat functionality
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("\n" + "=" * 70)
    print("DeepSeek V3 Web Chat - Quick Demo")
    print("=" * 70 + "\n")
    
    # Import
    print("1. Importing modules...")
    try:
        from chat.app import app, list_checkpoints, load_model
        print("   ✓ All modules imported successfully\n")
    except Exception as e:
        print(f"   ✗ Import failed: {e}\n")
        return 1
    
    # List checkpoints
    print("2. Scanning checkpoints...")
    checkpoints = list_checkpoints()
    print(f"   ✓ Found {len(checkpoints)} models:\n")
    
    for name, info in list(checkpoints.items())[:4]:
        print(f"      • {name}")
        print(f"        Type: {info['type']}")
        print(f"        Size: {info['size_mb']} MB\n")
    
    # Test model loading (just the smallest one for speed)
    print("3. Testing model loading (using smallest checkpoint)...")
    smallest = min(checkpoints.items(), key=lambda x: x[1]['size_mb'])
    print(f"   Loading: {smallest[0]}")
    
    try:
        import time
        start = time.time()
        inference = load_model(smallest[0])
        load_time = time.time() - start
        print(f"   ✓ Model loaded in {load_time:.2f} seconds\n")
    except Exception as e:
        print(f"   ✗ Loading failed: {e}\n")
        return 1
    
    # Show API routes
    print("4. API endpoints:")
    routes = [
        ("GET", "/api/models", "List all available models"),
        ("GET", "/api/models/<name>/status", "Check model loading status"),
        ("POST", "/api/generate", "Generate text with streaming"),
        ("POST", "/api/chat/clear", "Clear chat history"),
        ("POST", "/api/models/<name>/unload", "Unload model from memory"),
    ]
    for method, path, desc in routes:
        print(f"   {method:<6} {path:<35} - {desc}")
    
    print("\n" + "=" * 70)
    print("✓ All checks passed!")
    print("=" * 70)
    print("\nTo start the web server, run:")
    print("  ./scripts/run.sh web-chat")
    print("  or")
    print("  python3 chat/app.py")
    print("\nThen open your browser to: http://localhost:5001")
    print("=" * 70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
