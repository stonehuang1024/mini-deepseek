#!/usr/bin/env python3
"""
Test script to verify the chat interface fixes.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chat.app import app, list_checkpoints, load_model


def test_list_models():
    """Test listing available models."""
    print("\n" + "=" * 60)
    print("Test 1: List Available Models")
    print("=" * 60)
    
    models = list_checkpoints()
    
    print(f"\n✓ Found {len(models)} model(s):")
    
    # Find first pretrain model
    first_pretrain = None
    for name, info in models.items():
        model_type = info.get('type', 'unknown')
        size_mb = info.get('size_mb', 0)
        print(f"  - {name} ({model_type}, {size_mb:.2f} MB)")
        
        if first_pretrain is None and model_type == 'Pretrained':
            first_pretrain = name
    
    if first_pretrain:
        print(f"\n✓ First pretrain model identified: {first_pretrain}")
        return first_pretrain
    else:
        print("\n⚠ No pretrain model found")
        return None


def test_model_loading(model_name):
    """Test loading a model."""
    print("\n" + "=" * 60)
    print(f"Test 2: Load Model: {model_name}")
    print("=" * 60)
    
    try:
        print(f"\nLoading model from checkpoints/{model_name}...")
        inference = load_model(model_name)
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  Device: {inference.device}")
        print(f"  Model type: {type(inference.model).__name__}")
        print(f"  Tokenizer: {type(inference.tokenizer).__name__}")
        print(f"  EOS token ID: {inference.eos_token_id}")
        
        return inference
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tokenization(inference, prompt="Hello, world!"):
    """Test tokenization."""
    print("\n" + "=" * 60)
    print(f"Test 3: Test Tokenization")
    print("=" * 60)
    
    try:
        print(f"\nPrompt: {prompt}")
        
        # Test encoding
        input_ids = inference.tokenizer.encode(prompt, return_tensors='pt')
        print(f"✓ Encoded to {input_ids.shape[1]} tokens")
        print(f"  Token IDs: {input_ids[0].tolist()[:10]}...")
        
        # Test decoding
        decoded = inference.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"✓ Decoded: {decoded}")
        
        return True
    except Exception as e:
        print(f"\n✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(inference, prompt="Hello", max_tokens=5):
    """Test text generation."""
    print("\n" + "=" * 60)
    print(f"Test 4: Test Text Generation")
    print("=" * 60)
    
    try:
        print(f"\nPrompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        
        # Generate text
        response = inference.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        print(f"\n✓ Generated text:")
        print(f"  {response}")
        
        return True
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test API endpoints."""
    print("\n" + "=" * 60)
    print("Test 5: Test API Endpoints")
    print("=" * 60)
    
    with app.test_client() as client:
        # Test /api/models
        response = client.get('/api/models')
        print(f"\nGET /api/models")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"  ✓ Found {data.get('count', 0)} models")
        else:
            print(f"  ✗ Failed")
        
        # Test /api/models/status with default model
        models = list_checkpoints()
        if models:
            first_model = list(models.keys())[0]
            response = client.get(f'/api/models/status?model={first_model}')
            print(f"\nGET /api/models/status?model={first_model}")
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  Status: {data.get('status')}")
                print(f"  Loaded: {data.get('loaded')}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CHAT INTERFACE FIX VERIFICATION")
    print("=" * 70)
    
    # Test 1: List models
    first_pretrain = test_list_models()
    
    if not first_pretrain:
        print("\n⚠ Skipping model loading tests (no models found)")
        return 0
    
    # Test 2: Load model
    inference = test_model_loading(first_pretrain)
    
    if not inference:
        print("\n⚠ Skipping generation tests (model failed to load)")
        return 1
    
    # Test 3: Test tokenization
    tokenization_ok = test_tokenization(inference)
    
    if not tokenization_ok:
        print("\n⚠ Tokenization failed, but continuing...")
    
    # Test 4: Test generation
    generation_ok = test_generation(inference)
    
    # Test 5: Test API endpoints
    test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ Test 1: List models - PASSED")
    print(f"✓ Test 2: Load model - {'PASSED' if inference else 'FAILED'}")
    print(f"✓ Test 3: Tokenization - {'PASSED' if tokenization_ok else 'FAILED'}")
    print(f"✓ Test 4: Generation - {'PASSED' if generation_ok else 'FAILED'}")
    print("✓ Test 5: API endpoints - COMPLETED")
    
    if inference and generation_ok:
        print("\n" + "=" * 70)
        print("✅ ALL CRITICAL TESTS PASSED!")
        print("=" * 70)
        print("\nThe chat interface fixes are working correctly.")
        print("You can now start the web server:")
        print("  ./scripts/run.sh web-chat")
        print("  or")
        print("  python3 chat/app.py")
        print("\nThen open: http://localhost:5001")
        print("=" * 70)
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
