"""
Test script for DeepSeek V3 Web Chat API
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        from chat.app import app, list_checkpoints, load_model
        print("✓ Successfully imported chat.app")
    except Exception as e:
        print(f"✗ Failed to import chat.app: {e}")
        return False
    
    try:
        from config import load_config, get_device
        print("✓ Successfully imported config")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from deepseek.inference.inference import DeepSeekInference
        print("✓ Successfully imported DeepSeekInference")
    except Exception as e:
        print(f"✗ Failed to import DeepSeekInference: {e}")
        return False
    
    return True


def test_checkpoint_listing():
    """Test listing available checkpoints."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Listing")
    print("=" * 60)
    
    try:
        from chat.app import list_checkpoints
        
        checkpoints = list_checkpoints()
        print(f"✓ Found {len(checkpoints)} checkpoints")
        
        if len(checkpoints) > 0:
            print("\nAvailable checkpoints:")
            for name, info in list(checkpoints.items())[:5]:
                print(f"  - {name}")
                print(f"    Type: {info.get('type', 'unknown')}")
                print(f"    Size: {info.get('size_mb', 0)} MB")
        else:
            print("⚠ No checkpoints found in checkpoints/ directory")
        
        return True
    except Exception as e:
        print(f"✗ Failed to list checkpoints: {e}")
        return False


def test_model_loading():
    """Test loading a model."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        from chat.app import load_model, list_checkpoints
        
        checkpoints = list_checkpoints()
        
        if len(checkpoints) == 0:
            print("⚠ Skipping model loading test - no checkpoints available")
            return True
        
        # Load the first checkpoint
        model_name = list(checkpoints.keys())[0]
        print(f"Loading model: {model_name}")
        
        start_time = time.time()
        inference = load_model(model_name)
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        print(f"  Device: {inference.device}")
        print(f"  EOS token ID: {inference.eos_token_id}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)
    
    try:
        from chat.app import load_model, list_checkpoints, generate_stream
        
        checkpoints = list_checkpoints()
        
        if len(checkpoints) == 0:
            print("⚠ Skipping inference test - no checkpoints available")
            return True
        
        # Load the first checkpoint
        model_name = list(checkpoints.keys())[0]
        print(f"Loading model: {model_name}")
        
        inference = load_model(model_name)
        
        # Test streaming generation
        print("\nTesting streaming generation...")
        prompt = "Human: Hello\n\nAssistant:"
        
        start_time = time.time()
        
        # Use our custom streaming function
        full_response = ""
        token_count = 0
        for chunk in generate_stream(
            inference,
            prompt,
            max_new_tokens=30,  # Small limit for testing
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        ):
            import json
            data = json.loads(chunk)
            if data.get('token'):
                full_response += data['token']
                token_count += 1
            if data.get('done'):
                break
        
        generation_time = time.time() - start_time
        
        print(f"✓ Generation completed in {generation_time:.2f} seconds")
        print(f"  Tokens generated: {token_count}")
        print(f"  Response: {full_response[:200]}...")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_app():
    """Test Flask app configuration."""
    print("\n" + "=" * 60)
    print("Testing Flask App Configuration")
    print("=" * 60)
    
    try:
        from chat.app import app
        
        print(f"✓ Flask app created successfully")
        print(f"  Application name: {app.name}")
        print(f"  Debug mode: {app.debug}")
        print(f"  Threaded: {app.config.get('TESTING', False)}")
        
        # List routes
        print("\nAvailable routes:")
        for rule in app.url_map.iter_rules():
            print(f"  {rule.methods} {rule.rule}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test Flask app: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "DeepSeek V3 Web Chat API Test" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Checkpoint Listing", test_checkpoint_listing()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Inference", test_inference()))
    results.append(("Flask App", test_flask_app()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<10} {test_name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All tests passed! You can start the web server with:")
        print("  ./scripts/run.sh web-chat")
        print("  or")
        print("  python3 chat/app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
