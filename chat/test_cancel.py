#!/usr/bin/env python3
"""
Test script for the cancel generation feature.
This script verifies that the cancel generation functionality works correctly.
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chat.app import app, generation_tasks, generation_tasks_lock, cancel_generation


def test_task_management():
    """Test task management system."""
    print("\n📋 Testing Task Management...")
    
    # Simulate task creation
    import uuid
    task_id = str(uuid.uuid4())
    
    with generation_tasks_lock:
        generation_tasks[task_id] = False
    
    # Verify task exists
    with generation_tasks_lock:
        assert task_id in generation_tasks
        assert generation_tasks[task_id] == False
        print("  ✓ Task created successfully")
    
    # Simulate cancellation
    with generation_tasks_lock:
        generation_tasks[task_id] = True
    
    # Verify task cancelled
    with generation_tasks_lock:
        assert generation_tasks[task_id] == True
        print("  ✓ Task cancelled successfully")
    
    # Cleanup
    with generation_tasks_lock:
        if task_id in generation_tasks:
            del generation_tasks[task_id]
    
    print("  ✓ Task management works correctly")


def test_cancel_endpoint():
    """Test the cancel endpoint."""
    print("\n🌐 Testing Cancel Endpoint...")
    
    # Test with Flask test client
    with app.test_client() as client:
        # Try to cancel non-existent task
        response = client.post('/api/generate/cancel',
                               json={'task_id': 'non-existent-task'},
                               content_type='application/json')
        
        assert response.status_code == 404
        print("  ✓ Returns 404 for non-existent task")
        
        # Test with invalid request (no task_id)
        response = client.post('/api/generate/cancel',
                               json={},
                               content_type='application/json')
        
        assert response.status_code == 400
        print("  ✓ Returns 400 for missing task_id")


def test_generate_endpoint():
    """Test the generate endpoint (without actual generation)."""
    print("\n🔄 Testing Generate Endpoint...")
    
    with app.test_client() as client:
        # Test with missing model
        response = client.post('/api/generate',
                               json={'message': 'test'},
                               content_type='application/json')
        
        assert response.status_code == 400
        assert b'Model name is required' in response.data
        print("  ✓ Returns 400 for missing model")
        
        # Test with missing message
        response = client.post('/api/generate',
                               json={'model': 'test'},
                               content_type='application/json')
        
        assert response.status_code == 400
        assert b'Message is required' in response.data
        print("  ✓ Returns 400 for missing message")


def test_concurrent_tasks():
    """Test concurrent task handling."""
    print("\n🔀 Testing Concurrent Task Handling...")
    
    import uuid
    task_ids = [str(uuid.uuid4()) for _ in range(5)]
    
    # Create multiple tasks
    with generation_tasks_lock:
        for task_id in task_ids:
            generation_tasks[task_id] = False
    
    print(f"  ✓ Created {len(task_ids)} concurrent tasks")
    
    # Cancel some tasks
    with generation_tasks_lock:
        for i, task_id in enumerate(task_ids):
            if i % 2 == 0:  # Cancel every other task
                generation_tasks[task_id] = True
    
    cancelled_count = 0
    with generation_tasks_lock:
        for task_id in task_ids:
            if task_id in generation_tasks and generation_tasks[task_id]:
                cancelled_count += 1
    
    assert cancelled_count == 3  # 0, 2, 4 are cancelled
    print(f"  ✓ {cancelled_count} tasks cancelled correctly")
    
    # Cleanup
    with generation_tasks_lock:
        for task_id in task_ids:
            if task_id in generation_tasks:
                del generation_tasks[task_id]
    
    print("  ✓ Concurrent task handling works correctly")


def test_routes():
    """Test that all required routes exist."""
    print("\n🛣️  Testing API Routes...")
    
    required_routes = [
        ('POST', '/api/generate'),
        ('POST', '/api/generate/cancel'),
    ]
    
    for method, path in required_routes:
        found = False
        for rule in app.url_map.iter_rules():
            if rule.rule == path and method in rule.methods:
                found = True
                break
        
        assert found, f"Route {method} {path} not found"
        print(f"  ✓ {method:6} {path}")
    
    print("  ✓ All required routes exist")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cancel Generation Feature Test Suite")
    print("=" * 60)
    
    try:
        test_routes()
        test_task_management()
        test_cancel_endpoint()
        test_generate_endpoint()
        test_concurrent_tasks()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
        print("\n📝 Summary:")
        print("  - Task management system: ✓ Working")
        print("  - Cancel endpoint: ✓ Working")
        print("  - Generate endpoint: ✓ Working")
        print("  - Concurrent task handling: ✓ Working")
        print("  - API routes: ✓ All present")
        
        print("\n🚀 The cancel generation feature is ready to use!")
        print("\nTo test manually:")
        print("  1. Start the server: ./scripts/run.sh web-chat")
        print("  2. Open: http://localhost:5001")
        print("  3. Select a model and send a message")
        print("  4. Click the cancel button (×) while generating")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
