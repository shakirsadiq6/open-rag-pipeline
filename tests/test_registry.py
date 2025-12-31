import pytest
from pathlib import Path
from open_rag_pipeline.pipeline.registry import HashRegistry

def test_hash_registry(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = HashRegistry(registry_path)
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    
    # Initially, it should show as changed (not in registry)
    assert registry.has_changed(test_file) is True
    
    # Update registry
    registry.update(test_file)
    assert registry.has_changed(test_file) is False
    
    # Modify file
    test_file.write_text("hello world modified")
    assert registry.has_changed(test_file) is True
    
    # Update again
    registry.update(test_file)
    assert registry.has_changed(test_file) is False
    
    # Verify persistence
    new_registry = HashRegistry(registry_path)
    assert str(test_file.absolute()) in new_registry.hashes
    assert new_registry.has_changed(test_file) is False

def test_hash_registry_clear(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = HashRegistry(registry_path)
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    registry.update(test_file)
    
    registry.clear()
    assert not registry_path.exists()
    assert registry.hashes == {}
