"""
Test suite for loader_jolpica circuit mapping improvements.
"""

from __future__ import annotations
import pytest
import json
import tempfile
from pathlib import Path
from f1_predictor.data.loader_jolpica import JolpicaLoader, _BASE_CIRCUIT_TYPE_MAP


class TestJolpicaLoaderCircuitMapping:
    """Test circuit type mapping extensibility."""
    
    def setup_method(self):
        """Save original mapping before each test."""
        self.original_mapping = JolpicaLoader.circuit_type_map.copy()
    
    def teardown_method(self):
        """Restore original mapping after each test."""
        JolpicaLoader.circuit_type_map = self.original_mapping.copy()
    
    def test_base_mapping_exists(self):
        """Check that base mapping is loaded."""
        # circuit_type_map is a class attribute that copies base mapping
        assert len(JolpicaLoader.circuit_type_map) > 0
        # Should contain known circuits
        assert "monaco" in JolpicaLoader.circuit_type_map
        assert JolpicaLoader.circuit_type_map["monaco"] == "street"
    
    def test_add_circuit_type(self):
        """Test adding a new circuit mapping."""
        original_count = len(JolpicaLoader.circuit_type_map)
        # Add a new circuit
        JolpicaLoader.add_circuit_type("new_circuit", "mixed")
        assert "new_circuit" in JolpicaLoader.circuit_type_map
        assert JolpicaLoader.circuit_type_map["new_circuit"] == "mixed"
        assert len(JolpicaLoader.circuit_type_map) == original_count + 1
        
        # Update existing circuit
        JolpicaLoader.add_circuit_type("monaco", "high_df")
        assert JolpicaLoader.circuit_type_map["monaco"] == "high_df"
    
    def test_add_circuit_type_invalid_type(self):
        """Test validation of circuit_type."""
        with pytest.raises(ValueError, match="Invalid circuit_type"):
            JolpicaLoader.add_circuit_type("test", "invalid_type")
    
    def test_load_circuit_types_from_json(self, tmp_path):
        """Test loading mappings from JSON file."""
        # Create temporary JSON file
        json_content = {
            "new_circuit_1": "street",
            "new_circuit_2": "high_speed",
            "monaco": "desert",  # Override existing
        }
        json_file = tmp_path / "circuit_types.json"
        json_file.write_text(json.dumps(json_content))
        
        original_count = len(JolpicaLoader.circuit_type_map)
        JolpicaLoader.load_circuit_types_from_json(str(json_file))
        
        # New circuits added
        assert "new_circuit_1" in JolpicaLoader.circuit_type_map
        assert JolpicaLoader.circuit_type_map["new_circuit_1"] == "street"
        assert "new_circuit_2" in JolpicaLoader.circuit_type_map
        assert JolpicaLoader.circuit_type_map["new_circuit_2"] == "high_speed"
        # Override worked
        assert JolpicaLoader.circuit_type_map["monaco"] == "desert"
        # Count increased by 2 (one override, two new)
        assert len(JolpicaLoader.circuit_type_map) == original_count + 2
    
    def test_circuit_type_map_isolation(self):
        """Ensure circuit_type_map is independent of base mapping after modifications."""
        # Make a copy of current mapping
        current_map = JolpicaLoader.circuit_type_map.copy()
        # Modify base mapping (should not affect class attribute)
        _BASE_CIRCUIT_TYPE_MAP["test"] = "street"
        # Class attribute should not have changed
        assert "test" not in JolpicaLoader.circuit_type_map
        # Clean up base mapping
        del _BASE_CIRCUIT_TYPE_MAP["test"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])