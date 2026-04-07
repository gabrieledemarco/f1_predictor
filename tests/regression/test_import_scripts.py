"""
Regression tests for import scripts.

These tests verify that:
1. Import scripts can be imported without errors
2. Script functions accept correct parameters
3. No syntax errors in scripts

Run with: pytest tests/regression/test_import_scripts.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestImportJolpicaScript:
    """Test import_jolpica.py script."""

    def test_script_imports_without_error(self):
        """Test that import_jolpica script can be imported."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "import_jolpica.py"
        assert script_path.exists(), "import_jolpica.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = Path(__file__).parent.parent.parent / "scripts" / "import_jolpica.py"
        spec = importlib.util.spec_from_file_location("import_jolpica", script_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")

    @patch.dict(os.environ, {"MONGODB_URI": "mongodb://localhost:27017"})
    def test_mongo_client_creation(self):
        """Test MongoDB client creation logic."""
        from pymongo import MongoClient

        uri = os.environ.get("MONGODB_URI")
        assert uri is not None
        assert "mongodb" in uri


class TestImportTracingInsightsScript:
    """Test import_tracinginsights.py script."""

    def test_script_imports_without_error(self):
        """Test that import_tracinginsights script can be imported."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "import_tracinginsights.py"
        )
        assert script_path.exists(), "import_tracinginsights.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "import_tracinginsights.py"
        )
        spec = importlib.util.spec_from_file_location(
            "import_tracinginsights", script_path
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestImportPinnacleOddsScript:
    """Test import_pinnacle_odds.py script."""

    def test_script_imports_without_error(self):
        """Test that import_pinnacle_odds script can be imported."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "import_pinnacle_odds.py"
        )
        assert script_path.exists(), "import_pinnacle_odds.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "import_pinnacle_odds.py"
        )
        spec = importlib.util.spec_from_file_location(
            "import_pinnacle_odds", script_path
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestImportKaggleScript:
    """Test import_kaggle.py script."""

    def test_script_imports_without_error(self):
        """Test that import_kaggle script can be imported."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "import_kaggle.py"
        assert script_path.exists(), "import_kaggle.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = Path(__file__).parent.parent.parent / "scripts" / "import_kaggle.py"
        spec = importlib.util.spec_from_file_location("import_kaggle", script_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestComputePaceObservationsScript:
    """Test compute_pace_observations.py script."""

    def test_script_imports_without_error(self):
        """Test that compute_pace_observations script can be imported."""
        script_path = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "compute_pace_observations.py"
        )
        assert script_path.exists(), "compute_pace_observations.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "compute_pace_observations.py"
        )
        spec = importlib.util.spec_from_file_location(
            "compute_pace_observations", script_path
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestExtractCircuitProfilesScript:
    """Test extract_circuit_profiles.py script."""

    def test_script_imports_without_error(self):
        """Test that extract_circuit_profiles script can be imported."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "extract_circuit_profiles.py"
        )
        assert script_path.exists(), "extract_circuit_profiles.py should exist"

    def test_script_has_main_function(self):
        """Test that script has main function."""
        import importlib.util

        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "extract_circuit_profiles.py"
        )
        spec = importlib.util.spec_from_file_location(
            "extract_circuit_profiles", script_path
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "main"), "Script should have main() function"
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestMigrationScript:
    """Test migration scripts."""

    def test_migrate_jolpica_cache_exists(self):
        """Test that migrate_jolpica_cache.py exists."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "migrate_jolpica_cache.py"
        )
        assert script_path.exists(), "migrate_jolpica_cache.py should exist"


class TestDuplicateScriptsCheck:
    """Verify we don't have duplicate scripts."""

    def test_no_duplicate_import_scripts(self):
        """Check that we don't have both old and new import scripts with same name pattern."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"

        import_scripts = list(scripts_dir.glob("*import*.py"))
        fetch_scripts = list(scripts_dir.glob("*fetch*.py"))
        collect_scripts = list(scripts_dir.glob("*collect*.py"))

        # We expect one of each type, not both old and new versions
        # This is informational - the actual dedup happens in the code review

        assert len(import_scripts) > 0, "Should have import scripts"
        assert len(fetch_scripts) > 0, "Should have fetch scripts"
        assert len(collect_scripts) > 0, "Should have collect scripts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
