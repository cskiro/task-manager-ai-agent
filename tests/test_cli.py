"""Tests for CLI module."""

import subprocess
import sys


def test_help_command_does_not_crash():
    """Test that --help command runs without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "--help"],
        capture_output=True,
        text=True
    )
    
    # Should exit with 0 (success) not an error
    assert result.returncode == 0
    
    # Should contain expected help text
    assert "AI-powered task manager" in result.stdout
    assert "Usage:" in result.stdout