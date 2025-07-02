"""Tests for agent integration in CLI."""

import subprocess
import sys


def test_agent_command_exists():
    """Test that agent command exists in CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "agent" in result.stdout.lower()


def test_agent_plan_command():
    """Test agent plan subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "agent", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "plan" in result.stdout.lower()
