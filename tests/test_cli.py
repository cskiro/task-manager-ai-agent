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


def test_cli_without_args_shows_usage():
    """Running CLI without arguments should show usage message."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli"],
        capture_output=True,
        text=True
    )
    # Typer exits with code 2 for missing command
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert "Error: Missing command" in result.stderr
    assert "--help" in result.stderr


def test_interactive_command_explicit():
    """Interactive mode requires explicit command."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "interactive"],
        capture_output=True,
        text=True,
        input="quit\n"  # Immediate quit
    )
    assert result.returncode == 0
    assert "Task Manager AI Agent" in result.stdout or "Goodbye" in result.stdout