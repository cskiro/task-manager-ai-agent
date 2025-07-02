"""Tests for CLI module."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


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


def test_save_command_exists():
    """Test that save command exists in CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert "save" in result.stdout
    assert result.returncode == 0


def test_load_command_exists():
    """Test that load command exists in CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interface.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert "load" in result.stdout
    assert result.returncode == 0


def test_save_command_handles_no_project():
    """Test save command when no project is planned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_project.json"
        
        # Try to save without planning first
        result = subprocess.run(
            [sys.executable, "-m", "src.interface.cli", "save", 
             str(save_path)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "No project planned yet" in result.stdout
        assert not save_path.exists()


def test_load_command_loads_project():
    """Test loading a project from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample project file
        save_path = Path(tmpdir) / "test_project.json"
        project_data = {
            "tasks": [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "title": "Setup project",
                    "description": "Initialize the project",
                    "priority": 3,
                    "status": "pending",
                    "dependencies": [],
                    "time_estimate": {
                        "optimistic": 1.0,
                        "realistic": 2.0,
                        "pessimistic": 3.0
                    }
                }
            ]
        }
        save_path.write_text(json.dumps(project_data, indent=2))
        
        # Load the project
        result = subprocess.run(
            [sys.executable, "-m", "src.interface.cli", "load", 
             str(save_path)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Setup project" in result.stdout
        assert "loaded" in result.stdout.lower()