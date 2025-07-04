"""Tests for model selection precedence in CLI."""

import os
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from src.interface.cli import app

runner = CliRunner()


class TestModelPrecedence:
    """Test that CLI arguments take precedence over environment variables."""
    
    def test_cli_argument_overrides_env_variable(self):
        """CLI --model flag should override OPENAI_MODEL env var."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }):
            with patch("src.interface.cli.OpenAIProvider") as mock_provider:
                # Mock the LLM provider to avoid actual API calls
                mock_instance = MagicMock()
                mock_provider.return_value = mock_instance
                
                # Run with explicit model
                result = runner.invoke(app, [
                    "plan",
                    "Build a test API",
                    "--model", "gpt-4"
                ])
                
                # Should use CLI argument, not env var
                mock_provider.assert_called_with(
                    api_key="test-key",
                    model="gpt-4"  # CLI argument should win
                )
    
    def test_env_variable_used_when_no_cli_argument(self):
        """OPENAI_MODEL env var should be used when no --model flag."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key", 
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }):
            with patch("src.interface.cli.OpenAIProvider") as mock_provider:
                mock_instance = MagicMock()
                mock_provider.return_value = mock_instance
                
                # Run without --model flag
                result = runner.invoke(app, [
                    "plan",
                    "Build a test API"
                ])
                
                # Should use env var
                mock_provider.assert_called_with(
                    api_key="test-key",
                    model="gpt-3.5-turbo"
                )
    
    def test_default_model_when_no_env_or_cli(self):
        """Default model should be used when neither CLI nor env specified."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key"
        }, clear=True):
            # Remove OPENAI_MODEL from env
            os.environ.pop("OPENAI_MODEL", None)
            
            with patch("src.interface.cli.OpenAIProvider") as mock_provider:
                mock_instance = MagicMock()
                mock_provider.return_value = mock_instance
                
                # Run without --model flag
                result = runner.invoke(app, [
                    "plan", 
                    "Build a test API"
                ])
                
                # Should use default (o3)
                mock_provider.assert_called_with(
                    api_key="test-key",
                    model="o3"
                )
    
    def test_console_output_shows_correct_model(self):
        """Console should display the actual model being used."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }):
            with patch("src.interface.cli.OpenAIProvider"):
                with patch("src.interface.cli.StreamingPlanRunner"):
                    # Test with CLI override
                    result = runner.invoke(app, [
                        "plan",
                        "Test project",
                        "--model", "gpt-4"
                    ])
                    
                    # Should show gpt-4, not gpt-3.5-turbo
                    assert "Using OpenAI gpt-4" in result.output
                    assert "Using OpenAI gpt-3.5-turbo" not in result.output
    
    def test_welcome_wizard_passes_none_for_model(self):
        """Welcome wizard should pass None for model to use env/default."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }):
            with patch("src.interface.cli._plan_project") as mock_plan:
                with patch("src.interface.cli.Prompt.ask", return_value="1"):
                    with patch("src.interface.cli._show_progress_theater"):
                        with patch("src.interface.cli._show_success_celebration"):
                            runner.invoke(app, [])  # No command = welcome wizard
                            
                            # Welcome wizard should pass None for model
                            mock_plan.assert_called()
                            _, _, _, model = mock_plan.call_args[0]
                            assert model is None  # Will use env/default in _plan_project