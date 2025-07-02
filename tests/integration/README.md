# Integration Tests

This directory contains integration tests that use real external services.

## Running OpenAI Integration Tests

To run the OpenAI integration tests, you need to:

1. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

2. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. Run the integration tests:
   ```bash
   # Run all integration tests
   poetry run pytest tests/integration -v

   # Run only OpenAI tests
   poetry run pytest tests/integration/test_openai_planning.py -v

   # Run with output
   poetry run pytest tests/integration/test_openai_planning.py -v -s
   ```

## Test Markers

Integration tests are marked with `@pytest.mark.integration` and will be skipped if the required API keys are not set.

To run all tests except integration tests:
```bash
poetry run pytest -m "not integration"
```

## Cost Considerations

- The OpenAI integration tests use real API calls and will incur costs
- Each test run typically uses < 1000 tokens
- Tests use `gpt-4-turbo-preview` by default for best results
- You can modify the model in the test to use cheaper models like `gpt-3.5-turbo`

## Troubleshooting

If tests are skipped:
- Check that your `.env` file exists and contains `OPENAI_API_KEY`
- Ensure the API key is valid and has credits
- Check that `python-dotenv` is loading the environment variables

If tests fail:
- Check your OpenAI API quota and rate limits
- Ensure your internet connection is stable
- Review the error messages for API-specific issues