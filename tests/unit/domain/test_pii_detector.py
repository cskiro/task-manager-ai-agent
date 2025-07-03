"""Tests for PII detection functionality."""

import pytest

from src.domain.safety.pii_detector import PIIDetector


class TestPIIDetector:
    """Test PII detection capabilities."""

    def test_detects_social_security_numbers(self):
        """Should detect various SSN formats."""
        detector = PIIDetector()
        
        # Test different SSN formats
        assert detector.contains_pii("My SSN is 123-45-6789")
        assert detector.contains_pii("SSN: 123456789")
        assert detector.contains_pii("Social Security: 123 45 6789")
        
        # Should not match similar patterns that aren't SSNs
        assert not detector.contains_pii("Order #123-45-6789")  # Order number
        # Note: Phone numbers ARE detected as PII (just not as SSN)
        assert detector.contains_pii("Phone: 123-456-7890")  # Phone number (detected as phone PII)
        assert "phone" in detector.find_pii_types("Phone: 123-456-7890")
        assert "ssn" not in detector.find_pii_types("Phone: 123-456-7890")
        assert not detector.contains_pii("12345678")  # Too short for SSN
        # Note: 10 digits could be a phone number
        assert detector.contains_pii("1234567890")  # Valid phone format
        assert "phone" in detector.find_pii_types("1234567890")
        # Note: 11 digits starting with 1 is valid US phone with country code
        assert detector.contains_pii("12345678901")  # Valid as 1-234-567-8901
        assert not detector.contains_pii("92345678901")  # Invalid - doesn't start with 1

    def test_detects_email_addresses(self):
        """Should detect email addresses."""
        detector = PIIDetector()
        
        # Valid emails
        assert detector.contains_pii("Contact me at john@example.com")
        assert detector.contains_pii("Email: jane.doe+tag@company.co.uk")
        assert detector.contains_pii("Send to user123@test-domain.org")
        
        # Invalid emails
        assert not detector.contains_pii("Not an email: john@")
        assert not detector.contains_pii("Missing domain: @example.com")
        assert not detector.contains_pii("No extension: user@domain")

    def test_detects_credit_card_numbers(self):
        """Should detect credit card numbers."""
        detector = PIIDetector()
        
        # Various credit card formats with VALID test numbers
        assert detector.contains_pii("Card: 4532-0151-1283-0366")  # Valid Visa with dashes
        assert detector.contains_pii("CC: 5425 2334 3010 9903")  # Valid Mastercard with spaces
        assert detector.contains_pii("Payment: 4532015112830366")  # No separators
        assert detector.contains_pii("378282246310005")  # Valid Amex (15 digits)
        
        # Should not match invalid patterns
        assert not detector.contains_pii("1234567890123456")  # Fails Luhn check
        assert not detector.contains_pii("4532-1234-5678")  # Too short
        assert not detector.contains_pii("4532-1234-5678-90123")  # Too long

    def test_detects_phone_numbers(self):
        """Should detect US phone numbers."""
        detector = PIIDetector()
        
        # Various phone formats
        assert detector.contains_pii("Call me at (555) 123-4567")
        assert detector.contains_pii("Phone: 555-123-4567")
        assert detector.contains_pii("Cell: 5551234567")
        assert detector.contains_pii("Contact: +1-555-123-4567")
        assert detector.contains_pii("Tel: 1.555.123.4567")
        
        # Should not match invalid patterns
        assert not detector.contains_pii("123-4567")  # Too short
        assert not detector.contains_pii("555-1234-56789")  # Too long

    def test_find_pii_types(self):
        """Should identify which types of PII are present."""
        detector = PIIDetector()
        
        # Single PII type
        assert detector.find_pii_types("My SSN is 123-45-6789") == ["ssn"]
        assert detector.find_pii_types("Email: test@example.com") == ["email"]
        
        # Multiple PII types
        text = "Contact John at john@example.com or 555-123-4567"
        pii_types = detector.find_pii_types(text)
        assert "email" in pii_types
        assert "phone" in pii_types
        assert len(pii_types) == 2
        
        # No PII
        assert detector.find_pii_types("This text has no PII") == []

    def test_handles_edge_cases(self):
        """Should handle edge cases gracefully."""
        detector = PIIDetector()
        
        # Empty or None inputs
        assert not detector.contains_pii("")
        assert not detector.contains_pii("   ")
        assert detector.find_pii_types("") == []
        
        # Case sensitivity
        assert detector.contains_pii("email: JOHN@EXAMPLE.COM")
        assert detector.contains_pii("My ssn is 123-45-6789")

    def test_performance_with_large_text(self):
        """Should handle large texts efficiently."""
        detector = PIIDetector()
        
        # Create large text with PII in the middle
        large_text = "Lorem ipsum " * 1000
        large_text += " My email is test@example.com "
        large_text += "dolor sit amet " * 1000
        
        assert detector.contains_pii(large_text)
        assert "email" in detector.find_pii_types(large_text)

    def test_luhn_validation_for_credit_cards(self):
        """Should validate credit cards using Luhn algorithm."""
        detector = PIIDetector()
        
        # Valid credit card numbers (pass Luhn check)
        valid_cards = [
            "4532015112830366",  # Visa
            "5425233430109903",  # Mastercard
            "374245455400126",   # Amex
            "6011000991300009",  # Discover
        ]
        
        for card in valid_cards:
            assert detector.contains_pii(f"Card: {card}"), f"Failed to detect valid card: {card}"
        
        # Invalid credit card numbers (fail Luhn check)
        invalid_cards = [
            "4532015112830367",  # Changed last digit
            "1234567890123456",  # Random numbers
            "0000000000000000",  # All zeros
        ]
        
        for card in invalid_cards:
            assert not detector.contains_pii(f"Card: {card}"), f"Incorrectly detected invalid card: {card}"