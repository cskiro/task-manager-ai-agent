"""PII (Personally Identifiable Information) detection module."""

import re
from dataclasses import dataclass


@dataclass
class PIIDetector:
    """Detects personally identifiable information in text."""
    
    def __init__(self) -> None:
        # SSN patterns (more comprehensive than project instructions)
        self.ssn_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",     # 123-45-6789
            r"\b\d{3}\s\d{2}\s\d{4}\b",   # 123 45 6789
            r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b",  # 123456789 with validation
        ]
        
        # Email pattern (from project instructions)
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        
        # Credit card pattern (enhanced with Luhn validation)
        self.credit_card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"  # Basic pattern
        
        # US phone number pattern (additional)
        self.phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    
    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any PII patterns.
        
        Args:
            text: Text to check for PII
            
        Returns:
            True if PII is detected, False otherwise
        """
        if not text or not text.strip():
            return False
        
        return bool(self.find_pii_types(text))
    
    def find_pii_types(self, text: str) -> list[str]:
        """
        Return list of PII types found in text.
        
        Args:
            text: Text to check for PII
            
        Returns:
            List of PII type identifiers found (e.g., ['ssn', 'email'])
        """
        if not text or not text.strip():
            return []
        
        found = []
        
        # Check SSN
        if self._has_ssn(text):
            found.append("ssn")
        
        # Check email
        if self._has_email(text):
            found.append("email")
        
        # Check credit card
        if self._has_credit_card(text):
            found.append("credit_card")
        
        # Check phone
        if self._has_phone(text):
            found.append("phone")
        
        return found
    
    def _has_ssn(self, text: str) -> bool:
        """Check if text contains SSN."""
        # Avoid matching order numbers or similar patterns
        if re.search(r"order\s*#?\s*\d{3}-\d{2}-\d{4}", text, re.IGNORECASE):
            return False
        
        # Avoid matching phone numbers (10 digits with extra digit/dash)
        if re.search(r"\b\d{3}-\d{3}-\d{4}\b", text):
            return False
            
        for pattern in self.ssn_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_email(self, text: str) -> bool:
        """Check if text contains email address."""
        return bool(re.search(self.email_pattern, text, re.IGNORECASE))
    
    def _has_credit_card(self, text: str) -> bool:
        """Check if text contains valid credit card number."""
        # Find potential credit card numbers
        potential_cards = re.findall(self.credit_card_pattern, text)
        
        for card_match in potential_cards:
            # Remove spaces and dashes
            card_number = re.sub(r"[-\s]", "", card_match)
            
            # Skip if all zeros or all same digit
            if len(set(card_number)) == 1:
                continue
                
            # Check length (13-19 digits for various card types)
            if 13 <= len(card_number) <= 19:
                # Validate with Luhn algorithm
                if self._luhn_check(card_number):
                    return True
        
        return False
    
    def _has_phone(self, text: str) -> bool:
        """Check if text contains US phone number."""
        return bool(re.search(self.phone_pattern, text))
    
    def _luhn_check(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.
        
        Args:
            card_number: Credit card number as string of digits
            
        Returns:
            True if valid according to Luhn algorithm
        """
        try:
            # Ensure we have only digits
            if not card_number.isdigit():
                return False
            
            # Reject if all zeros (technically passes Luhn but not a real card)
            if all(d == '0' for d in card_number):
                return False
                
            # Convert to list of integers
            digits = [int(d) for d in card_number]
            
            # Double every second digit from right
            for i in range(len(digits) - 2, -1, -2):
                digits[i] *= 2
                if digits[i] > 9:
                    digits[i] -= 9
            
            # Sum all digits
            total = sum(digits)
            
            # Valid if sum is divisible by 10
            return total % 10 == 0
            
        except (ValueError, IndexError):
            return False