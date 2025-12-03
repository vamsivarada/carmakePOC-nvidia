#!/usr/bin/env python3
"""
Post-processing fixes for common license plate OCR errors
"""

import re

class LicensePlateCorrector:
    """
    Corrects common OCR misreads in license plates
    """
    
    def __init__(self):
        # Common character confusions in OCR
        self.confusions = {
            # Letter vs Number confusions
            'O': '0',  # Letter O often confused with zero
            'I': '1',  # Letter I often confused with one
            'S': '5',  # Letter S confused with five
            'Z': '2',  # Letter Z confused with two
            'B': '8',  # Letter B confused with eight
            'G': '6',  # Letter G confused with six
        }
        
        # UK/US plate patterns (adjust for your region)
        # UK: 2 letters, 2 numbers, space, 3 letters (e.g., PF16 XEH)
        self.uk_pattern = re.compile(r'^([A-Z]{2})(\d{2})\s?([A-Z]{3})$')
        
    def correct_uk_plate(self, plate_text):
        """
        Correct a UK-format license plate
        Expected: 2 letters + 2 digits + 3 letters
        """
        # Remove spaces for processing
        clean = plate_text.replace(' ', '').upper()
        
        if len(clean) != 7:
            print(f"  ⚠️  Unusual length: {len(clean)} chars (expected 7)")
        
        # Try to parse as UK format: LLDDLLL
        # L = Letter, D = Digit
        
        corrected = list(clean)
        
        # Positions 0-1: Should be LETTERS
        for i in [0, 1]:
            if i < len(corrected) and corrected[i].isdigit():
                # Try to convert number to letter
                if corrected[i] == '0':
                    corrected[i] = 'O'
                elif corrected[i] == '1':
                    corrected[i] = 'I'
                elif corrected[i] == '5':
                    corrected[i] = 'S'
                print(f"  🔧 Position {i}: Converted digit to letter: {corrected[i]}")
        
        # Positions 2-3: Should be DIGITS
        for i in [2, 3]:
            if i < len(corrected) and corrected[i].isalpha():
                # Try to convert letter to digit
                if corrected[i] == 'O':
                    corrected[i] = '0'
                elif corrected[i] == 'I':
                    corrected[i] = '1'
                elif corrected[i] == 'S':
                    corrected[i] = '5'
                elif corrected[i] == 'Z':
                    corrected[i] = '2'
                elif corrected[i] == 'B':
                    corrected[i] = '8'
                elif corrected[i] == 'G':
                    corrected[i] = '6'
                print(f"  🔧 Position {i}: Converted letter to digit: {corrected[i]}")
        
        # Positions 4-6: Should be LETTERS
        for i in [4, 5, 6]:
            if i < len(corrected) and corrected[i].isdigit():
                # Try to convert number to letter
                if corrected[i] == '0':
                    corrected[i] = 'O'
                elif corrected[i] == '1':
                    corrected[i] = 'I'
                elif corrected[i] == '5':
                    corrected[i] = 'S'
                print(f"  🔧 Position {i}: Converted digit to letter: {corrected[i]}")
        
        result = ''.join(corrected)
        
        # Format with space: LLDD LLL
        if len(result) == 7:
            result = f"{result[0:2]}{result[2:4]} {result[4:7]}"
        
        return result
    
    def fix_missing_chars(self, plate_text, expected_length=7):
        """
        Try to detect and fix missing characters
        Common issue: 'I6' detected as just '6'
        """
        clean = plate_text.replace(' ', '').upper()
        
        if len(clean) >= expected_length:
            return plate_text  # No fix needed
        
        print(f"  ⚠️  Detected {len(clean)} chars, expected {expected_length}")
        print(f"  🔍 Analyzing: {plate_text}")
        
        # UK plate should be: 2 letters + 2 digits + 3 letters
        # If we have 6 chars instead of 7, we're missing one
        
        if len(clean) == 6:
            # Common case: Missing 'I' before a digit in positions 2-3
            # Example: PF6XEH should be PFI6XEH
            
            # Check if position 2 is a digit
            if len(clean) >= 3 and clean[2].isdigit():
                # Try inserting 'I' before the digit
                fixed = clean[:2] + 'I' + clean[2:]
                print(f"  🔧 Inserted 'I' before digit: {fixed}")
                
                # Format with space
                if len(fixed) == 7:
                    fixed = f"{fixed[0:2]}{fixed[2:4]} {fixed[4:7]}"
                
                return fixed
        
        return plate_text  # Return original if can't fix
    
    def process(self, plate_text, region='UK'):
        """
        Main processing function
        """
        print(f"\n{'='*60}")
        print(f"Processing: {plate_text}")
        print(f"{'='*60}")
        
        original = plate_text
        
        # Step 1: Fix missing characters
        plate_text = self.fix_missing_chars(plate_text)
        
        # Step 2: Apply regional corrections
        if region.upper() == 'UK':
            plate_text = self.correct_uk_plate(plate_text)
        
        if plate_text != original:
            print(f"✅ Corrected: {original} → {plate_text}")
        else:
            print(f"ℹ️  No changes needed")
        
        print(f"{'='*60}\n")
        
        return plate_text


def main():
    """Test the corrector with examples"""
    corrector = LicensePlateCorrector()
    
    # Test cases
    test_plates = [
        "PF6XEH",      # Should be PFI6XEH (missing I)
        "PF16XEH",     # Correct, should not change
        "ABC123",      # Different format
        "PF0GXEH",     # O/0 and G/6 confusions
    ]
    
    print("\n" + "="*60)
    print("License Plate OCR Corrector - Test Cases")
    print("="*60)
    
    for plate in test_plates:
        corrected = corrector.process(plate, region='UK')
        print(f"Original:  {plate}")
        print(f"Corrected: {corrected}\n")


if __name__ == '__main__':
    main()