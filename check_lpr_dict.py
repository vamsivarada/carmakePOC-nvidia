#!/usr/bin/env python3
"""
Check and validate the LPRNet dictionary file
"""

import os

dict_file = "dict.txt"
config_dict_file = "/workspaces/numberplate-detection/python_lpr/config/dict_us.txt"

# Standard US license plate characters
EXPECTED_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',  # Letters
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def check_dict_file(filepath):
    """Check if dictionary file exists and has all expected characters"""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist!")
        return None
    
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    # Parse the dictionary - could be one char per line or all in one line
    chars = []
    if '\n' in content:
        # One char per line
        chars = [line.strip() for line in content.split('\n') if line.strip()]
    else:
        # All chars in one line
        chars = list(content)
    
    print(f"✓ File exists")
    print(f"✓ Contains {len(chars)} characters")
    print(f"\nCharacters found:")
    print(''.join(chars))
    
    # Check for missing expected characters
    missing = [char for char in EXPECTED_CHARS if char not in chars]
    extra = [char for char in chars if char not in EXPECTED_CHARS and char not in [' ', '-']]
    
    if missing:
        print(f"\n⚠️  Missing characters: {missing}")
        print(f"   Particularly important: {'I' if 'I' in missing else '✓ Has I'}")
    else:
        print(f"\n✅ All expected characters present!")
        print(f"   ✓ Has 'I' (letter I)")
    
    if extra:
        print(f"\n📝 Extra characters (spaces, special chars): {extra}")
    
    return chars

def create_correct_dict(filepath="dict_us_correct.txt"):
    """Create a corrected dictionary file"""
    print(f"\n{'='*60}")
    print(f"Creating corrected dictionary: {filepath}")
    print(f"{'='*60}")
    
    with open(filepath, 'w') as f:
        # Write each character on its own line
        for char in EXPECTED_CHARS:
            f.write(char + '\n')
    
    print(f"✅ Created {filepath} with {len(EXPECTED_CHARS)} characters")
    print(f"Characters: {''.join(EXPECTED_CHARS)}")
    
    return filepath

def main():
    print("\n" + "="*60)
    print("LPRNet Dictionary Diagnostic Tool")
    print("="*60)
    
    # Check common dictionary file locations
    files_to_check = [
        "dict.txt",
        "dict_us.txt",
        "/workspaces/numberplate-detection/python_lpr/config/dict_us.txt",
        "/opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/dict_us.txt"
    ]
    
    found_dicts = []
    for filepath in files_to_check:
        if os.path.exists(filepath):
            chars = check_dict_file(filepath)
            if chars:
                found_dicts.append((filepath, chars))
    
    if not found_dicts:
        print("\n⚠️  No dictionary files found in common locations!")
    
    # Create a corrected version
    print("\n")
    new_dict = create_correct_dict("dict_us_correct.txt")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. Check your sgie_lpr_us_config.txt file")
    print("2. Look for the 'dictionary-file' parameter")
    print("3. Make sure it points to a dictionary with all characters")
    print("4. Use the generated dict_us_correct.txt if needed")
    print(f"\n   Example config line:")
    print(f"   dictionary-file=/path/to/dict_us_correct.txt")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()