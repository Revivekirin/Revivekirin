import pefile
import sys
from ppdeep import hash_from_file

def calculate_fuzzy_hash(file_path):
    pe = pefile.PE(file_path)
    try:
        fuzzy_hash = hash_from_file(file_path)
        print(f"Fuzzy Hash for {file_path}: {fuzzy_hash}")
    except Exception as e:
        print(f"Error calculating fuzzy hash for {file_path}: {e}")
        

