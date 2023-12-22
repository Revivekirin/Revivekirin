import os
import sys


#모듈 import
from entropy_calculator import calculate_entropy
from pe_header_printer import *
from pe_section_printer import *
from imports_and_exports_printer import *
from resource_printer import *
from relocation_printer import *
from tls_printer import *
from hash_calculator import *
from exe_finder import *
from rich_header_printer import *
from certificate_info import print_certificate_info
from print_json import *
from string_printer import *

if __name__ == "__main__":
    # 테스트를 위한 PE 파일 경로
    file_path = input("Enter the path of the PE file: ")   
    
    save_json_result(file_path)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
        
    entropy_value = calculate_entropy(file_path)
    print(f"Entropy: {entropy_value}")
    print_pe_header(file_path)
    print_pe_section(file_path)
    print_imports_and_exports(file_path) 
    print_resource(file_path) 
    print_relocation(file_path)
    print_tls(file_path)
    print_hash(file_path)
    print_rich_header(file_path)
    print_strings(file_path)