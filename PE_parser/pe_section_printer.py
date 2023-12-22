import pefile

def print_pe_section(file_path):
    try:

        pe = pefile.PE(file_path)

        sections = pe.sections

        for section in sections:
            
            # print(f" \nSection Name: {section.Name.decode().rstrip('\x00')}") # Ascii
            print("\nSection Name : {}".format(section.Name.decode().rstrip('\x00')))
            
            print(f" Virtual Size: {section.Misc_VirtualSize}") # 메모리 섹션 크기
            print(f" Virtual Address: {hex(section.VirtualAddress)}") # RVA, optinal header의 base of code와 동일, 실제주소: Image Base+RVA
            print(f" Raw Size: {section.SizeOfRawData}") # 파일에서의 섹션 크기
            print(f" Raw Address: {hex(section.PointerToRawData)}") # 파일에서 섹션 시작 위치
            print(f" Characteristics: {hex(section.Characteristics)}") # 섹션의 정보 표시 , ex) 60000020 -> 읽고 쓰기가 가능한 코드 섹션
    except pefile.PEFormatError as e:
        print(f"Error: {e}")
