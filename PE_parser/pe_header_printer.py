import pefile
import datetime
from datetime import timezone

def print_pe_header(file_path):
    try:
        pe = pefile.PE(file_path)

        # DOS Header
        print(f"\nDOS Header: ")
        print(f" e_magic: {hex(pe.DOS_HEADER.e_magic)}") # 4D 5A (MZ), PE파일 구조
        print(f" e_lfanew: {hex(pe.DOS_HEADER.e_lfanew)}") # IMAG_NT_HEADER 시작주소
        print(f"\nNT Header: ")
        print(f" Signature: {hex(pe.NT_HEADERS.Signature)}")
        
        # File Header
        print(f"\n File Header: ")
        print(f"  Machine: {hex(pe.FILE_HEADER.Machine)}") # ex) 014C-intel CPU
        print(f"  Number of Sections: {pe.FILE_HEADER.NumberOfSections}") # Section 개수
        #timestamp 출력을 위해 코드 추가
        #기존 코드 : print(f"  Time Date Stamp: {pe.FILE_HEADER.TimeDateStamp}") # 시간차이가 미래일 경우 악성코드일 수 있음
        timestamp = pe.FILE_HEADER.TimeDateStamp
        human_readable_timestamp = datetime.datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Time Date Stamp: {human_readable_timestamp}")
        # Size of Optional Header 생략
        print(f"  Characteristics: {hex(pe.FILE_HEADER.Characteristics)}") # 파일 형식 정보, 32bit - win32, 64bit - winnt.h 문서 참고
        
        # Optional Header
        print(f"\n Optional Header: ")
        # Magic 추가
        #print(f"  Magic: {hex(pe.OPTIONAL_HEADER.Magic)}")
        # Magic이 몇 비트 파일인지 수정
        magic = pe.OPTIONAL_HEADER.Magic
        if magic == 0x20b:
            print(" Magic: 0x20b, 64-bit File")
        elif magic == 0x10b:
            print(" Magic: 0x10b, 32-bit File")
        else:
            print(f" Magic: {hex(magic)}, Unknown architecture")
        # Size of code 추가
        print(f"  SizeOfCode: {hex(pe.OPTIONAL_HEADER.SizeOfCode)}")
        print(f"  Address Entry point: {hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)}") # 프로그램 시작 주소, 파일싫애시 Image Base + Address of Entry Point에서 시작
        # Base of code (RVA 시작주소) 추가
        print(f"  BaseOfCode: {hex(pe.OPTIONAL_HEADER.BaseOfCode)}")
        print(f"  Image Base: {hex(pe.OPTIONAL_HEADER.ImageBase)}") # PE 파일이 메모리에 로드되는 시작 주소
        print(f"  Section Alignment: {pe.OPTIONAL_HEADER.SectionAlignment}") # 메모리에서 섹션의 최소 단위/ 패딩크기 유추 가능
        print(f"  File Alignment: {pe.OPTIONAL_HEADER.FileAlignment}") # 파일에서 섹션의 최소 단위
        print(f"  Size of Image: {pe.OPTIONAL_HEADER.SizeOfImage}") # PE파일이 메모리에 로딩될 때 전체 크기, File alignment의 배수
        # Size of header 추가
        print(f"  SizeOfHeaders: {pe.OPTIONAL_HEADER.SizeOfHeaders}")
        # Subsystem 추가
        print(f"  Subsystem: {pe.OPTIONAL_HEADER.Subsystem}")

    except pefile.PEFormatError as e:
        print(f"Error: {e}")

