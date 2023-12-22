import pefile

def print_imports_and_exports(file_path):
    try:

        pe = pefile.PE(file_path)

        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            print('\nImported Functions:')
            for file in pe.DIRECTORY_ENTRY_IMPORT:
                print(f"\n DLL: {file.dll.decode()}")
                for function in file.imports:
                    # 함수 이름이 없을 경우 dll파일의 ordinal number로 대체
                    print(f"  Function: {function.name.decode() if function.name else 'Ordinal ' + str(function.ordinal)}")

        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            print("\nExported Functions:")
            for function in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                print(f"  Function: {function.name.decode() if function.name else 'Ordinal ' + str(function.ordinal)}")

    except pefile.PEFormatError as e:
        print(f"Error: {e}")

