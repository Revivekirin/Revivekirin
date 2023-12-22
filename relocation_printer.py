import pefile

def print_relocation(file_path):
    try:
        pe = pefile.PE(file_path)

        if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'):
            print("Relocation Information:")

            for relocation in pe.DIRECTORY_ENTRY_BASERELOC:   # 각 block을 relocation에 할당
                # print(f"\nVirtual Address: ", hex(virtual_address))
                # print(f" Size of Blocks: {relocation_block}")

                for entry in relocation.entries: #각 block의 entries(type, offset)출력
                    type = entry.type 
                    offset = entry.rva

                    print(f" Type: {type}")
                    print(f" Offset: {hex(offset)}")

        else:
            print("There are no relocation information.")

    except pefile.PEFormatError as e:
        print("Error: {e}")

