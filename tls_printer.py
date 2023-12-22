import pefile

def print_tls(file_path):
    try:

        pe = pefile.PE(file_path)

        if hasattr(pe, "DIRECTORY_ENTRY_TLS"):
            tls = pe.DIRECTORY_ENTRY_TLS.struct

            print("TLS Information")
            print(f" Start Address of Raw Data: {hex(tls.StartAddressOfRawData)}")
            print(f" End Address of Raw Data: {hex(tls.EndAddressOfRawData)}")
            print(f" Address of Callbacks: {hex(tls.AddressOfCallBacks)}")
            print(f" Characteristics: {hex(tls.Characteristics)}")

            if tls.AddressOfCallBacks:
                print("\nTLS Callbacks:")
                for callback_address in tls.AddressOfCallBacks:
                    print(hex(callback_address))

        else:
            print("No Tls information in PE file")

    except pefile.PEFormatError as e:
        print(f"Error: {e}")

