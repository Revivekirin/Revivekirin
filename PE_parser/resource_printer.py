import pefile

def print_resource(file_path):
    try:
        pe = pefile.PE(file_path)

        if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            print("\nResource Information:")
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                resource_type_str = pefile.RESOURCE_TYPE.get(resource_type.struct.Id, resource_type.struct.Id)  # ex) RT_ICON, RT_STRING, RT_BITMAP
                print(f" Resource Type: {resource_type_str}")
                for resource_entry in resource_type.directory.entries:
                    if hasattr(resource_entry, 'data'):
                        if hasattr(resource_entry.data, 'struct'):
                            print(f" Resource ID: {resource_entry.id}")
                            print(f" Size: {resource_entry.data.struct.Size}")
                            print(f" RVA: {hex(resource_entry.data.struct.OffsetToData)}")
                            offset = pe.get_offset_from_rva(resource_entry.data.struct.OffsetToData)
                            print(f" Offset: {hex(offset)}")
    except pefile.PEFormatError as e:
        print(f"Error: {e}")
