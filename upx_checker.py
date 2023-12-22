def is_upx_packed(pe):
    """
    Check if the PE file is packed with UPX.

    :param pe: An instance of pefile.PE representing the PE file.
    :return: True if UPX packed, False otherwise.
    """
    try:
        # UPX 패킹 여부 확인
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if resource_type.name is not None and b"UPX" in resource_type.name.string:
                    return True
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False

