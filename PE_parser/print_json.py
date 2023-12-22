import pefile
import math
import datetime
import os
import json
import collections
from string_printer import *
from hash_calculator import *
import sys
from ppdeep import hash_from_file

def json_entropy(file_path):
    # 파일 읽기
    with open(file_path, 'rb') as file:
        data = file.read()

    # 각 바이트 값의 발생 빈도를 계산
    byte_counters = collections.Counter(data)
    file_length = len(data)

    # 샤논 엔트로피 계산
    entropy = 0
    for count in byte_counters.values():
        # 각 바이트의 확률
        probability = count / file_length
        # 엔트로피 계산
        entropy -= probability * math.log2(probability)

    return entropy

def json_pe_header(file_path):
    try:
        pe = pefile.PE(file_path)
        
        def format_timestamp(timestamp):
            timestamp_datetime = datetime.datetime.utcfromtimestamp(timestamp)
            return timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        header_info = {
            "Dos_Header": {
                "e_magic": hex(pe.DOS_HEADER.e_magic),
                "e_lfanew": hex(pe.DOS_HEADER.e_lfanew)
            },
            "NT Header": {
                "Signature": hex(pe.NT_HEADERS.Signature)
            },
            "File Header": {
                "Machine": hex(pe.FILE_HEADER.Machine), 
                "Number of Sections": pe.FILE_HEADER.NumberOfSections,
                "Time Date Stamp": format_timestamp(pe.FILE_HEADER.TimeDateStamp),
                "Characteristics": hex(pe.FILE_HEADER.Characteristics)
            },
            "Optional Header": {
                "Magic": hex(pe.OPTIONAL_HEADER.Magic),
                "SizeOfCode": hex(pe.OPTIONAL_HEADER.SizeOfCode),
                "Address Entry point": hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint),
                "BaseOfCode": hex(pe.OPTIONAL_HEADER.BaseOfCode),
                "Image Base": hex(pe.OPTIONAL_HEADER.ImageBase), 
                "Section Alignment": pe.OPTIONAL_HEADER.SectionAlignment, 
                "File Alignment": pe.OPTIONAL_HEADER.FileAlignment, 
                "Size of Image": pe.OPTIONAL_HEADER.SizeOfImage, 
                "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
                "Subsystem": pe.OPTIONAL_HEADER.Subsystem
                }
            }  
        return header_info

    except pefile.PEFormatError as e:
        print(f"Error: {e}")
        
def json_pe_section(file_path):
    pe = pefile.PE(file_path)
    sections_info = []

    sections = pe.sections

    for section in sections:
        section_info = {
            "Name": section.Name.decode().rstrip('\x00'),
            "Virtual_Size": section.Misc_VirtualSize,
            "Virtual_Address": hex(section.VirtualAddress),
            "Raw_Size": section.SizeOfRawData,
            "Raw_Address": hex(section.PointerToRawData),
            "Characteristics": hex(section.Characteristics)
        }

        sections_info.append(section_info)

    return sections_info

def json_imports_and_exports(file_path):
    pe = pefile.PE(file_path)
    imports_info = []
    exports_info = []

    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for file in pe.DIRECTORY_ENTRY_IMPORT:
            dll_info = {
                "DLL": file.dll.decode(),
                "Functions": [
                    {"Function": function.name.decode() if function.name else f"Ordinal {function.ordinal}"}
                    for function in file.imports
                ]
            }
            imports_info.append(dll_info)

    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        exports_info = [
            {"Function": function.name.decode() if function.name else f"Ordinal {function.ordinal}"}
            for function in pe.DIRECTORY_ENTRY_EXPORT.symbols
        ]

    return {"Imported_Functions": imports_info, "Exported_Functions": exports_info}

def json_resource(file_path):
    pe = pefile.PE(file_path)
    resource_info = []

    if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
            resource_type_str = pefile.RESOURCE_TYPE.get(resource_type.struct.Id, resource_type.struct.Id)
            resource_type_entry = {
                "Resource_Type": resource_type_str,
                "Resources": []
            }

            for resource_entry in resource_type.directory.entries:
                if hasattr(resource_entry, 'data') and hasattr(resource_entry.data, 'struct'):
                    resource_entry_info = {
                        "Resource_ID": resource_entry.id,
                        "Size": resource_entry.data.struct.Size,
                        "RVA": hex(resource_entry.data.struct.OffsetToData),
                        "Offset": hex(pe.get_offset_from_rva(resource_entry.data.struct.OffsetToData))
                    }
                    resource_type_entry["Resources"].append(resource_entry_info)

            resource_info.append(resource_type_entry)

    return resource_info

#계속 오류 발생
def json_relocation(file_path):
    pe = pefile.PE(file_path)
    relocation_info = []

    if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'):
        for relocation in pe.DIRECTORY_ENTRY_BASERELOC.entries:
            relocation_entry_info = {
                "Virtual_Address": hex(relocation.directory.VirtualAddress),
                "Size_of_Blocks": relocation.directory.Size,
                "Entries": []
            }

            type_value = relocation.type
            offset_value = relocation.rva

            entry_info = {
                "Type": type_value,
                "Offset": hex(offset_value)
            }

            relocation_entry_info["Entries"].append(entry_info)

            relocation_info.append(relocation_entry_info)

    return relocation_info


# 엔트리포인트 이전에 시작되는 TLS콜백 함수를 확인하여 디버깅 여부를 확인한다. / 보통 출력되는 값이 없어야 한다.
def json_tls(file_path):
    tls_info = {}

    try:
        pe = pefile.PE(file_path)

        if hasattr(pe, "DIRECTORY_ENTRY_TLS"):
            tls = pe.DIRECTORY_ENTRY_TLS.struct

            tls_info = {
                "Start_Address_of_Raw_Data": hex(tls.StartAddressOfRawData),
                "End_Address_of_Raw_Data": hex(tls.EndAddressOfRawData),
                "Characteristics": hex(tls.Characteristics)
            }
            if tls.AddressOfCallBacks:
                for callback_address in tls.AddressOfCallBacks:
                    call_back  = {"Callback Address": callback_address}
                    tls_info.append(call_back)

    except pefile.PEFormatError as e:
        print(f"Error: {e}")

    return tls_info

def json_hash(file_path):
    hash_info = {}

    hash = calc_hashes(file_path)
    imphash = calc_imphash(file_path)

    if hash and imphash:
        md5_hash, sha1_hash, sha256_hash = hash
        hash_info = {
            "MD5_Hash": md5_hash,
            "SHA1_Hash": sha1_hash,
            "SHA256_Hash": sha256_hash,
            "ImpHash": imphash
        }

    return hash_info

def json_ssdeep(file_path):
    pe = pefile.PE(file_path)
    try:
        fuzzy_hash = hash_from_file(file_path)
        ssdeep_info = {
            "Fuzzy hash": fuzzy_hash
        }
        return ssdeep_info
    except Exception as e:
        print(f"Error calculating fuzzy hash")
    
def json_certificate(file_path):
    pe = pefile.PE(file_path)
    try:
        if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
            cert_entry = pe.DIRECTORY_ENTRY_SECURITY
            certy = {
                "Digital Signature Information" : {},
                "Signer" : cert_entry.signer,
                "Issuer" : cert_entry.issuer,
                "Serial Number" : cert_entry.serial_number,
                "Timestamp" : datetime.datetime.utcfromtimestamp(cert_entry.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        }
        return certy
    except Exception as e:
        print(f"Error: {e}")

def all_pe_info(file_path):
    try:
        pe_info = {
            "Entropy": json_entropy(file_path) or 0.0,
            "PE_Header_Info": json_pe_header(file_path) or {},
            "Sections": json_pe_section(file_path) or [],
            "Imports_and_Exports": json_imports_and_exports(file_path) or {},
            "Resource": json_resource(file_path) or {},
            "Relocation": json_relocation(file_path) or {},
            "TLS": json_tls(file_path) or {},
            "Hash": json_hash(file_path) or {},
            "ssdeep": json_ssdeep(file_path) or {},
            "certification": json_certificate(file_path) or {},
            # "contains_another_exe": find_exe_files(file_path) or {},
            # "is_upx_packed": is_upx_packed(file_path) or {},
            # "file_type": check_if_dll_or_exe(file_path) or {}
        }
        return pe_info
    except Exception as e:
        print(f"Error in all_pe_info: {e}")
        return {}
       
def save_json_file(json_data, json_file_path):
    def convert_to_serializable(obj):
        try:
            # Try to serialize the object
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If a circular reference error occurs, convert to string representation
            return repr(obj)

    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, default=convert_to_serializable, indent=4)
        
def save_json_result(file_path):
    json_output_path = input("Enter the path to save the JSON files: ")
    
    try:
        pe_info = all_pe_info(file_path)
        ascii_strings, unicode_strings = print_strings(file_path)

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        json_file_name = f"{file_name}_pe_info.json"
        ascii_file_name = f"{file_name}_ascii_strings.json"
        unicode_file_name = f"{file_name}_unicode_strings.json"
        
        json_file_path = os.path.join(json_output_path, json_file_name)
        ascii_file_path = os.path.join(json_output_path, ascii_file_name)
        unicode_file_path = os.path.join(json_output_path, unicode_file_name)
        
        save_json_file(pe_info, json_file_path)
        save_json_file(unicode_strings, unicode_file_path)
        save_json_file(ascii_strings, ascii_file_path)
        
        print("\nJSON files created successfully.{json_file_path}")

    except pefile.PEFormatError as e:
        print(f"Save Json Error: {e}")
