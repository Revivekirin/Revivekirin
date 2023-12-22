import pefile

def print_rich_header(file_path):

    pe = pefile.PE(file_path)

    rich_header = pe.parse_rich_header()
    if rich_header is not None:
        print("Rich Header Information: ")
        print(f"Key: {rich_header.keys}")
        if 'records' in rich_header:
            records = rich_header['records']
            print("Records:")
    
            for record in records:
                # Access individual elements in the record
                print(f"ID: {record.get('id', 'N/A')}, Version: {record.get('version', 'N/A')}, Count: {record.get('count', 'N/A')}")
    else:
        print("No 'records' key found in rich_header.")
