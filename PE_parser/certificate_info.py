import pefile
import datetime

def print_certificate_info(pe):
    """
    Print information about the digital certificate in the PE file.

    :param pe: An instance of pefile.PE representing the PE file.
    """
    try:
        # 디지털 서명 정보가 있는지 확인
        if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
            # 디지털 서명 디렉토리 엔트리 가져오기
            cert_entry = pe.DIRECTORY_ENTRY_SECURITY

            # 디지털 서명 정보 출력
            print("Digital Signature Information:")
            print("Signer: ", cert_entry.signer)
            print("Issuer: ", cert_entry.issuer)
            print("Serial Number: ", cert_entry.serial_number)
            print("Timestamp: ", datetime.datetime.utcfromtimestamp(cert_entry.timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            print("")

        else:
            print("No digital signature information found.")

    except Exception as e:
        print(f"Error: {e}")

