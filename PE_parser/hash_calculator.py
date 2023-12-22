import hashlib
import pefile

def calc_hashes(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = file.read()

            md5_hash = hashlib.md5(data).hexdigest()
            sha1_hash = hashlib.sha1(data).hexdigest()
            sha256_hash = hashlib.sha256(data).hexdigest()
            
        return md5_hash, sha1_hash, sha256_hash
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None

#imphash 값 계산
def calc_imphash(file_path):
    try:
        pe = pefile.PE(file_path)
        imphash = pe.get_imphash()
        print(imphash)
        return imphash
    except pefile.PEFormatError as e:
        print(f"Error: {e}")
        return None
            
#hash 값 출력
def print_hash(file_path):
    
    hash = calc_hashes(file_path)
    imphash = calc_imphash(file_path)
    
    if hash or imphash:
        md5_hash, sha1_hash, sha256_hash = hash
        print(f"Hashes: ")
        print(f"MD5 Hash: {md5_hash}")
        print(f"SHA-1 Hash: {sha1_hash}")
        print(f"SHA-256 Hash: {sha256_hash}")
        print(f"ImpHash: {imphash}\n")
    else:
        print("Failed to calculate hashes or imphash.")
