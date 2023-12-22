import os

def find_executables(directory):
    """
    Find executable files (with .exe extension) in the specified directory.

    :param directory: The directory to search for executable files.
    :return: A list of executable files found in the directory.
    """
    executables = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".exe"):
                    executable_path = os.path.join(root, file)
                    executables.append(executable_path)
        return executables

    except Exception as e:
        print(f"Error: {e}")
        return None

