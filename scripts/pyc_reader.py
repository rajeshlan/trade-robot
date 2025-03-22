import marshal
import dis
import struct
import sys

def read_pyc(file_path):
    try:
        with open(file_path, "rb") as f:
            magic_number = f.read(4)  # Read magic number
            timestamp = f.read(4)  # Read timestamp
            if sys.version_info >= (3, 7):
                f.read(4)  # Skip size field for Python 3.7+

            extracted_object = marshal.load(f)  # Attempt to load bytecode

        print(f"Magic Number: {magic_number.hex()} (indicates Python version)")
        print(f"Timestamp: {struct.unpack('I', timestamp)[0]}")
        print(f"Extracted Object Type: {type(extracted_object)}")

        if isinstance(extracted_object, type(compile('', '', 'exec'))):
            print("\nDisassembled Bytecode:")
            dis.dis(extracted_object)
        else:
            print("Error: Extracted object is not a code object. Here’s its value:")
            print(extracted_object)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_path = input("Enter the path to the .pyc file: ")
    read_pyc(file_path)
