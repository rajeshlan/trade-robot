import os
import h5py

# Automatically detect the project's root directory
PROJECT_ROOT = r"D:\RAJESH FOLDER\PROJECTS\trade-robot"  # Update this path if needed

def find_h5_files(root_dir):
    """Recursively scans the project root for all .h5 files."""
    h5_files = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(folder, file))
    return h5_files

def inspect_h5_model(file_path):
    """Opens and inspects an .h5 file to check its contents."""
    try:
        with h5py.File(file_path, "r") as f:
            print(f"\n🔍 Inspecting: {file_path}")
            print("📂 Keys in HDF5 file:", list(f.keys()))

            if "model_weights" in f.keys():
                print("✅ Model contains weights.")
            if "optimizer_weights" in f.keys():
                print("✅ Model contains optimizer weights.")
            if "training_config" in f.keys():
                print("✅ Model has training configuration.")

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    h5_files = find_h5_files(PROJECT_ROOT)
    
    if not h5_files:
        print("⚠️ No .h5 files found in the project directory.")
    else:
        print("\n📁 Found .h5 files:")
        for i, file in enumerate(h5_files):
            print(f"{i+1}. {file}")

        # Ask user to inspect a specific file
        choice = input("\nEnter the number of the file you want to inspect: ")
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(h5_files):
                inspect_h5_model(h5_files[choice])
            else:
                print("❌ Invalid choice. Please enter a valid number.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
