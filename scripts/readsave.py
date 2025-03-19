import os
import json
import pickle
import gzip
import bz2
import sqlite3
import pandas as pd
import h5py
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def detect_file_type(filepath):
    """Detect if the file is text or binary"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)  # Read a small portion to check encoding
        return 'text'
    except UnicodeDecodeError:
        return 'binary'

def read_sqlite_file(filepath):
    """Read SQLite database file"""
    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("SQLite Tables:", [t[0] for t in tables])
        conn.close()
    except Exception as e:
        print("Error reading SQLite file:", e)

def read_hdf5_file(filepath):
    """Read HDF5 file"""
    try:
        with h5py.File(filepath, 'r') as f:
            print("HDF5 datasets:", list(f.keys()))
    except Exception as e:
        print("Error reading HDF5 file:", e)

def inspect_scaler(scaler):
    """Display scaler details if applicable"""
    print("\nScaler Type:", type(scaler))
    if hasattr(scaler, 'mean_'):
        print("Scaler mean:", scaler.mean_)
    if hasattr(scaler, 'scale_'):
        print("Scaler scale:", scaler.scale_)
    if hasattr(scaler, 'var_'):
        print("Scaler variance:", scaler.var_)
    if hasattr(scaler, 'data_min_'):
        print("Scaler data min:", scaler.data_min_)
    if hasattr(scaler, 'data_max_'):
        print("Scaler data max:", scaler.data_max_)
    if hasattr(scaler, 'feature_names_in_'):
        print("Scaler feature names:", scaler.feature_names_in_)

def read_save_file(filepath):
    """Attempt to read the content of a .save file"""
    if not os.path.exists(filepath):
        print("File not found.")
        return
    
    # Check for SQLite database
    if filepath.endswith(".db") or filepath.endswith(".sqlite"):
        read_sqlite_file(filepath)
        return
    
    # Check for HDF5 file
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        read_hdf5_file(filepath)
        return
    
    file_type = detect_file_type(filepath)
    print(f"Detected file type: {file_type}\n")
    
    if file_type == 'text':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print("First 500 characters:\n", content[:500])
        except Exception as e:
            print("Error reading file:", e)
    else:
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Try unpickling
            try:
                data = pickle.loads(content)
                print("Pickle content preview:", data)
                if isinstance(data, (list, np.ndarray)):
                    print("Feature names:", data)
                elif isinstance(data, (StandardScaler, MinMaxScaler)):
                    inspect_scaler(data)
                return
            except pickle.UnpicklingError:
                pass
            
            # Try loading with joblib
            try:
                data = joblib.load(filepath)
                print("Joblib content preview:", data)
                if isinstance(data, (StandardScaler, MinMaxScaler)):
                    inspect_scaler(data)
                return
            except Exception:
                pass
            
            # Try decompressing
            try:
                data = gzip.decompress(content)
                print("Gzip decompressed data preview:\n", data[:500])
                return
            except OSError:
                pass
            
            try:
                data = bz2.decompress(content)
                print("Bz2 decompressed data preview:\n", data[:500])
                return
            except OSError:
                pass
            
            print("Binary file detected. First 500 bytes:", content[:500])
        except Exception as e:
            print("Error reading binary file:", e)

if __name__ == "__main__":
    filepath = input("Enter the .save file path: ").strip()
    read_save_file(filepath)
