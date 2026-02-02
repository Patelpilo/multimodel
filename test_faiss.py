try:
    import faiss
    print("faiss import successful")
except ImportError as e:
    print(f"Import failed: {e}")
