try:
    from database.vector_store import SearchResult
    print("SearchResult import successful")
except ImportError as e:
    print(f"SearchResult import failed: {e}")
