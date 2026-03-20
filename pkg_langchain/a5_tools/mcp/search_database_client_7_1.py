import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("7-1_search_database_client.py")
    runpy.run_path(str(target), run_name="__main__")
