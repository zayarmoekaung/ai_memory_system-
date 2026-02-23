import sys
from pathlib import Path

def setup_project_path():
    """Add the project root to Python's module search path"""
    root_dir = Path(__file__).parent.parent  
    
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

setup_project_path()