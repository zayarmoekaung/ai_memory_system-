from collections import deque
from typing import List, Dict, Any
import time

class WorkingMemory:
    def __init__(self, capacity: int = 10):
        """
        Initializes the WorkingMemory (Sensory Input Buffer) with a fixed capacity.
        Stores recent interactions and active thoughts for immediate recall.

        Args:
            capacity (int): The maximum number of items (e.g., messages, thoughts) to store.
        """
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add_item(self, item: Dict[str, Any]):
        """
        Adds a new item to the working memory buffer.
        Each item should at least have 'content' and 'timestamp'.
        """
        item['timestamp'] = time.time()
        self.buffer.append(item)

    def get_recent_items(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent items from working memory.

        Args:
            n (int, optional): The number of recent items to retrieve. If None, retrieves all.

        Returns:
            List[Dict[str, Any]]: A list of recent memory items, ordered from oldest to newest.
        """
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def clear(self):
        """
        Clears all items from the working memory.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"WorkingMemory(capacity={self.buffer.maxlen}, size={len(self.buffer)})
Items: {list(self.buffer)}"
