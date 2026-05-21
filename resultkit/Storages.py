# Annotated with HEAVY_LEVEL comments.
# Levels used: Light, Medium, Heavy, Critical, or conditional combinations.
# These comments are performance estimates based on the visible code paths and backend/callback behavior.

# from https://github.com/qinhy/singleton-key-value-storage.git
import base64
import sys
import uuid
import fnmatch
import json
from pathlib import Path
from collections import OrderedDict

# HEAVY_LEVEL: Light
# Reason: Base64-url encodes one string; work grows with input length but is usually cheap.
# Complexity: O(n), n = length of s.
def b64url_encode(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")

# HEAVY_LEVEL: Light
# Reason: Adds padding and base64-url decodes one string.
# Complexity: O(n), n = length of s.
def b64url_decode(s: str) -> str:
    # add back missing padding
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")

# HEAVY_LEVEL: Light
# Reason: Performs one decode and one encode to validate the string.
# Complexity: O(n), n = length of s.
def is_b64url(s: str) -> bool:
    try:
        return b64url_encode(b64url_decode(s)) == s
    except Exception:
        return False
        
# HEAVY_LEVEL: Heavy
# Reason: Recursively walks the reachable object graph, including dicts, containers, __dict__, and __slots__.
# Complexity: O(total reachable objects/items); expensive on large or deeply nested data.
def get_deep_bytes_size(obj, seen=None):
    obj_id = id(obj)
    if seen is None:
        seen = set()
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += get_deep_bytes_size(k, seen) + get_deep_bytes_size(v, seen)
        return size
    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_bytes_size(i, seen) for i in obj)
        return size
    if hasattr(obj, "__dict__"):
        size += get_deep_bytes_size(vars(obj), seen)
    if hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            try:
                size += get_deep_bytes_size(getattr(obj, slot), seen)
            except AttributeError:
                pass
    return size

# HEAVY_LEVEL: Light
# Reason: Fixed-size loop over a small list of units.
# Complexity: O(1).
def humanize_bytes(n):
    size = float(n)
    for unit in ['B','KB','MB','GB','TB']:
        if size < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

class AbstractStorage:
    _uuid = uuid.uuid4()
    _store = None
    _is_singleton = True
    _meta = {}
        
    # HEAVY_LEVEL: Light
    # Reason: Assigns a few attributes and may generate one UUID.
    # Complexity: O(1).
    def __init__(self,id=None,store=None,is_singleton=None):
        self.uuid = uuid.uuid4() if id is None else id
        self.store = None if store is None else store
        self.is_singleton = False if is_singleton is None else is_singleton
    
    # HEAVY_LEVEL: Light
    # Reason: Constructs one instance using class-level singleton fields.
    # Complexity: O(1), assuming subclass initialization stays lightweight.
    def get_singleton(self):
        return self.__class__(self._uuid,self._store,self._is_singleton)    
    
    # HEAVY_LEVEL: Light
    # Reason: Abstract placeholder that only raises NotImplementedError.
    # Complexity: O(1).
    def bytes_used(self, deep=True, human_readable=True):
        raise NotImplementedError("Subclasses must implement memory_usage method")
    
class DictStorage(AbstractStorage):
    _uuid = uuid.uuid4()
    _store = OrderedDict()
        
    # HEAVY_LEVEL: Light
    # Reason: Calls parent initializer and creates or assigns an OrderedDict.
    # Complexity: O(1) for an empty OrderedDict.
    def __init__(self,id=None,store=None,is_singleton=None):
        super().__init__(id,store,is_singleton)
        self.store = OrderedDict() if store is None else store

    # HEAVY_LEVEL: Heavy when deep=True; Light when deep=False
    # Reason: deep=True calls get_deep_bytes_size(self), which recursively scans reachable objects.
    # Complexity: O(total reachable objects/items) if deep=True; O(1) if deep=False.
    def bytes_used(self, deep=True, human_readable=True):
        size = get_deep_bytes_size(self) if deep else sys.getsizeof(self)
        return humanize_bytes(size) if human_readable else size
    
    @staticmethod
    # HEAVY_LEVEL: Light
    # Reason: Factory method creating an empty DictStorage and lightweight controller.
    # Complexity: O(1).
    def build_tmp(): return DictStorageController(DictStorage())

    @staticmethod
    # HEAVY_LEVEL: Light
    # Reason: Factory method creating a singleton-backed DictStorage controller.
    # Complexity: O(1).
    def build(): return DictStorageController(DictStorage().get_singleton())
    
class AbstractStorageController:
    # HEAVY_LEVEL: Light
    # Reason: Stores the model reference.
    # Complexity: O(1).
    def __init__(self, model): self.model:AbstractStorage = model
    # HEAVY_LEVEL: Light
    # Reason: Intended as a simple boolean check on model state.
    # Complexity: O(1).
    def is_singleton(self)->bool: return self.model.is_singleton if 'is_singleton' in self.model else False
    # HEAVY_LEVEL: Light
    # Reason: Placeholder method that only prints a message.
    # Complexity: O(1).
    def exists(self, key: str)->bool: print(f'[{self.__class__.__name__}]: not implement')
    # HEAVY_LEVEL: Light
    # Reason: Placeholder method that only prints a message.
    # Complexity: O(1).
    def set(self, key: str, value: dict): print(f'[{self.__class__.__name__}]: not implement')
    # HEAVY_LEVEL: Light
    # Reason: Placeholder method that only prints a message.
    # Complexity: O(1).
    def get(self, key: str)->dict: print(f'[{self.__class__.__name__}]: not implement')
    # HEAVY_LEVEL: Light
    # Reason: Placeholder method that only prints a message.
    # Complexity: O(1).
    def delete(self, key: str): print(f'[{self.__class__.__name__}]: not implement')
    # HEAVY_LEVEL: Light
    # Reason: Placeholder method that only prints a message.
    # Complexity: O(1).
    def keys(self, pattern: str='*')->list[str]: print(f'[{self.__class__.__name__}]: not implement')
    # HEAVY_LEVEL: Medium
    # Reason: Iterates all keys and deletes each one through the backend.
    # Complexity: O(k), k = number of keys; backend delete cost may add more.
    def clean(self): [self.delete(k) for k in self.keys('*')]
    
    # # HEAVY_LEVEL: Heavy
    # # Reason: Reads every key/value and serializes the full store to JSON.
    # # Complexity: O(total stored data size).
    # def dumps(self): return json.dumps({k:self.get(k) for k in self.keys('*')})
    # # HEAVY_LEVEL: Heavy
    # # Reason: Parses a JSON string and writes every item to the backend.
    # # Complexity: O(JSON size + number of items * backend set cost).
    # def loads(self, json_string=r'{}'): [ self.set(k,v) for k,v in json.loads(json_string).items()]
    # # HEAVY_LEVEL: Heavy
    # # Reason: Serializes the full store and writes it to disk.
    # # Complexity: O(total stored data size + file I/O).
    # def dump(self, path: str):return Path(path).write_text(self.dumps())
    # # HEAVY_LEVEL: Heavy
    # # Reason: Reads a file, parses JSON, and writes all entries to the backend.
    # # Complexity: O(file size + number of items * backend set cost).
    # def load(self, path: str):return self.loads(Path(path).read_text())


class DictStorageController(AbstractStorageController):
    # HEAVY_LEVEL: Light
    # Reason: Stores references to model and model.store.
    # Complexity: O(1).
    def __init__(self, model:DictStorage):
        self.model:DictStorage = model
        self.store = self.model.store
    # HEAVY_LEVEL: Light
    # Reason: OrderedDict membership check.
    # Complexity: Average O(1).
    def exists(self, key: str)->bool: return key in self.store
    # HEAVY_LEVEL: Light
    # Reason: Single OrderedDict assignment.
    # Complexity: Average O(1), excluding object size.
    def set(self, key: str, value: dict): self.store[key] = value
    # HEAVY_LEVEL: Light
    # Reason: Single OrderedDict lookup.
    # Complexity: Average O(1).
    def get(self, key: str)->dict: return self.store.get(key,None)
    # HEAVY_LEVEL: Light
    # Reason: Single OrderedDict pop.
    # Complexity: Average O(1).
    def delete(self, key: str): return self.store.pop(key)
    # HEAVY_LEVEL: Medium
    # Reason: fnmatch.filter scans all keys to match the pattern.
    # Complexity: O(k * p), k = number of keys, p = pattern/key match cost.
    def keys(self, pattern: str='*'): return fnmatch.filter(self.store.keys(), pattern)
