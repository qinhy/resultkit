# from https://github.com/qinhy/singleton-key-value-storage.git
from datetime import datetime
import json
from typing import Callable, Optional, TypeVar, Type, overload
import unittest
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
try:
    from Storages import DictStorageController, DictStorage
except Exception as e:
    from .Storages import DictStorageController, DictStorage
try:
    from typing import ParamSpec  # Py 3.10+
except ImportError:               # Py <3.10 -> pip install typing_extensions
    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


# HEAVY_LEVEL: Light
# Reason: Returns the current UTC timestamp; constant-time system clock call.
def now_utc():
    return datetime.now(timezone.utc)

class BasicModel(BaseModel):
    # HEAVY_LEVEL: Light
    # Reason: Only raises NotImplementedError; no meaningful work.
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('This method should be implemented by subclasses.')
    
    # HEAVY_LEVEL: Light
    # Reason: Formats an error message string.
    def _log_error(self, e):
        return f"[{self.__class__.__name__}] Error: {str(e)}"
    
    # # HEAVY_LEVEL: Variable / Delegated
    # # Reason: Wrapper overhead is light, but it executes an arbitrary callback whose cost may be light, medium, or heavy.
    # def _try_error(self, func, default_value=('NULL',None)):
    #     try:
    #         return (True,func())
    #     except Exception as e:
    #         self._log_error(e)
    #         return (False,default_value)
        
    # # HEAVY_LEVEL: Variable / Delegated
    # # Reason: Delegates to _try_error and returns only the success flag.
    # def _try_binary_error(self, func):
    #     return self._try_error(func)[0]
    
    # # HEAVY_LEVEL: Variable / Delegated
    # # Reason: Delegates to _try_error and returns only the result object.
    # def _try_obj_error(self, func, default_value=('NULL',None)):
    #     return self._try_error(func,default_value)[1]
    
class Controller4Basic:
    class AbstractObjController:
        # HEAVY_LEVEL: Light
        # Reason: Stores references to the model and backing store.
        def __init__(self, store, model):
            self.model:Model4Basic.AbstractObj = model
            self._store:BasicStore = store
        
        # HEAVY_LEVEL: Light
        # Reason: Returns the stored backend reference.
        def storage(self):return self._store

        # HEAVY_LEVEL: Medium
        # Reason: Iterates over provided fields, updates timestamps, and persists the model to storage.
        def update(self, **kwargs):
            assert self.model is not None, 'controller has null model!'
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
            self._update_timestamp()
            self.store()
            return self

        # HEAVY_LEVEL: Light
        # Reason: Assigns one timestamp.
        def _update_timestamp(self):
            self.model.update_time = now_utc()
            
        # HEAVY_LEVEL: Medium
        # Reason: Serializes/dumps the model and writes it to storage; cost grows with model size.
        def store(self):
            assert self.model._id is not None
            self.storage().set(self.model._id,self.model.model_dump_json_dict())
            return self

        # HEAVY_LEVEL: Medium
        # Reason: Deletes one storage entry and clears the model controller reference; backend-dependent.
        def delete(self):
            self.storage().delete(self.model.get_id())
            self.model.controller = None

        # HEAVY_LEVEL: Medium
        # Reason: Copies metadata, updates the model, and persists it.
        def update_metadata(self, key, value):
            updated_metadata = {**self.model.metadata, key: value}
            self.update(metadata = updated_metadata)
            return self
        
    class AbstractGroupController(AbstractObjController):
        # HEAVY_LEVEL: Light
        # Reason: Stores references to the group model and backing store.
        def __init__(self, store, model):
            self.model: Model4Basic.AbstractGroup = model
            self._store: BasicStore = store

        # HEAVY_LEVEL: Medium
        # Reason: For non-root groups, loads the parent and filters the parent child list before deleting.
        def delete(self):
            if not self.model.is_root():
                parent: Model4Basic.AbstractGroup = self.storage().find(self.model.parent_id)
                remaining_ids = [cid for cid in parent.children_id if cid != self.model.get_id()]
                parent.controller.update(children_id=remaining_ids)
            return super().delete()
        
        # HEAVY_LEVEL: Heavy
        # Reason: Recursively walks and deletes every descendant in the subtree.
        def delete_recursive(self):
            for child, _ in self.model.yield_children_recursive():
                child.controller.delete()
            self.delete()
            
        # HEAVY_LEVEL: Medium
        # Reason: Finds a child, updates parent children_id, and updates the child depth/parent.
        def add_child(self, child_id: str):
            if hasattr(child_id,'get_id'):
                child_id = child_id.get_id()
            child: Model4Basic.AbstractGroup = self.storage().find(child_id)
            if child:
                self.update(children_id= self.model.children_id + [child_id])
                child.controller.update(depth=self.model.depth+1,
                                        parent_id=self.model.get_id())

        # HEAVY_LEVEL: Heavy
        # Reason: May recursively delete an entire child subtree before updating the parent.
        def delete_child(self, child_id:str):
            if child_id not in self.model.children_id:return self
            remaining_ids = [cid for cid in self.model.children_id if cid != child_id]
            child_con = self.storage().find(child_id).controller
            if hasattr(child_con, 'delete_recursive'):
                child_con:Controller4Basic.AbstractGroupController = child_con
                child_con.delete_recursive()
            else:
                child_con.delete()
            self.update(children_id = remaining_ids)
            return self

class Model4Basic:
    class AbstractObj(BasicModel):
        _id: str=None
        rank: list = [0]
        create_time: datetime = Field(default_factory=now_utc)
        update_time: datetime = Field(default_factory=now_utc)
        status: str = ""
        metadata: dict = {}
        auto_del: bool = False # auto delete when removed from memory 
          
        # auto exclude when model dump
        model_config = ConfigDict(arbitrary_types_allowed=True)
        controller: Optional[Controller4Basic.AbstractObjController] = None

        # HEAVY_LEVEL: Medium / Delegated
        # Reason: Delegates deletion to the controller; backend and relationship cleanup determine cost.
        def __obj_del__(self):
            # print(f'BasicApp.store().delete({self.id})')
            self.controller.delete()
        
        # HEAVY_LEVEL: Conditional / Medium
        # Reason: Normally just checks auto_del; if auto_del is true, delegates deletion.
        def __del__(self):
            if hasattr(self,'auto_del') and self.auto_del: self.__obj_del__()
        
        # HEAVY_LEVEL: Light
        # Reason: Only raises NotImplementedError in this abstract base.
        def model_dump_json_dict(self,exclude=None):
            raise NotImplementedError('This method should be implemented by subclasses.')
            # return json.loads(self.model_dump_json(exclude=exclude))
        
        # HEAVY_LEVEL: Light
        # Reason: No-op hook.
        def model_post_store_add(self):
            pass

        # HEAVY_LEVEL: Light
        # Reason: Returns the class name string.
        def class_name(self): return self.__class__.__name__

        # HEAVY_LEVEL: Medium
        # Reason: Pydantic copy cost grows with model size; deep=True can be heavier.
        def model_copy(self, *, update = None, deep = False):
            res = super().model_copy(update=update, deep=deep)
            res.set_id(None,ast=False)
            return res

        # HEAVY_LEVEL: Light
        # Reason: Performs an assertion and assigns the id.
        def set_id(self,id:str,ast=True):
            if ast:
                assert self._id is None, 'this obj is been setted! can not set again!'       
            setattr(self,'_id',id)
            self.__dict__['_id'] = id
            return self
        
        # HEAVY_LEVEL: Light
        # Reason: Builds a string and generates one UUID.
        def gen_new_id(self): 
            return f"{self.class_name()}:{uuid4()}"

        # HEAVY_LEVEL: Light
        # Reason: Asserts id exists and returns it.
        def get_id(self):
            assert self._id is not None, 'this obj is not setted!'
            return self._id

        # HEAVY_LEVEL: Medium
        # Reason: Serializes the Pydantic model; cost grows with fields and nested data.
        def model_dump_json(self, *, indent = None, include = None, exclude = None, context = None, by_alias = False, exclude_unset = False, exclude_defaults = False, exclude_none = False, round_trip = False, warnings = True, serialize_as_any = False):
            if exclude:
                exclude += ['controller']
            else:
                exclude = ['controller']
            return super().model_dump_json(indent=indent, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, serialize_as_any=serialize_as_any)

        # HEAVY_LEVEL: Medium
        # Reason: Dumps the Pydantic model to Python objects; cost grows with fields and nested data.
        def model_dump(self, *, mode = 'python', include = None, exclude = None, context = None, by_alias = False, exclude_unset = False, exclude_defaults = False, exclude_none = False, round_trip = False, warnings = True, serialize_as_any = False):
            if exclude:
                exclude += ['controller']
            else:
                exclude = ['controller']
            return super().model_dump(mode=mode, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, serialize_as_any=serialize_as_any)
       
        # HEAVY_LEVEL: Medium
        # Reason: Reflects over controller class attributes and builds a lookup dictionary.
        def _get_controller_class(self,modelclass=Controller4Basic):
            class_type = self.class_name()+'Controller'
            res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
            res = res.get(class_type, None)
            if res is None: 
                print(f'[warning]: No such class of {class_type}, use Controller4Basic.AbstractObjController')
                res = Controller4Basic.AbstractObjController
            return res
        
        # HEAVY_LEVEL: Light / Medium
        # Reason: Instantiates a controller; usually light, but depends on chosen controller constructor.
        def init_controller(self,store):
            self.controller = self._get_controller_class()(store,self)

    class AbstractGroup(AbstractObj):
        owner_id: str=''
        parent_id: str = ''
        children_id: list[str] = []
        depth: int = -1
        # auto exclude when model dump
        controller: Optional[Controller4Basic.AbstractGroupController] = None

        # HEAVY_LEVEL: Medium / Delegated
        # Reason: Performs a storage lookup for the owner object.
        def get_own(self):
            return self.controller.storage().find(self.owner_id)
        
        # HEAVY_LEVEL: Medium / Delegated
        # Reason: Performs a storage lookup for the parent object.
        def get_parent(self):
            return self.controller.storage().find(self.parent_id)
        
        # HEAVY_LEVEL: Light
        # Reason: Single integer comparison.
        def is_root(self) -> bool:
            return self.depth == 0

        # HEAVY_LEVEL: Medium
        # Reason: Iterates direct child ids and performs storage checks/lookups for each child.
        def foreach_child(self):
            for child_id in self.children_id:
                if not self.controller.storage().exists(child_id): continue
                child: Model4Basic.AbstractGroup = self.controller.storage().find(child_id)
                yield child, hasattr(child, 'children_id')

        # HEAVY_LEVEL: Heavy
        # Reason: Recursively traverses all descendants in the group tree.
        def yield_children_recursive(self, depth: int = 0):
            for child,has_children in self.foreach_child():
                if has_children:
                    yield from child.yield_children_recursive(depth + 1)
                yield child, depth

        # HEAVY_LEVEL: Heavy
        # Reason: Recursively traverses and materializes all descendants into nested lists.
        def get_children_recursive(self):
            children_list = []
            for child,has_children in self.foreach_child():
                if has_children:
                    children_list.append(child.get_children_recursive())
                else:
                    children_list.append(child)
            return children_list

        # HEAVY_LEVEL: Medium
        # Reason: Performs one storage lookup per direct child.
        def get_children(self):
            return [self.controller.storage().find(child_id) for child_id in self.children_id]

        # HEAVY_LEVEL: Medium
        # Reason: Checks membership in children_id and may perform one storage lookup.
        def get_child(self, child_id: str):
            if child_id in self.children_id:
                return self.controller.storage().find(child_id)
     
class BasicStore(DictStorageController):
    MODEL_CLASS_GROUP = Model4Basic
    
    # HEAVY_LEVEL: Light / Unknown dependency
    # Reason: Creates a singleton-backed DictStorage controller; actual backend construction cost is external.
    @staticmethod
    def build(): return BasicStore(DictStorage().get_singleton())

    # HEAVY_LEVEL: Medium
    # Reason: Splits the id and scans model class attributes to build a class lookup.
    def _get_class(self, id: str, modelclass=MODEL_CLASS_GROUP):
        class_type = id.split(':')[0]
        res = [i for k,i in modelclass.__dict__.items() if '_' not in k]
        res = {c.__name__:c for c in res}
        res = res.get(class_type, None)
        if res is None: raise ValueError(f'No such class of {class_type}')
        return res
    
    # HEAVY_LEVEL: Light
    # Reason: Compares id prefix to object class name and may build a new id string.
    def _auto_fix_id(self,obj:MODEL_CLASS_GROUP.AbstractObj, id:str="None"):
        class_type = id.split(':')[0]
        obj_class_type = obj.class_name()
        if class_type != obj_class_type: id = f'{obj_class_type}:{id}'
        return id
    
    # HEAVY_LEVEL: Medium
    # Reason: Constructs or accepts a model object, sets id, and initializes its controller.
    def _get_as_obj(self,id,data_dict)->MODEL_CLASS_GROUP.AbstractObj:
        if data_dict is None : return None
        if isinstance(data_dict,str):
            obj:Model4Basic.AbstractObj = self._get_class(id)(**data_dict)
        else:
            obj:Model4Basic.AbstractObj = data_dict
        obj.set_id(id,ast=False).init_controller(self)
        return obj
    
    # HEAVY_LEVEL: Medium
    # Reason: Generates/determines id, dumps object data, stores it, reconstructs controller-bound object, then runs a hook.
    def _add_new_obj(self, obj:MODEL_CLASS_GROUP.AbstractObj, id:str=None):
        id,d = (obj.gen_new_id() if id is None else id), obj.model_dump_json_dict()
        self.set(  self._auto_fix_id(obj,id)  ,d)
        obj = self._get_as_obj(id,d)        
        obj.model_post_store_add()
        return obj
    
    # HEAVY_LEVEL: Light
    # Reason: Checks and registers a model class attribute.
    def add_new_class(self,obj_class_type:Type[MODEL_CLASS_GROUP.AbstractObj]):
        if not hasattr(self.MODEL_CLASS_GROUP,obj_class_type.__name__):
            setattr(self.MODEL_CLASS_GROUP,obj_class_type.__name__,obj_class_type)
            
    # HEAVY_LEVEL: Light
    # Reason: Registers a class and returns a closure; the returned closure does the heavier object creation.
    def add_new(self, obj_class_type:Type[T],id:str=None):
        self.add_new_class(obj_class_type)
        # HEAVY_LEVEL: Medium
        # Reason: Constructs a model instance and stores it through _add_new_obj.
        def add_obj(*args: P.args, **kwargs: P.kwargs)->T:
            obj:BasicStore.MODEL_CLASS_GROUP.AbstractObj = obj_class_type(*args,**kwargs)
            if obj._id is not None: raise ValueError(f'obj._id is "{obj._id}", must be none')
            return self._add_new_obj(obj,id)
        return add_obj
    
    # HEAVY_LEVEL: Medium
    # Reason: Registers class, validates id, and stores an existing object through _add_new_obj.
    def add_new_obj(self, obj:T, id:str=None)->T:
        self.add_new_class(obj.__class__)
        if obj._id is not None: raise ValueError(f'obj._id is {obj._id}, must be none')
        return self._add_new_obj(obj,id)
    
    # HEAVY_LEVEL: Medium / Heavy on fallback
    # Reason: Direct lookup is medium; if missing and fa=True, falls back to find_all and scans matching keys.
    def find(self,id:str, fa:bool=True) -> MODEL_CLASS_GROUP.AbstractObj:
        if self.exists(id): return self._get_as_obj(id, self.get(id) )
        res = self.find_all(f'*:{id}') if fa else []
        return res[0] if len(res) == 1 else None
    
    # HEAVY_LEVEL: Heavy
    # Reason: Scans matching keys and calls find for each result, materializing all matching objects.
    def find_all(self,id:str=f'AbstractObj:*')->list[MODEL_CLASS_GROUP.AbstractObj]:
        return [self.find(k,False) for k in self.keys(id)]
