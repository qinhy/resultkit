from __future__ import annotations

from typing import Annotated, Any, Dict, Iterable, Iterator, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .arrays import to_jsonable
from .geometry import BoundingBox, Keypoints, Mask, Polygon, Vector
from .text import TextSpan

ResultPayload = Annotated[
    Union[BoundingBox, Polygon, Keypoints, Mask, Vector, TextSpan],
    Field(discriminator="type"),
]


class ResultNode(BaseModel):
    """A single AI result node.

    A node can represent one detection, segmentation, OCR span, embedding,
    classification, tool output, or any higher-level result. Children make it
    possible to model hierarchical outputs such as object -> face -> landmarks.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str = "generic"
    label: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    class_id: Optional[int] = None
    payload: Optional[ResultPayload] = None
    value: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["ResultNode"] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_serializer("value")
    def serialize_value(self, value: Any) -> Any:
        return to_jsonable(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)

    def add_child(self, child: "ResultNode") -> "ResultNode":
        """Append and return a child node."""
        self.children.append(child)
        return child

    def walk(self, *, include_self: bool = True) -> Iterator["ResultNode"]:
        """Depth-first traversal over this node and its descendants."""
        if include_self:
            yield self
        for child in self.children:
            yield from child.walk(include_self=True)

    def find(
        self,
        *,
        kind: Optional[str] = None,
        label: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List["ResultNode"]:
        """Find descendants matching simple predicates."""
        out: List[ResultNode] = []
        for node in self.walk():
            if kind is not None and node.kind != kind:
                continue
            if label is not None and node.label != label:
                continue
            if min_score is not None and (node.score is None or node.score < min_score):
                continue
            out.append(node)
        return out

    def to_flat_rows(self, *, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Flatten a result tree into a list of dictionaries."""
        row = {
            "id": self.id,
            "parent_id": parent_id,
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "class_id": self.class_id,
            "payload_type": getattr(self.payload, "type", None),
            "value": to_jsonable(self.value),
            "metadata": to_jsonable(self.metadata),
        }
        rows = [row]
        for child in self.children:
            rows.extend(child.to_flat_rows(parent_id=self.id))
        return rows


class ResultSet(BaseModel):
    """A collection of top-level AI result nodes."""

    items: List[ResultNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return to_jsonable(value)

    def add(self, node: ResultNode) -> ResultNode:
        self.items.append(node)
        return node

    def extend(self, nodes: Iterable[ResultNode]) -> None:
        self.items.extend(nodes)

    def walk(self) -> Iterator[ResultNode]:
        for item in self.items:
            yield from item.walk(include_self=True)

    def find(
        self,
        *,
        kind: Optional[str] = None,
        label: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[ResultNode]:
        out: List[ResultNode] = []
        for node in self.items:
            out.extend(node.find(kind=kind, label=label, min_score=min_score))
        return out

    def to_flat_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in self.items:
            rows.extend(item.to_flat_rows())
        return rows


ResultNode.model_rebuild()
ResultSet.model_rebuild()
