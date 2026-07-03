# resultkit

`resultkit` is a small starter package for structured AI model outputs.

It is designed for results that are more expressive than a flat detection table:

- bounding boxes
- masks
- polygons
- keypoints
- embeddings / numeric vectors
- OCR, ASR, NER, or LLM text spans
- parent-child result trees
- JSON serialization
- Pydantic validation

## Install locally

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Quick example

```python
from resultkit import Box, Keypoints, ResultNode, ResultSet, TextSpan

person = ResultNode(
    kind="object",
    label="person",
    score=0.98,
    payload=Box.xyxy([10, 20, 220, 440], image_size=(640, 480)),
)

person.add_child(
    ResultNode(
        kind="part",
        label="face",
        score=0.95,
        payload=Box.xyxy([60, 35, 150, 140], image_size=(640, 480)),
    )
)

person.add_child(
    ResultNode(
        kind="pose",
        label="body_pose",
        payload=Keypoints(
            points=[[90, 80, 0.99], [80, 180, 0.91], [120, 180, 0.88]],
            names=["nose", "left_shoulder", "right_shoulder"],
            image_size=(640, 480),
        ),
    )
)

ocr = ResultNode(
    kind="ocr",
    label="invoice_number",
    score=0.93,
    payload=TextSpan(text="INV-2026-001", start=0, end=12),
)

results = ResultSet(items=[person, ocr], metadata={"source": "demo.jpg"})

print(results.model_dump_json(indent=2))
print(results.to_flat_rows())
```

## Core concepts

### `ResultNode`

A single AI result. It can represent a detection, segmentation, OCR span, embedding,
classification, or a higher-level grouped result.

Important fields:

- `kind`: broad type, such as `object`, `part`, `pose`, `ocr`, `embedding`, `classification`
- `label`: semantic label, such as `person`, `face`, `invoice_number`
- `score`: confidence score from 0 to 1
- `class_id`: optional numeric class ID
- `payload`: structured payload such as `Box`, `Mask`, `Keypoints`, `TextSpan`, or `Vector`
- `children`: nested child results
- `metadata`: arbitrary extra metadata

### `ResultSet`

A collection of top-level `ResultNode` objects.

Use it to store all outputs for one image, document, audio clip, video frame, or model call.

### `Box`

Bounding boxes support:

- `xyxy`
- `xywh`
- `cxcywh`
- raw pixel coordinates
- normalized `0..1` coordinates
- conversion
- area
- clipping
- normalization / denormalization
- pairwise IoU

```python
from resultkit import Box

box = Box.xywh([10, 20, 100, 200])
print(box.to("xyxy").data)
print(box.area())
```

## Notes

This is a starter implementation. Good next additions would be:

- COCO import/export
- YOLO import/export
- LabelMe import/export
- pandas/DataFrame export
- image/video frame IDs
- relation edges for graph-like outputs, not only trees
- batch vectorized detection containers
- visualization helpers

## Matrix backend helpers

`resultkit` also includes a small backend-neutral matrix operation layer inspired by
`MatOps`. This is useful when AI outputs may come from either NumPy or Torch, but
you want a consistent interface for common operations.

```python
from resultkit import Mat

mat = Mat(lib="numpy", dtype="float32")
x = mat.asarray([[1, 2], [3, 4]])
print(mat.ops.mean(x, dim=0))

# Torch is optional. Install with: pip install -e '.[torch]'
torch_mat = Mat(lib="torch", dtype="float32")
t = torch_mat.asarray([[1, 2], [3, 4]])
print(torch_mat.ops.to_numpy(t))
```

Available classes:

- `Mat`
- `MatLib`
- `MatOps`
- `NumpyMatOps`
- `TorchMatOps`

There is no `pycuda[gl]` extra. You enable PyCUDA OpenGL support **at build time** with:

```bash
--cuda-enable-gl
```

or by setting this in `siteconf.py`:

```python
CUDA_ENABLE_GL = True
```

PyCUDA’s build config has `CUDA_ENABLE_GL` disabled by default, and when enabled it adds the CUDA/OpenGL wrapper source and defines `HAVE_GL`. The current PyCUDA docs also show `pycuda.gl` APIs such as `make_context`, `RegisteredBuffer`, and `RegisteredImage`. ([GitHub][1])


## Build PyCUDA with `uv`

Then rebuild PyCUDA from source:

```bash
git clone https://github.com/inducer/pycuda.git
cd pycuda

uv pip install -U setuptools wheel numpy mako pytools platformdirs
uv run python configure.py --cuda-root="$CUDA_HOME" --cuda-enable-gl
uv pip install -v --no-build-isolation .
```
