import tempfile
from pathlib import Path
import unittest

from custom_record_store import (
    CustomStore,
    record_id_from_timestamp_ns,
    timestamp_ns_from_date_and_record_id,
)


class CustomRecordStoreTests(unittest.TestCase):
    def test_timestamp_roundtrip(self):
        ts = 1783521310000000000
        record_id = record_id_from_timestamp_ns(ts, 1)
        self.assertEqual(record_id, "143510.000000000JST")
        self.assertEqual(timestamp_ns_from_date_and_record_id("2026-07-08", record_id), ts)

    def test_rgb_stereo_minimal_record_validation(self):
        from PIL import Image

        root = Path(tempfile.mkdtemp()) / "recording"
        ts = 1783521310000000000
        store = CustomStore(root)
        record = store.add_record("rgb_stereo", ts, "field_all")

        img = Image.new("RGB", (2, 2))
        disp = Image.new("I;16", (2, 2))
        record.add_image("cam_c", "rgb", img)
        record.add_image("cam_c", "left", img)
        record.add_image("cam_c", "right", img)
        record.add_disparity("cam_c", disp, {"algorithm": "sgbm"})
        record.add_calibration("cam_c", {"camera_id": "cam_c"})
        record.add_gnss("location", {"timestamp_ns": ts})
        record.add_gnss("pose", {"timestamp_ns": ts})
        record.add_gnss("coordinate_system", {"frame": "local"})
        issues = record.close(validate=True)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
