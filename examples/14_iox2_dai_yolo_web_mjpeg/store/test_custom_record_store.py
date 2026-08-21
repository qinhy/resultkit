from __future__ import annotations

import json
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import custom_record_store as store_module


class TimestampTests(unittest.TestCase):
    def test_timestamp_round_trip_across_jst_date_boundary(self) -> None:
        for hour in (0, 14, 15, 23):
            dt = datetime(2026, 8, 19, hour, 30, tzinfo=timezone.utc)
            timestamp_ns = int(dt.timestamp()) * store_module.NS_PER_SECOND + 123_456_789
            record_id, date_utc = store_module.timestamp_info(timestamp_ns)
            reconstructed = store_module.timestamp_ns_from_date_and_record_id(
                date_utc, record_id
            )
            self.assertEqual(reconstructed, timestamp_ns)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = store_module.CustomStore(self.temp_dir.name)
        dt = datetime(2026, 8, 19, 16, 2, 3, tzinfo=timezone.utc)
        self.timestamp_ns = (
            int(dt.timestamp()) * store_module.NS_PER_SECOND + 987_654_321
        )
        self.record = self.store.add_record(
            "dual_rgb", self.timestamp_ns, field_id="field_all"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_get_and_list_record(self) -> None:
        loaded = self.store.get_record(
            "dual_rgb",
            self.record.date_utc,
            self.record.field_id,
            self.record.record_id,
        )
        self.assertEqual(loaded.timestamp_ns_utc, self.timestamp_ns)
        self.assertEqual(self.store.list_records(), [self.record.path])
        self.assertEqual(self.store.list_record_objects()[0].path, self.record.path)

    def test_calibration_gnss_yolo_and_images(self) -> None:
        calibration_path = self.record.add_calibration("cam_a", {"fx": 100.0})
        self.assertEqual(json.loads(calibration_path.read_text()), {"fx": 100.0})

        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        self.assertTrue(self.record.add_image("cam_a", "rgb", rgb).is_file())

        gray = np.zeros((8, 10), dtype=np.uint8)
        self.assertTrue(self.record.add_image("cam_b", "left", gray).is_file())

        gnss_path = self.record.add_gnss({"lat": 35.0})
        self.assertTrue(gnss_path.is_file())
        self.assertEqual(
            store_module.GnssRecord(parent=self.record).load(), {"lat": 35.0}
        )

        self.record.add_yolo(
            "cam_a",
            "rgb",
            {"detections": [{"bbox_xyxy": [1, 2, 3, 4]}]},
        )
        yolo = self.record.get_camera("cam_a").images["rgb"].yolo_record()
        self.assertEqual(yolo.get_bbox_xyxy(), [[1, 2, 3, 4]])

    def test_rejects_path_traversal_components(self) -> None:
        with self.assertRaises(ValueError):
            self.record.get_camera("../escape")


class PCDTests(unittest.TestCase):
    def test_ascii_and_binary_pcd(self) -> None:
        packed_rgb = (255 << 16) | (128 << 8) | 7
        rgb_float = np.array([packed_rgb], dtype=np.uint32).view(np.float32)[0]
        header = (
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            "WIDTH 1\n"
            "HEIGHT 1\n"
            "POINTS 1\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ascii_path = root / "ascii.pcd"
            ascii_path.write_text(
                header + f"DATA ascii\n1 2 3 {rgb_float:.9e}\n",
                encoding="ascii",
            )
            points, colors = store_module._read_pcd(ascii_path)
            np.testing.assert_allclose(points, [[1, 2, 3]])
            self.assertEqual(colors.tolist(), [[255, 128, 7]])

            binary_path = root / "binary.pcd"
            binary_path.write_bytes(
                (header + "DATA binary\n").encode("ascii")
                + struct.pack("<ffff", 1.0, 2.0, 3.0, float(rgb_float))
            )
            points, colors = store_module._read_pcd(binary_path)
            np.testing.assert_allclose(points, [[1, 2, 3]])
            self.assertEqual(colors.tolist(), [[255, 128, 7]])


if __name__ == "__main__":
    unittest.main()
