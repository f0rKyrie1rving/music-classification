"""Unit tests for bounded audio and model-file acquisition checks."""

import unittest
import hashlib
import tempfile
from pathlib import Path

from prepare_dataset import leading_id3_size
from prepare_mert import verify


class AcquisitionTests(unittest.TestCase):
    def test_no_id3(self):
        self.assertEqual(leading_id3_size(b"\xff\xfb" + b"\0" * 8), 0)

    def test_synchsafe_size_and_footer(self):
        header = b"ID3\x04\x00\x00\x01\x45\x02\x57"
        self.assertEqual(leading_id3_size(header), 3228001)
        footer_header = header[:5] + b"\x10" + header[6:]
        self.assertEqual(leading_id3_size(footer_header), 3228011)
        older_header = b"ID3\x03\x00\x00\x00\x00\x00\x7f"
        self.assertEqual(leading_id3_size(older_header), 137)

    def test_invalid_headers(self):
        for header in [b"ID3", b"ID3\x09\x00\x00\x00\x00\x00\x00",
                       b"ID3\x04\x00\x00\x80\x00\x00\x00"]:
            with self.subTest(header=header), self.assertRaises(ValueError):
                leading_id3_size(header)

    def test_model_download_accepts_sha256_and_git_blob_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample"
            path.write_bytes(b"abc")
            sha256 = hashlib.sha256(b"abc").hexdigest()
            git_blob = hashlib.sha1(b"blob 3\0abc").hexdigest()
            self.assertEqual(verify(path, (3, sha256)), sha256)
            self.assertEqual(verify(path, (3, git_blob)), sha256)

    def test_model_download_rejects_wrong_size_or_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample"
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "Size mismatch"):
                verify(path, (4, "0" * 64))
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify(path, (3, "0" * 64))


if __name__ == "__main__":
    unittest.main()
