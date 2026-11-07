"""Submission checks that require no torch, model, or competition download."""

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "submission.py"
SPEC = importlib.util.spec_from_file_location("vessel_submission_under_test", MODULE_PATH)
submission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submission
SPEC.loader.exec_module(submission)


def sample_csv(tmp_path):
    path = tmp_path / "sample_submission.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Predicted"])
        writer.writerows((str(i), "") for i in range(1, 21))
    return path


def probabilities():
    array = np.zeros((41, 37), dtype=np.float32)
    array[4:28, 11:14] = 0.9
    array[10:12, 4:30] = 0.6
    return {str(i): array.copy() for i in range(1, 21)}


def test_rle_first_last_pixel_and_row_order():
    mask = np.zeros((512, 512), dtype=bool)
    mask[0, 0] = mask[-1, -1] = True
    mask[0, 2:5] = True
    mask[1, 0] = True
    encoded = submission.encode_rle(mask)
    assert encoded == "1 1 3 3 513 1 262144 1"
    assert np.array_equal(submission.decode_rle(encoded), mask)


def test_rle_rectangle_empty_and_full():
    rectangle = np.zeros((3, 5), dtype=bool)
    rectangle[1:3, 1:4] = True
    assert submission.encode_rle(rectangle) == "7 3 12 3"
    assert np.array_equal(submission.decode_rle("7 3 12 3", (3, 5)), rectangle)
    assert submission.encode_rle(np.zeros((512, 512), bool)) == ""
    assert not submission.decode_rle(" \n ").any()
    assert submission.encode_rle(np.ones((512, 512), bool)) == "1 262144"
    assert submission.decode_rle("1 262144").all()


@pytest.mark.parametrize(
    "text",
    ["1", "0 2", "1 0", "-1 2", "1 -2", "1.0 2", "+1 2", "nan 1",
     "1 3 3 2", "10 1 1 2", "262144 2", "262145 1"],
)
def test_decode_rejects_malformed_overlap_and_bounds(text):
    with pytest.raises(ValueError):
        submission.decode_rle(text)


def test_official_resize_dilates_and_is_not_polarity_symmetric(tmp_path):
    mask = np.zeros((584, 565), dtype=np.uint8)
    mask[40:540, 200:203] = 255
    image = Image.fromarray(mask)
    expected = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BILINEAR)) > 0
    nearest = np.asarray(image.resize((512, 512), Image.Resampling.NEAREST)) > 0
    actual = submission.official_mask(mask)
    assert np.array_equal(actual, expected)
    assert actual.sum() > nearest.sum()
    path = tmp_path / "lossless.png"
    image.save(path)
    assert np.array_equal(submission.official_mask(path), actual)
    # Both sides of interpolated boundaries are nonzero; inverting before resize
    # is not the same as inverting the post-resize boolean mask.
    inverse = submission.official_mask(255 - mask)
    assert not np.array_equal(inverse, ~actual)
    assert (inverse & actual).any()


@pytest.mark.parametrize("polarity", ["vessel-white", "vessel-black"])
@pytest.mark.parametrize("resize_mode", ["official", "probability"])
def test_export_roundtrip_modes_polarities_and_sample_order(tmp_path, polarity, resize_mode):
    samples = sample_csv(tmp_path)
    predictions = probabilities()
    metadata = submission.export_submission(
        predictions, tmp_path / "export", samples, 0.5, polarity, resize_mode
    )
    assert metadata["roundtrip_verified"]
    assert not metadata["official_script_verified"]
    assert "unknown" in metadata["hidden_ground_truth"]
    with Path(metadata["submission_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Id"] for row in rows] == [str(i) for i in range(1, 21)]
    for row in rows:
        path = Path(metadata["image_dir"]) / f"{row['Id']}.png"
        pixels = np.asarray(Image.open(path))
        assert set(np.unique(pixels)) <= {0, 255}
        assert pixels.shape == ((41, 37) if resize_mode == "official" else (512, 512))
        assert np.array_equal(submission.decode_rle(row["Predicted"]), submission.official_mask(path))
    if resize_mode == "official":
        expected = predictions["1"] >= 0.5
    else:
        resized = np.asarray(Image.fromarray(predictions["1"]).resize((512, 512), Image.Resampling.BILINEAR))
        expected = resized >= 0.5
    if polarity == "vessel-black":
        expected = ~expected
    assert np.array_equal(np.asarray(Image.open(Path(metadata["image_dir"]) / "1.png")) > 0, expected)
    # Reruns may overwrite artifacts that carry this exporter's ownership marker.
    submission.export_submission(predictions, tmp_path / "export", samples, 0.6, polarity, resize_mode)


def test_export_rejects_mismatched_ids_bad_probabilities_and_stale_images(tmp_path):
    samples = sample_csv(tmp_path)
    predictions = probabilities()
    predictions.pop("20")
    with pytest.raises(ValueError, match="IDs mismatch"):
        submission.export_submission(predictions, tmp_path / "missing", samples, 0.5)
    predictions = probabilities()
    predictions["1"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        submission.export_submission(predictions, tmp_path / "nan", samples, 0.5)
    stale = tmp_path / "stale" / "image"
    stale.mkdir(parents=True)
    Image.new("L", (512, 512)).save(stale / "preview.png")
    with pytest.raises(ValueError, match="stale"):
        submission.export_submission(probabilities(), stale.parent, samples, 0.5)
    assert (stale / "preview.png").exists()


def test_export_does_not_overwrite_unowned_csv(tmp_path):
    samples = sample_csv(tmp_path)
    output_dir = tmp_path / "unowned"
    output_dir.mkdir()
    path = output_dir / "submission.csv"
    path.write_text("unrelated content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="metadata"):
        submission.export_submission(probabilities(), output_dir, samples, 0.5)
    assert path.read_text(encoding="utf-8") == "unrelated content"


def test_original_script_is_run_and_its_rows_checked_by_id(tmp_path):
    samples = sample_csv(tmp_path)
    # A small independent reference converter lets this test exercise the
    # subprocess path without requiring torchvision or running model code.
    script = tmp_path / "reference_converter.py"
    script.write_text(
        "import csv\nfrom pathlib import Path\nimport numpy as np\nfrom PIL import Image\n"
        "rows=[]\n"
        "for path in sorted(Path('image').glob('*.png'), reverse=True):\n"
        " a=np.asarray(Image.open(path).convert('L').resize((512,512), Image.Resampling.BILINEAR))\n"
        " dots=np.flatnonzero(a.flatten()>0); runs=[]; previous=-2\n"
        " for point in dots:\n"
        "  if point>previous+1: runs.extend([int(point+1),0])\n"
        "  runs[-1]+=1; previous=point\n"
        " rows.append([path.stem,' '.join(map(str,runs))])\n"
        "with open('submission.csv','w',newline='') as f:\n"
        " w=csv.writer(f); w.writerow(['Id','Predicted']); w.writerows(rows)\n",
        encoding="utf-8",
    )
    metadata = submission.export_submission(
        probabilities(), tmp_path / "reference_export", samples, 0.5, official_script=script
    )
    assert metadata["official_script_verified"]
    with Path(metadata["submission_csv"]).open(newline="") as handle:
        assert [row["Id"] for row in csv.DictReader(handle)] == [str(i) for i in range(1, 21)]
