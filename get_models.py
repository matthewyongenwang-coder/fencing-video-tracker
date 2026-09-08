"""
get_models.py

Downloads the three pretrained models this project uses.

    python3 get_models.py

They are NOT in the repository. Together they are about 42MB, they never
change, and they are published by OpenCV, so keeping them in git would
bloat every clone forever to save one command. Run this once after
cloning.

Nothing here is trained by this project. All three are released models
being used as-is.
"""

import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = [
    {
        "file": "yolox_person.onnx",
        "url": f"{ZOO}/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        "size_mb": 36,
        "needed_for": "--detect (person detection). The important one: it is "
                      "what stops a box drifting onto a sponsor banner.",
    },
    {
        "file": "pose_landmarks.onnx",
        "url": f"{ZOO}/pose_estimation_mediapipe/"
               f"pose_estimation_mediapipe_2023mar.onnx",
        "size_mb": 5,
        "needed_for": "--pose (body landmarks: feet, wrists, joints).",
    },
    {
        "file": "vittrack.onnx",
        "url": f"{ZOO}/object_tracking_vittrack/"
               f"object_tracking_vittrack_2023sep.onnx",
        "size_mb": 1,
        "needed_for": "--tracker vit (optional alternative tracker).",
    },
]

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def _via_urllib(url, target):
    """
    Straight download. Needs a working certificate bundle, which a
    python.org install on macOS often does NOT have. It ships without
    running Install Certificates.command, and every https fetch then
    dies with CERTIFICATE_VERIFY_FAILED. Hence the curl fallback below.
    """
    context = ssl.create_default_context()
    def progress(count, block, total):
        if total > 0:
            done = min(100, 100 * count * block / total)
            sys.stdout.write(f"\r      {done:5.1f}%")
            sys.stdout.flush()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, target, progress)


def _via_curl(url, target):
    """
    Fallback. curl is on every Mac and uses the system keychain, so it
    works when Python's own certificate store does not.
    """
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl not found")
    subprocess.run([curl, "-fsSL", "--retry", "2", "-o", target, url],
                   check=True)


def download(entry):
    target = os.path.join(MODEL_DIR, entry["file"])
    if os.path.exists(target) and os.path.getsize(target) > 100_000:
        have = os.path.getsize(target) / 1e6
        print(f"  already have {entry['file']}  ({have:.1f} MB)")
        return True

    print(f"  downloading {entry['file']}  (~{entry['size_mb']} MB)"
          f"\n      {entry['needed_for']}")

    first_error = None
    for attempt in (_via_urllib, _via_curl):
        try:
            attempt(entry["url"], target)
            if os.path.getsize(target) < 100_000:
                raise RuntimeError("file came back suspiciously small")
            print(f"\r      done, {os.path.getsize(target) / 1e6:.1f} MB")
            return True
        except Exception as exc:                           # noqa: BLE001
            if first_error is None:
                first_error = exc
            if os.path.exists(target):
                os.remove(target)

    print(f"\r      FAILED: {first_error}")
    return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Fetching models into {MODEL_DIR}\n")

    results = [download(entry) for entry in MODELS]

    print()
    if all(results):
        print("All models ready. Check everything works with:")
        print('  python3 selftest.py "/path/to/a/fencing/clip.MOV"')
        return 0

    print("Some downloads failed. The tracker still runs without them:")
    print("  - without yolox_person.onnx you cannot use --detect")
    print("  - without pose_landmarks.onnx you cannot use --pose")
    print("  - without vittrack.onnx you cannot use --tracker vit")
    print("Plain tracking works with no models at all.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
