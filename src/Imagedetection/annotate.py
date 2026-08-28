"""
Interactive dataset-annotation helper: draw a box on a photo (or a live camera
frame) and save just that cropped region straight into the right
Data/training/<class> or Data/validation/<class> folder - so a tester never
has to hand-crop photos in an external editor, manually sort files, or
calculate pixel coordinates for capture_and_predict's x/y/width/height
arguments by trial and error.

This file has no dependency on Base_detection.py / torch - it only touches
cv2 and the filesystem. It's a data-prep tool a tester runs before writing
any Robot Framework suite, not a library keyword.

Usage:
    # Annotate every photo already sitting in a folder:
    uv run python -m Imagedetection.annotate --input-dir raw_photos/ --class Celsius --out atest/Data

    # Annotate live from a webcam:
    uv run python -m Imagedetection.annotate --camera 0 --class Celsius --out atest/Data

Each saved crop is randomly routed to training/ or validation/ (--val-ratio,
default 20% to validation) so one photo session produces both splits with no
duplicate photos between them - copying the same photo into both would make
validation accuracy meaningless (the model would just be "tested" on
something it already memorized).

Shortcuts (also shown in the window's hint bar):
    Enter       Save the current box's crop, move to the next photo
    Esc         Clear the box you just drew (stay on this photo) - or, if
                nothing's drawn, skip this photo and move to the next
    Backspace   Same as Esc's clear - kept as a fallback
    Left/Right  Go to the previous/next photo
    Space       (camera mode only) pause the live feed to draw / resume it
    Q           Quit

A magnifier in the bottom-right corner shows the pixels around the cursor at
5x before you've started a box, then switches to showing the whole box plus
some margin once one's drawn - useful since these targets are often only
~20-30px in a much bigger frame.
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2

WINDOW = "Imagedetection Annotator"

HELP_TEXT_FILE = "Enter=Save  Esc=Undo box (or Skip if none)  Left/Right=Prev/Next  Q=Quit"
HELP_TEXT_CAMERA = "Space=Pause/Resume  " + HELP_TEXT_FILE

# Extended key codes cv2.waitKeyEx() returns for the arrow keys on Windows -
# these differ on Linux/macOS (typically 65361/65363 under GTK backends), so
# adjust here if this ever needs to run cross-platform.
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_ENTER = 13
KEY_ESC = 27
KEY_BACKSPACE = 8
KEY_SPACE = 32


class _BoxDrawer:
    """
    Tracks a click-and-drag rectangle on whatever frame is currently shown.

    The box locks the instant the mouse button is released - a stray click or
    bump while reaching for Enter can't silently start a new drag and discard
    a carefully-placed selection. Esc/Backspace (see _annotate_frame) is the
    only way back into drawing mode from a locked box.
    """

    def __init__(self):
        self.box = None  # (x, y, w, h) once a drag has produced a rectangle
        self.cursor = None  # (x, y) - tracked on every move, drag or not, for the magnifier
        self._dragging = False
        self._locked = False
        self._start = (0, 0)

    def reset(self):
        self.box = None
        self._dragging = False
        self._locked = False

    def on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        if self._locked:
            return  # box is finalized - ignore clicks/drags until reset()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._start = (x, y)
            self.box = None
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self.box = self._rect_from(self._start, (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self.box = self._rect_from(self._start, (x, y))
            if self.box[2] > 0 and self.box[3] > 0:
                self._locked = True

    @staticmethod
    def _rect_from(p1, p2):
        x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
        return (x0, y0, x1 - x0, y1 - y0)


# Magnifier: shows either the pixels right around the cursor (before a box
# exists - helps place the first click precisely) or the whole box plus some
# margin on all four sides (once one's been drawn - so the full selection and
# its surroundings are visible, not just whatever's under the cursor right
# now). Always nearest-neighbor so pixel edges stay crisp, not blurred.
MAGNIFIER_RADIUS = 15       # source pixels shown on each side of the cursor, no-box case
MAGNIFIER_BOX_MARGIN = 15   # extra source pixels shown around the box on each side
MAGNIFIER_SCALE = 5
MAGNIFIER_MAX_OUTPUT = 320  # cap the on-screen inset size if an unusually large box is drawn
MAGNIFIER_MARGIN = 10


def _draw_magnifier(display, cursor, box):
    """
    Draws the magnifier by reading from `display` itself (which already has
    the green box outline drawn on it, if any) - so the box is visible inside
    the zoomed inset too, not just the raw pixels.
    """
    if cursor is None:
        return
    dh, dw = display.shape[:2]

    has_box = box is not None and box[2] > 0 and box[3] > 0
    if has_box:
        bx, by, bw, bh = box
        x0 = max(0, bx - MAGNIFIER_BOX_MARGIN)
        y0 = max(0, by - MAGNIFIER_BOX_MARGIN)
        x1 = min(dw, bx + bw + MAGNIFIER_BOX_MARGIN)
        y1 = min(dh, by + bh + MAGNIFIER_BOX_MARGIN)
    else:
        cx, cy = cursor
        x0, y0 = max(0, cx - MAGNIFIER_RADIUS), max(0, cy - MAGNIFIER_RADIUS)
        x1, y1 = min(dw, cx + MAGNIFIER_RADIUS), min(dh, cy + MAGNIFIER_RADIUS)

    if x1 <= x0 or y1 <= y0:
        return

    patch = display[y0:y1, x0:x1].copy()
    # Keep the fixed 5x scale for the normal small-box/cursor case, but back
    # it off if an unusually large box would otherwise blow the inset up.
    scale = min(MAGNIFIER_SCALE, MAGNIFIER_MAX_OUTPUT / max(x1 - x0, y1 - y0))
    zoomed = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zh, zw = zoomed.shape[:2]

    if not has_box:
        # No box yet - a crosshair on the cursor is the useful reference.
        cx, cy = cursor
        mx, my = int((cx - x0) * scale), int((cy - y0) * scale)
        cv2.line(zoomed, (mx, 0), (mx, zh), (0, 255, 0), 1)
        cv2.line(zoomed, (0, my), (zw, my), (0, 255, 0), 1)
    cv2.rectangle(zoomed, (0, 0), (zw - 1, zh - 1), (255, 255, 255), 1)

    # Place it in the bottom-right corner, clear of the hint bar up top.
    ox, oy = dw - zw - MAGNIFIER_MARGIN, dh - zh - MAGNIFIER_MARGIN
    if ox >= 0 and oy >= 0:
        display[oy:oy + zh, ox:ox + zw] = zoomed


def _draw_overlay(frame, box, hint, status_line="", cursor=None):
    display = frame.copy()
    if box is not None and box[2] > 0 and box[3] > 0:
        x, y, w, h = box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Sourced from `display` (post-box-draw) so the box shows up inside the
    # magnified inset too, once one exists.
    _draw_magnifier(display, cursor, box)

    bar_h = 44 if status_line else 24
    cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(display, hint, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    if status_line:
        cv2.putText(display, status_line, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return display


def _annotate_frame(frame, hint, status_line=""):
    """
    Show one still frame, let the tester drag a box on it, and wait for a
    decision key. Returns (action, box): action is one of
    "save"/"skip"/"prev"/"next"/"pause"/"quit", and box is the (x, y, w, h)
    tuple to crop (only meaningful when action == "save").
    """
    drawer = _BoxDrawer()
    cv2.setMouseCallback(WINDOW, drawer.on_mouse)

    while True:
        cv2.imshow(WINDOW, _draw_overlay(frame, drawer.box, hint, status_line, drawer.cursor))
        key = cv2.waitKeyEx(30)

        if key == -1:
            continue
        if key in (KEY_ENTER, 10):
            if drawer.box and drawer.box[2] > 0 and drawer.box[3] > 0:
                return "save", drawer.box
            # No box drawn yet - Enter does nothing, keep waiting for one.
        elif key == KEY_ESC:
            # Context-aware: clear the box if one's drawn (so a tester can
            # keep their left hand on Esc to undo and redraw); only skip the
            # photo once there's nothing left to clear.
            if drawer.box:
                drawer.reset()
            else:
                return "skip", None
        elif key == KEY_BACKSPACE:
            drawer.reset()
        elif key == KEY_LEFT:
            return "prev", None
        elif key == KEY_RIGHT:
            return "next", None
        elif key == KEY_SPACE:
            return "pause", None
        elif key in (ord("q"), ord("Q")):
            return "quit", None


def _save_crop(image, box, class_name, out_root, val_ratio, rng):
    x, y, w, h = box
    crop = image[y:y + h, x:x + w]
    split = "validation" if rng.random() < val_ratio else "training"
    class_dir = Path(out_root) / split / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{class_name.lower()}_{int(time.time() * 1000)}.png"
    path = class_dir / filename
    cv2.imwrite(str(path), crop)
    return path


def _finalize_roi(box, out_root):
    """
    Prints the last-used crop region for reference, and - more importantly -
    saves it as <out_root>/last_roi.json, where train_model() picks it up
    automatically and embeds it in the trained checkpoint. That's the
    durable place this value lives; the printout is just a convenience, not
    the source of truth to hand-copy into a .robot file (that copy is what
    goes stale when a tester recalibrates and forgets to update it there).
    """
    x, y, w, h = box
    roi = {"x": x, "y": y, "width": w, "height": h}
    roi_path = Path(out_root) / "last_roi.json"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text(json.dumps(roi))

    print()
    print(f"Saved crop region to {roi_path} - Train Model will embed it in the")
    print("checkpoint automatically, so Detect From Webcam picks it up on its own.")
    print(f"    x={x}    y={y}    width={w}    height={h}")


def _count_photos(out_root, class_name):
    """Returns (training_count, validation_count) already saved for this class."""
    root = Path(out_root)
    counts = []
    for split in ("training", "validation"):
        split_dir = root / split / class_name
        counts.append(sum(1 for p in split_dir.iterdir() if p.is_file()) if split_dir.is_dir() else 0)
    return tuple(counts)


def _print_session_banner(class_name, out_root, val_ratio):
    """
    Prints exactly where this session's crops will land, *before* the tester
    starts annotating - so a wrong --out or a copy-pasted --class from a
    previous run is obvious immediately instead of discovered afterwards in
    the file browser (see the training/Fahrenheit/training/Celsius double-
    nesting + mislabeling this caught in practice).
    """
    root = Path(out_root)
    train_dir = root / "training" / class_name
    val_dir = root / "validation" / class_name
    train_count, val_count = _count_photos(out_root, class_name)
    print()
    print(f"Class:      {class_name}")
    print(f"Saving to:  {train_dir}   (currently {train_count} photo(s))")
    print(f"      and:  {val_dir}   (currently {val_count} photo(s))   (~{val_ratio:.0%} of new crops)")

    reserved = {"training", "validation"}
    hit = next((part for part in root.parts if part.lower() in reserved), None)
    if hit is not None:
        print(f"WARNING: --out ({root}) already contains '{hit}' as a path segment.")
        print("         --out should be the Data ROOT only (e.g. 'Data') - this tool")
        print("         adds training/<class> or validation/<class> itself, so this")
        print("         will likely nest one inside the other. Ctrl+C now to fix it.")
    print()


def run_input_dir(input_dir, class_name, out_root, val_ratio, seed=None):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    photos = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts)
    if not photos:
        print(f"No photos found in {input_dir}")
        return

    _print_session_banner(class_name, out_root, val_ratio)
    print(f"{len(photos)} photo(s) to annotate. {HELP_TEXT_FILE}")
    rng = random.Random(seed)
    cv2.namedWindow(WINDOW)
    saved = skipped = 0
    last_box = None
    index = 0

    while 0 <= index < len(photos):
        photo_path = photos[index]
        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"Could not read {photo_path}, skipping.")
            index += 1
            continue

        train_count, val_count = _count_photos(out_root, class_name)
        status = (f"Class: {class_name}  |  Photo {index + 1}/{len(photos)}: {photo_path.name}"
                  f"  |  Saved so far: {train_count} training / {val_count} validation")
        action, box = _annotate_frame(image, HELP_TEXT_FILE, status)

        if action == "save":
            path = _save_crop(image, box, class_name, out_root, val_ratio, rng)
            last_box = box
            print(f"Saved {path}")
            saved += 1
            index += 1
        elif action == "skip":
            skipped += 1
            index += 1
        elif action == "prev":
            index = max(0, index - 1)
        elif action == "next":
            index = min(len(photos) - 1, index + 1)
        elif action == "quit":
            break
        # "pause" has no meaning outside camera mode - _annotate_frame never
        # returns it here unless Space is pressed, in which case just ignore.

    cv2.destroyAllWindows()
    print(f"Done. Saved {saved}, skipped {skipped}.")
    if last_box:
        _finalize_roi(last_box, out_root)


def _lock_focus(camera):
    """
    Best-effort: trigger one autofocus pass, then lock focus wherever it
    lands - mirroring the "click to focus, then it holds" behavior a tester
    would otherwise do manually in their camera's own app (e.g. Windows
    Camera). A webcam built for video calls tends to keep hunting/refocusing
    on whatever looks most face-like, which is the wrong behavior for a
    fixed close-up of a small symbol.

    Safe to call on any camera: cv2.VideoCapture.set() returns False (not an
    exception) when the camera/driver doesn't support a property, so this is
    a no-op - not a crash - on hardware with no focus control at all. Wrapped
    in try/except too, as a second safety net against an unusual backend
    raising instead of returning False.
    """
    try:
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        for _ in range(10):  # give the autofocus algorithm a moment to settle
            camera.read()
        locked = camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except cv2.error:
        locked = False

    if locked:
        print("Autofocus triggered once and locked for a stable session.")
    else:
        print("This camera doesn't support autofocus control - continuing with whatever focus it's currently using.")


def run_camera(camera_index, class_name, out_root, val_ratio, seed=None):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Error: camera (index {camera_index}) not accessible.")

    _lock_focus(camera)
    _print_session_banner(class_name, out_root, val_ratio)
    print(f"Live feed - Space to pause and draw a box. {HELP_TEXT_CAMERA}")
    rng = random.Random(seed)
    cv2.namedWindow(WINDOW)
    saved = skipped = 0
    train_count, val_count = _count_photos(out_root, class_name)  # updated incrementally below
    last_box = None
    session = []       # (frame, box) pairs already saved this session, for review
    review_index = -1  # -1 = live feed; >=0 = reviewing session[review_index]

    try:
        while True:
            if review_index == -1:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Error: failed to read a frame from the camera.")
                    break
                live_status = (f"Class: {class_name}  |  LIVE - Space to pause"
                              f"  |  Saved so far: {train_count} training / {val_count} validation")
                cv2.imshow(WINDOW, _draw_overlay(frame, None, HELP_TEXT_CAMERA, live_status))
                key = cv2.waitKeyEx(30)

                if key == KEY_SPACE:
                    status = (f"Class: {class_name}  |  PAUSED"
                             f"  |  Saved so far: {train_count} training / {val_count} validation")
                    action, box = _annotate_frame(frame, HELP_TEXT_CAMERA, status)
                    if action == "save":
                        path = _save_crop(frame, box, class_name, out_root, val_ratio, rng)
                        last_box = box
                        print(f"Saved {path}")
                        saved += 1
                        if path.parent.parent.name == "validation":
                            val_count += 1
                        else:
                            train_count += 1
                        session.append((frame, box))
                    elif action == "skip":
                        skipped += 1
                    elif action == "prev" and session:
                        review_index = len(session) - 1
                    elif action == "quit":
                        break
                    # "next"/"pause" while live-paused just falls through and
                    # resumes the live feed on the next loop iteration.
                elif key in (ord("q"), ord("Q")):
                    break
            else:
                frame, box = session[review_index]
                status = (f"Class: {class_name}  |  Reviewing saved {review_index + 1}/{len(session)}"
                         f"  |  Saved so far: {train_count} training / {val_count} validation")
                action, new_box = _annotate_frame(frame, HELP_TEXT_CAMERA, status)
                if action == "save" and new_box:
                    path = _save_crop(frame, new_box, class_name, out_root, val_ratio, rng)
                    last_box = new_box
                    print(f"Re-saved {path}")
                    saved += 1
                    if path.parent.parent.name == "validation":
                        val_count += 1
                    else:
                        train_count += 1
                    session[review_index] = (frame, new_box)
                elif action == "prev":
                    review_index = max(0, review_index - 1)
                elif action in ("next", "skip"):
                    review_index += 1
                    if review_index >= len(session):
                        review_index = -1  # back to live
                elif action == "quit":
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved}, skipped {skipped}.")
    if last_box:
        _finalize_roi(last_box, out_root)


def main():
    parser = argparse.ArgumentParser(
        description="Draw a box on a photo (or live camera) and save the crop "
                    "straight into Data/training or Data/validation.",
    )
    parser.add_argument("--class", dest="class_name", required=True,
                        help="Class name for this annotation session, e.g. Celsius")
    parser.add_argument("--out", default="Data",
                        help="Data root to save into (creates <out>/training/<class>/ "
                            "and <out>/validation/<class>/). Default: Data")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of saved crops randomly routed to validation "
                            "instead of training. Default: 0.2")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Folder of already-taken photos to annotate")
    source.add_argument("--camera", type=int, metavar="INDEX",
                        help="Camera index to capture live from, e.g. 0")
    args = parser.parse_args()

    if args.input_dir:
        run_input_dir(args.input_dir, args.class_name, args.out, args.val_ratio)
    else:
        run_camera(args.camera, args.class_name, args.out, args.val_ratio)


if __name__ == "__main__":
    main()
