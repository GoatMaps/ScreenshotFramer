#!/usr/bin/env python3

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageOps


INSTAGRAM_SIZE = (1080, 1080)


@dataclass
class ScreenBBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame one or more iPhone screenshots for Instagram (1080x1080).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("background", help="Background image file")
    parser.add_argument("frame", help="Device frame PNG with transparent screen cutout")
    parser.add_argument("screenshots", nargs="+", help="One or more screenshot images")
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Directory to write output images (created if missing)",
    )
    parser.add_argument(
        "--prefix",
        default="post",
        help="Output filename prefix (index appended for multiple outputs)",
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"Failed to open image '{path}': {exc}")


def detect_screen_bbox_from_frame(frame_img: Image.Image) -> Tuple[ScreenBBox, Image.Image]:
    """Detect the screen bounding box from the device frame PNG.

    Strategy:
    - Work on the alpha channel, treating alpha <= 5 as transparent.
    - Flood-fill transparency from image borders to mark outside/background transparency.
    - Remaining transparent pixels correspond to interior holes; choose the largest interior hole (expected to be the screen) and return its bounding box.
    - Fallback: if no interior hole exists, raise an error.
    """
    if frame_img.mode != "RGBA":
        frame_img = frame_img.convert("RGBA")

    alpha = frame_img.split()[3]
    width, height = frame_img.size

    # Build binary transparency mask
    # True where transparent (alpha <= 5), False otherwise
    alpha_data = alpha.load()
    transparent = [[alpha_data[x, y] <= 5 for x in range(width)] for y in range(height)]

    # Flood fill from borders to mark outside transparency
    from collections import deque

    visited = [[False] * width for _ in range(height)]
    q = deque()

    def enqueue_if_transparent(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height and transparent[y][x] and not visited[y][x]:
            visited[y][x] = True
            q.append((x, y))

    # Enqueue all border transparent pixels
    for x in range(width):
        enqueue_if_transparent(x, 0)
        enqueue_if_transparent(x, height - 1)
    for y in range(height):
        enqueue_if_transparent(0, y)
        enqueue_if_transparent(width - 1, y)

    # BFS
    while q:
        cx, cy = q.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            enqueue_if_transparent(nx, ny)

    # Any transparent pixel not visited is an interior hole
    # Compute bounding boxes of interior holes and pick largest by area
    best_bbox = None
    best_area = -1
    best_seed = None

    # Simple single pass to find bbox; no need to label components fully if we only need the overall bbox of all interior holes
    # But we prefer the largest contiguous interior region: perform a second BFS to label holes
    labeled = [[False] * width for _ in range(height)]

    def bfs_collect(x0: int, y0: int) -> Tuple[int, int, int, int, int]:
        q2 = deque()
        q2.append((x0, y0))
        labeled[y0][x0] = True
        min_x = max_x = x0
        min_y = max_y = y0
        count = 0
        while q2:
            x, y = q2.popleft()
            count += 1
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    if transparent[ny][nx] and not visited[ny][nx] and not labeled[ny][nx]:
                        labeled[ny][nx] = True
                        q2.append((nx, ny))
        return min_x, min_y, max_x, max_y, count

    for y in range(height):
        for x in range(width):
            if transparent[y][x] and not visited[y][x] and not labeled[y][x]:
                bx0, by0, bx1, by1, cnt = bfs_collect(x, y)
                area = (bx1 - bx0 + 1) * (by1 - by0 + 1)
                # Prefer larger area; if tie, prefer more pixels (denser hole)
                if area > best_area:
                    best_area = area
                    best_bbox = (bx0, by0, bx1, by1)
                    best_seed = (x, y)

    if best_bbox is None:
        raise RuntimeError("Could not detect interior transparent screen area in frame image.")

    left, top, right, bottom = best_bbox
    # Build a precise hole mask for the largest interior transparent region using inverted alpha values
    from collections import deque
    inv_alpha = ImageOps.invert(alpha)
    hole_mask = Image.new("L", (width, height), 0)
    if best_seed is not None:
        q3 = deque([best_seed])
        filled = set([best_seed])
        while q3:
            x, y = q3.popleft()
            hole_mask.putpixel((x, y), inv_alpha.getpixel((x, y)))
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    if transparent[ny][nx] and not visited[ny][nx] and (nx, ny) not in filled:
                        filled.add((nx, ny))
                        q3.append((nx, ny))

    # Convert to half-open box and return mask
    return ScreenBBox(left=left, top=top, right=right + 1, bottom=bottom + 1), hole_mask


def resize_to_cover(src: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    src_ratio = src.width / src.height
    tgt_w, tgt_h = target_size
    tgt_ratio = tgt_w / tgt_h
    if src_ratio > tgt_ratio:
        # Source is wider; height matches, width scales up
        new_h = tgt_h
        new_w = int(round(new_h * src_ratio))
    else:
        # Source is taller; width matches
        new_w = tgt_w
        new_h = int(round(new_w / src_ratio))
    return src.resize((new_w, new_h), Image.LANCZOS)


def center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    w, h = img.size
    tgt_w, tgt_h = size
    left = (w - tgt_w) // 2
    top = (h - tgt_h) // 2
    return img.crop((left, top, left + tgt_w, top + tgt_h))


def left_aligned_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    w, h = img.size
    tgt_w, tgt_h = size
    if w < tgt_w or h < tgt_h:
        raise RuntimeError("Background image is smaller than target crop size.")
    return img.crop((0, 0, tgt_w, tgt_h))


def crop_from_left(img: Image.Image, left: int, size: Tuple[int, int]) -> Image.Image:
    tgt_w, tgt_h = size
    if left + tgt_w > img.width or tgt_h > img.height:
        raise RuntimeError("Not enough background width for all screenshots; increase background width or reduce count.")
    return img.crop((left, 0, left + tgt_w, tgt_h))


def compose_frame(
    background: Image.Image,
    frame_img: Image.Image,
    scaled_screen_bbox: ScreenBBox,
    screenshot: Image.Image,
    frame_scaled_size: Tuple[int, int],
    screen_mask_scaled: Image.Image,
) -> Image.Image:
    # Prepare 1080x1080 canvas (RGBA)
    canvas = Image.new("RGBA", INSTAGRAM_SIZE, (0, 0, 0, 0))

    # Paste background already sized 1080x1080
    canvas.alpha_composite(background)

    # Compute centering offsets for the uniformly scaled frame
    frame_w, frame_h = frame_scaled_size
    offset_x = (INSTAGRAM_SIZE[0] - frame_w) // 2
    offset_y = (INSTAGRAM_SIZE[1] - frame_h) // 2

    # Resize frame uniformly to fit, then center and prepare mask for the screen region
    frame_scaled = frame_img.resize(frame_scaled_size, Image.LANCZOS)

    # Prepare screenshot to fit inside screen bbox while preserving aspect ratio
    screen_w, screen_h = scaled_screen_bbox.width, scaled_screen_bbox.height
    shot = screenshot.convert("RGBA")
    shot_resized = resize_to_cover(shot, (screen_w, screen_h))
    shot_cropped = center_crop(shot_resized, (screen_w, screen_h))

    # Extract the exact mask for the screen area from the precomputed interior-hole mask
    screen_mask = screen_mask_scaled.crop((
        scaled_screen_bbox.left,
        scaled_screen_bbox.top,
        scaled_screen_bbox.right,
        scaled_screen_bbox.bottom,
    ))

    # Place masked screenshot relative to the centered frame
    shot_layer = Image.new("RGBA", INSTAGRAM_SIZE, (0, 0, 0, 0))
    shot_layer.paste(
        shot_cropped,
        (offset_x + scaled_screen_bbox.left, offset_y + scaled_screen_bbox.top),
        mask=screen_mask,
    )
    canvas.alpha_composite(shot_layer)

    # Overlay the centered frame last
    frame_layer = Image.new("RGBA", INSTAGRAM_SIZE, (0, 0, 0, 0))
    frame_layer.paste(frame_scaled, (offset_x, offset_y))
    canvas.alpha_composite(frame_layer)

    return canvas.convert("RGB")


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bg = load_image(args.background)
    frame_img = load_image(args.frame)

    # Detect screen bbox from the ORIGINAL frame size
    screen_bbox_orig, screen_mask_orig = detect_screen_bbox_from_frame(frame_img)

    # Compute uniform scale to fit frame inside 1080x1080 without distortion
    scale = min(INSTAGRAM_SIZE[0] / frame_img.width, INSTAGRAM_SIZE[1] / frame_img.height)
    frame_scaled_size = (
        int(round(frame_img.width * scale)),
        int(round(frame_img.height * scale)),
    )
    scaled_screen = ScreenBBox(
        left=int(round(screen_bbox_orig.left * scale)),
        top=int(round(screen_bbox_orig.top * scale)),
        right=int(round(screen_bbox_orig.right * scale)),
        bottom=int(round(screen_bbox_orig.bottom * scale)),
    )
    # Scale the precise screen mask to match the scaled frame
    screen_mask_scaled = screen_mask_orig.resize(frame_scaled_size, Image.LANCZOS)

    # Prepare background tiles
    # If one screenshot: center the background crop
    # If multiple: create sequential left-aligned crops across the background width
    bg_for_instagram = resize_to_cover(bg, INSTAGRAM_SIZE)

    num_shots = len(args.screenshots)
    crops: List[Image.Image] = []
    if num_shots == 1:
        crops.append(center_crop(bg_for_instagram, INSTAGRAM_SIZE))
    else:
        # Use left-aligned start, then move right by 1080 for each image, keeping y at top
        # Ensure background is tall enough; it is by construction of resize_to_cover
        # But width must accommodate num_shots * 1080
        if bg_for_instagram.width < INSTAGRAM_SIZE[0] * num_shots:
            raise RuntimeError(
                f"Background too narrow for {num_shots} images after resizing. "
                f"Need at least {INSTAGRAM_SIZE[0] * num_shots}px width, got {bg_for_instagram.width}px."
            )
        for i in range(num_shots):
            left = i * INSTAGRAM_SIZE[0]
            crops.append(crop_from_left(bg_for_instagram, left, INSTAGRAM_SIZE))

    # Compose outputs
    outputs: List[Image.Image] = []
    for i, (s_path, bg_crop) in enumerate(zip(args.screenshots, crops)):
        screenshot = load_image(s_path)
        out = compose_frame(
            bg_crop,
            frame_img,
            scaled_screen,
            screenshot,
            frame_scaled_size,
            screen_mask_scaled,
        )
        outputs.append(out)

    # Save outputs
    if num_shots == 1:
        out_path = os.path.join(args.out_dir, f"{args.prefix}.jpg")
        outputs[0].save(out_path, quality=95)
        print(out_path)
    else:
        for i, out_img in enumerate(outputs, start=1):
            out_path = os.path.join(args.out_dir, f"{args.prefix}-{i:02d}.jpg")
            out_img.save(out_path, quality=95)
            print(out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


