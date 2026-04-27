from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import streamlit as st

try:
    from skimage.color import deltaE_ciede2000
except Exception:  # pragma: no cover
    deltaE_ciede2000 = None


IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
COLOR_TABLE_PATH = Path(__file__).with_name("internal_color_hex_table_full_rows_by_fabric_code.csv")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2rem;
          max-width: 1320px;
        }
        div[data-testid="stImage"] img {
          border-radius: 14px;
        }
        .compact-card {
          border: 1px solid #eadfce;
          background: #fffdf9;
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 8px 24px rgba(77, 54, 26, 0.06);
          margin-bottom: 14px;
        }
        .compact-title {
          font-size: 14px;
          font-weight: 700;
          color: #334155;
          margin-bottom: 8px;
        }
        .mini-note {
          font-size: 12px;
          color: #64748b;
        }
        .delta-pill {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          background: #f5efe4;
          color: #7c5d32;
          font-size: 12px;
          font-weight: 600;
          margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def slugify(value: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def normalize_token(value: str) -> str:
    return value.strip().lower()


def read_uploaded_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    if not data:
        return None
    array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_UNCHANGED)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[:2] == target_hw:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > target_h or image.shape[1] > target_w else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def thumbnail_for_ui(image: np.ndarray, max_w: int = 300, max_h: int = 320) -> np.ndarray:
    image = ensure_bgr(image)
    h, w = image.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 1.0:
        return image
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def image_to_jpg_bytes(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(".jpg", ensure_bgr(image_bgr), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPG 编码失败")
    return encoded.tobytes()


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"非法 HEX: {hex_color}")
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    return (b, g, r)


def bgr_to_hex(color: tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = [int(round(float(x))) for x in color]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_lab_color(color_bgr: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[list(color_bgr)]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def lab_to_bgr_color(lab_color: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab_color, dtype=np.float32).reshape(1, 1, 3)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]


def solid_swatch(color_hex: str, size: int = 76) -> np.ndarray:
    return np.full((size, size, 3), hex_to_bgr(color_hex), dtype=np.uint8)


def make_color_entry(
    token: str,
    hex_value: str,
    *,
    name: str | None = None,
    source: str = "code",
    swatch_size: int = 76,
    **extra: Any,
) -> dict[str, Any]:
    if not hex_value.startswith("#"):
        hex_value = "#" + hex_value
    entry = {
        "token": token,
        "name": name or token,
        "hex": hex_value.upper(),
        "lab": bgr_to_lab_color(hex_to_bgr(hex_value)),
        "source": source,
        "swatch": solid_swatch(hex_value, size=swatch_size),
    }
    entry.update(extra)
    return entry


def default_code_library() -> list[dict[str, Any]]:
    presets = [
        ("BK", "#1F1F1F", "Black"),
        ("WT", "#F7F7F2", "White"),
        ("PK", "#F62FA7", "Pink"),
        ("LP", "#FBAAE0", "Light Pink"),
        ("GN", "#859A72", "Green"),
        ("BL", "#39D7E6", "Blue"),
    ]
    entries: list[dict[str, Any]] = []
    for code, hex_value, name in presets:
        entries.append(make_color_entry(code, hex_value, name=name, source="fallback"))
    return entries


def load_internal_code_library(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []

    best_rows: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            product_code = row[1].strip()
            fabric_code = row[2].strip()
            color_code = row[3].strip()
            color_hex = row[5].strip().upper()
            status = row[8].strip().upper()
            if not product_code or not fabric_code or not color_code:
                continue
            if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", color_hex):
                continue
            if status and status != "OK":
                continue
            if not color_hex.startswith("#"):
                color_hex = "#" + color_hex
            try:
                confidence = float(row[6]) if row[6].strip() else 0.0
            except ValueError:
                confidence = 0.0
            current = best_rows.get(product_code)
            if current is None or confidence > current["confidence"]:
                best_rows[product_code] = {
                    "token": product_code,
                    "name": product_code,
                    "hex": color_hex,
                    "source": "fabric_table",
                    "fabric_code": fabric_code,
                    "color_code": color_code,
                    "reference_url": row[4].strip(),
                    "confidence": confidence,
                    "size": row[7].strip(),
                }

    entries: list[dict[str, Any]] = []
    for _, payload in sorted(best_rows.items(), key=lambda item: item[0]):
        entries.append(
            make_color_entry(
                payload["token"],
                payload["hex"],
                name=payload["name"],
                source=payload["source"],
                fabric_code=payload["fabric_code"],
                color_code=payload["color_code"],
                reference_url=payload["reference_url"],
                confidence=payload["confidence"],
                size=payload["size"],
            )
        )
    return entries


def filter_code_library(entries: list[dict[str, Any]], fabric_code: str) -> list[dict[str, Any]]:
    fabric = fabric_code.strip().upper()
    return [item for item in entries if item.get("fabric_code", "").upper() == fabric]


def build_token_map(reference_colors: list[dict[str, Any]], code_library: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    token_map: dict[str, dict[str, Any]] = {}

    def register(token: str, entry: dict[str, Any]) -> None:
        key = normalize_token(token)
        if key and key not in token_map:
            token_map[key] = entry

    for entry in reference_colors:
        register(entry["token"], entry)

    color_alias_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in code_library:
        register(entry["token"], entry)
        color_code = entry.get("color_code", "").strip()
        if color_code:
            color_alias_groups.setdefault(normalize_token(color_code), []).append(entry)

    for alias, matches in color_alias_groups.items():
        if len(matches) == 1 and alias not in token_map:
            token_map[alias] = matches[0]
    return token_map


def build_option_labels(reference_colors: list[dict[str, Any]], code_library: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in reference_colors:
        ratio = f"{entry.get('ratio', 0):.0%}" if entry.get("ratio") is not None else ""
        labels[entry["token"]] = f"{entry['token']} · {entry['hex']} {ratio}".strip()
    for entry in code_library:
        labels[entry["token"]] = f"{entry['token']} · {entry.get('color_code', '')} · {entry['hex']}".strip()
    return labels


def refine_entry_from_reference(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("source") != "fabric_table":
        return entry
    reference_url = entry.get("reference_url", "").strip()
    if not reference_url:
        return entry
    remote_image = load_image_from_url(reference_url)
    if remote_image is None:
        return entry
    palette = extract_palette_from_reference(remote_image, None, max_colors=1)
    if not palette:
        return entry
    chosen = palette[0]
    refined = make_color_entry(
        entry["token"],
        chosen["hex"],
        name=entry.get("name", entry["token"]),
        source="fabric_table_refined",
        fabric_code=entry.get("fabric_code", ""),
        color_code=entry.get("color_code", ""),
        reference_url=reference_url,
        confidence=entry.get("confidence", 0.0),
        size=entry.get("size", ""),
        refined_from_url=True,
    )
    return refined


def build_nearby_color_entries(entry: dict[str, Any], count: int = 3) -> list[dict[str, Any]]:
    adjustments = [
        ("原色", 0.0, 0.0, 0.0),
        ("偏亮", 8.0, 0.0, 0.0),
        ("偏深", -8.0, 0.0, 0.0),
        ("偏暖", 3.0, 4.0, 4.0),
        ("偏冷", 3.0, -4.0, -4.0),
    ]
    base_lab = np.asarray(entry["lab"], dtype=np.float32).copy()
    results: list[dict[str, Any]] = []
    seen_hex: set[str] = set()
    for label, delta_l, delta_a, delta_b in adjustments:
        lab = base_lab.copy()
        lab[0] = np.clip(lab[0] + delta_l, 36, 232)
        lab[1] = np.clip(lab[1] + delta_a, 18, 238)
        lab[2] = np.clip(lab[2] + delta_b, 18, 238)
        color_hex = bgr_to_hex(lab_to_bgr_color(lab))
        if color_hex in seen_hex:
            continue
        seen_hex.add(color_hex)
        variant_token = entry["token"] if label == "原色" else f"{entry['token']}·{label}"
        variant = make_color_entry(
            variant_token,
            color_hex,
            name=entry.get("name", entry["token"]),
            source=f"{entry.get('source', 'code')}_nearby",
            fabric_code=entry.get("fabric_code", ""),
            color_code=entry.get("color_code", ""),
            variant_label=label,
            base_token=entry["token"],
            display=f"{entry['token']} {label}",
        )
        results.append(variant)
        if len(results) >= count:
            break
    return results


def delta_e_lab(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    a = np.asarray(lab_a, dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(lab_b, dtype=np.float32).reshape(1, 1, 3)
    if deltaE_ciede2000 is not None:
        return float(deltaE_ciede2000(a, b)[0, 0])
    return float(np.linalg.norm(a - b))


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_url_image_bytes(url: str) -> bytes | None:
    if not url:
        return None
    try:
        with urlopen(url, timeout=10) as response:
            return response.read()
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def load_image_from_url(url: str) -> np.ndarray | None:
    data = fetch_url_image_bytes(url)
    if not data:
        return None
    array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def preprocess_mask(mask_image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    mask_image = resize_to_match(mask_image, target_hw)
    if mask_image.ndim == 3 and mask_image.shape[2] == 4:
        alpha = mask_image[:, :, 3]
        if int(alpha.max()) > 0 and float(alpha.mean()) < 250:
            return cv2.GaussianBlur(alpha, (0, 0), 1.2)
    bgr = ensure_bgr(mask_image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (((gray < 245) & ((sat > 14) | (val < 245))) * 255).astype(np.uint8)
    if int(mask.mean()) < 3:
        mask = ((gray > 12) * 255).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.4)
    return mask


def extract_reference_valid_mask(reference_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    valid = ((sat > 18) & (val > 18) & (val < 248)) | ((gray > 12) & (gray < 235))
    valid = valid.astype(np.uint8) * 255
    valid = cv2.medianBlur(valid, 5)
    return valid


def build_reference_garment_mask(reference_bgr: np.ndarray) -> np.ndarray:
    image = ensure_bgr(reference_bgr)
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    focus = np.zeros((h, w), dtype=np.uint8)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    y0, y1 = int(h * 0.14), int(h * 0.95)
    focus[y0:y1, x0:x1] = 255

    near_white_bg = ((val > 236) & (sat < 28)) | (gray > 245)
    skin = (cr > 132) & (cr < 176) & (cb > 76) & (cb < 128) & (y > 55)
    cloth_like = ((sat > 26) | (val < 160) | (gray < 205))
    candidate = (focus > 0) & (~near_white_bg) & (~skin) & cloth_like

    raw_mask = (candidate.astype(np.uint8) * 255)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    keep = np.zeros_like(raw_mask)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((raw_mask > 0).astype(np.uint8), connectivity=8)
    ranked: list[tuple[int, int]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < max(120, int(h * w * 0.0035)):
            continue
        cx, cy = centroids[component_id]
        if cy < h * 0.10 or cy > h * 0.97:
            continue
        if cx < w * 0.04 or cx > w * 0.96:
            continue
        ranked.append((area, component_id))
    ranked.sort(reverse=True)
    for _, component_id in ranked[:4]:
        keep[labels == component_id] = 255

    mask = cv2.GaussianBlur(keep, (0, 0), 1.4)

    if int(mask.mean()) < 6:
        fallback = ((focus > 0) & (~near_white_bg) & ((sat > 20) | (val < 175))).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(fallback, (0, 0), 1.2)
    return mask


def extract_palette_from_reference(reference_bgr: np.ndarray, mask_u8: np.ndarray | None, max_colors: int) -> list[dict[str, Any]]:
    image = ensure_bgr(reference_bgr)
    max_side = max(image.shape[:2])
    if max_side > 720:
        scale = 720.0 / float(max_side)
        image = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    if mask_u8 is not None:
        mask_small = resize_to_match(mask_u8, image.shape[:2])
        valid = mask_small > 24
    else:
        garment_mask = build_reference_garment_mask(image)
        valid = garment_mask > 24
        if int(valid.sum()) < 100:
            valid = extract_reference_valid_mask(image) > 0

    pixels_bgr = image[valid]
    if pixels_bgr.size == 0:
        return []

    if len(pixels_bgr) > 14000:
        rng = np.random.default_rng(42)
        choice = rng.choice(len(pixels_bgr), 14000, replace=False)
        pixels_bgr = pixels_bgr[choice]

    pixels_lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    cluster_count = max(1, min(max_colors + 3, len(pixels_lab)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 36, 0.8)
    _, labels, centers = cv2.kmeans(pixels_lab, cluster_count, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    groups: list[dict[str, Any]] = []
    for index in range(cluster_count):
        cluster_pixels = pixels_lab[labels == index]
        if len(cluster_pixels) < 70:
            continue
        center_lab = centers[index]
        cluster_bgr = cv2.cvtColor(np.uint8([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
        ycrcb = cv2.cvtColor(np.uint8([[cluster_bgr]]), cv2.COLOR_BGR2YCrCb)[0, 0]
        hsv = cv2.cvtColor(np.uint8([[cluster_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        gray = int(round(float(np.mean(cluster_bgr))))
        is_skin_like = (
            133 <= int(ycrcb[1]) <= 176
            and 76 <= int(ycrcb[2]) <= 128
            and int(ycrcb[0]) > 55
            and int(hsv[1]) < 110
            and gray > 70
        )
        if is_skin_like:
            continue
        groups.append({"lab": center_lab.astype(np.float32), "count": int(len(cluster_pixels))})

    groups.sort(key=lambda item: item["count"], reverse=True)
    merged: list[dict[str, Any]] = []
    for item in groups:
        merged_to_existing = False
        for existing in merged:
            if delta_e_lab(item["lab"], existing["lab"]) < 9.0:
                total = existing["count"] + item["count"]
                existing["lab"] = (existing["lab"] * existing["count"] + item["lab"] * item["count"]) / total
                existing["count"] = total
                merged_to_existing = True
                break
        if not merged_to_existing:
            merged.append(item)

    merged.sort(key=lambda item: item["count"], reverse=True)
    merged = merged[:max_colors]
    total_count = sum(item["count"] for item in merged) or 1

    colors: list[dict[str, Any]] = []
    for index, item in enumerate(merged, start=1):
        bgr = cv2.cvtColor(np.uint8([[item["lab"]]]), cv2.COLOR_LAB2BGR)[0, 0]
        color_hex = bgr_to_hex(bgr)
        colors.append(
            {
                "token": f"颜色{index}",
                "name": f"颜色{index}",
                "hex": color_hex,
                "lab": item["lab"],
                "ratio": round(item["count"] / total_count, 4),
                "source": "reference",
                "swatch": solid_swatch(color_hex),
            }
        )
    return colors


def parse_code_library(uploaded_file, manual_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []

    if uploaded_file is not None:
        raw = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")
        suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
        if suffix == "json":
            payload = json.loads(raw)
            if isinstance(payload, dict):
                for code, value in payload.items():
                    if isinstance(value, dict):
                        rows.append({"code": str(code), "hex": str(value.get("hex", "")), "name": str(value.get("name", ""))})
                    else:
                        rows.append({"code": str(code), "hex": str(value), "name": ""})
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        rows.append({"code": str(item.get("code", "")), "hex": str(item.get("hex", "")), "name": str(item.get("name", ""))})
        else:
            sample = raw.splitlines()
            dialect = csv.excel
            if sample:
                try:
                    dialect = csv.Sniffer().sniff("\n".join(sample[:3]), delimiters=",\t;")
                except csv.Error:
                    dialect = csv.excel
            reader = csv.reader(io.StringIO(raw), dialect)
            parsed_rows = list(reader)
            if parsed_rows:
                header = [cell.strip().lower() for cell in parsed_rows[0]]
                has_header = "code" in header or "hex" in header
                start_index = 1 if has_header else 0
                for row in parsed_rows[start_index:]:
                    if len(row) < 2:
                        continue
                    rows.append({"code": row[0].strip(), "hex": row[1].strip(), "name": row[2].strip() if len(row) > 2 else ""})

    for line in manual_text.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[,;\t|]", text)]
        if len(parts) >= 2:
            rows.append({"code": parts[0], "hex": parts[1], "name": parts[2] if len(parts) > 2 else ""})

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        code = row["code"].strip()
        hex_value = row["hex"].strip()
        if not code or not re.fullmatch(r"#?[0-9A-Fa-f]{6}", hex_value):
            continue
        if not hex_value.startswith("#"):
            hex_value = "#" + hex_value
        key = normalize_token(code)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "token": code,
                "name": row["name"].strip() or code,
                "hex": hex_value.upper(),
                "lab": bgr_to_lab_color(hex_to_bgr(hex_value)),
                "source": "code",
                "swatch": solid_swatch(hex_value),
            }
        )
    return entries


def masked_lab_mean(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(ensure_bgr(image_bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
    region = mask_u8 > 24
    if not np.any(region):
        return np.array([0.0, 128.0, 128.0], dtype=np.float32)
    return lab[region].mean(axis=0).astype(np.float32)


def resolve_token_to_color(token: str, token_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    text = token.strip()
    if not text:
        return None
    if re.fullmatch(r"#?[0-9A-Fa-f]{6}", text):
        hex_value = text if text.startswith("#") else "#" + text
        return {
            "token": text,
            "name": text,
            "hex": hex_value.upper(),
            "lab": bgr_to_lab_color(hex_to_bgr(hex_value)),
            "source": "manual",
            "swatch": solid_swatch(hex_value),
        }
    entry = token_map.get(normalize_token(text))
    if entry is None:
        return None
    return refine_entry_from_reference(entry)


def resolve_assignment_to_color(payload: dict[str, str], token_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    direct_hex = payload.get("hex", "").strip()
    if re.fullmatch(r"#?[0-9A-Fa-f]{6}", direct_hex):
        return make_color_entry(
            payload.get("token", direct_hex) or direct_hex,
            direct_hex,
            name=payload.get("name", payload.get("token", direct_hex) or direct_hex),
            source="assignment",
            variant_label=payload.get("variant_label", "原色"),
            base_token=payload.get("base_token", payload.get("token", direct_hex)),
            display=payload.get("display", payload.get("token", direct_hex) or direct_hex),
        )
    return resolve_token_to_color(payload.get("token", ""), token_map)


def recolor_region(base_bgr: np.ndarray, mask_u8: np.ndarray, target_hex: str) -> tuple[np.ndarray, float]:
    original = ensure_bgr(base_bgr)
    mask = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    region = mask > 0.02
    if not np.any(region):
        return original.copy(), 0.0

    base_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = bgr_to_lab_color(hex_to_bgr(target_hex))

    light = base_lab[:, :, 0]
    mean_light = float(light[region].mean())
    centered = light - mean_light

    new_lab = base_lab.copy()
    new_lab[:, :, 0] = np.clip(target_lab[0] * 0.70 + light * 0.30 + centered * 0.88, 0, 255)
    new_lab[:, :, 1] = np.clip(target_lab[1] * 0.985 + base_lab[:, :, 1] * 0.015, 0, 255)
    new_lab[:, :, 2] = np.clip(target_lab[2] * 0.985 + base_lab[:, :, 2] * 0.015, 0, 255)

    recolored_lab = new_lab.copy()
    for _ in range(3):
        mean_lab = recolored_lab[region].mean(axis=0)
        delta = target_lab - mean_lab
        recolored_lab[:, :, 0] = np.clip(recolored_lab[:, :, 0] + mask * delta[0] * 0.78, 0, 255)
        recolored_lab[:, :, 1] = np.clip(recolored_lab[:, :, 1] + mask * delta[1] * 0.92, 0, 255)
        recolored_lab[:, :, 2] = np.clip(recolored_lab[:, :, 2] + mask * delta[2] * 0.92, 0, 255)

    recolored = cv2.cvtColor(recolored_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), 2.6)
    detail = gray - low
    recolored = np.clip(recolored + detail[:, :, None] * 0.12, 0, 255)

    alpha = cv2.GaussianBlur(mask, (0, 0), 1.0)[:, :, None]
    blended = original.astype(np.float32) * (1.0 - alpha) + recolored * alpha
    result = np.clip(blended, 0, 255).astype(np.uint8)
    region_mean = masked_lab_mean(result, mask_u8)
    delta_e = delta_e_lab(region_mean, target_lab)
    return result, delta_e


def render_scheme(base_image: np.ndarray, mask_specs: list[dict[str, Any]], assignments: dict[str, dict[str, str]], token_map: dict[str, dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    output = ensure_bgr(base_image).copy()
    region_scores: dict[str, float] = {}
    for spec in mask_specs:
        payload = assignments.get(spec["name"], {})
        if not payload:
            continue
        color_entry = resolve_assignment_to_color(payload, token_map)
        if color_entry is None:
            continue
        output, delta_e = recolor_region(output, spec["mask"], color_entry["hex"])
        region_scores[spec["name"]] = delta_e
    return output, region_scores


def build_contact_sheet(results: list[dict[str, Any]], thumb_width: int = 300) -> np.ndarray:
    if not results:
        return np.full((400, 700, 3), 255, dtype=np.uint8)

    cards: list[np.ndarray] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for item in results:
        image = item["image"]
        scale = thumb_width / float(image.shape[1])
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        card_h = thumb.shape[0] + 88
        card = np.full((card_h, thumb_width, 3), 250, dtype=np.uint8)
        cv2.putText(card, item["name"], (14, 28), font, 0.72, (42, 51, 69), 2, cv2.LINE_AA)
        cv2.putText(card, f"DeltaE {item['delta_e']:.2f}", (14, 58), font, 0.58, (124, 93, 50), 2, cv2.LINE_AA)
        card[88:, :] = thumb
        cards.append(card)

    gap = 20
    cols = 2 if len(cards) > 1 else 1
    rows = int(np.ceil(len(cards) / float(cols)))
    card_w = max(card.shape[1] for card in cards)
    card_h = max(card.shape[0] for card in cards)
    canvas = np.full((rows * card_h + (rows + 1) * gap, cols * card_w + (cols + 1) * gap, 3), 255, dtype=np.uint8)
    for idx, card in enumerate(cards):
        row = idx // cols
        col = idx % cols
        y = gap + row * (card_h + gap)
        x = gap + col * (card_w + gap)
        canvas[y:y + card.shape[0], x:x + card.shape[1]] = card
    return canvas


def build_export_zip(
    job_name: str,
    base_image: np.ndarray,
    reference_image: np.ndarray,
    reference_colors: list[dict[str, Any]],
    code_library: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    schemes: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> bytes:
    manifest = {
        "job_name": job_name,
        "reference_colors": [{"token": item["token"], "hex": item["hex"], "ratio": item.get("ratio", 0)} for item in reference_colors],
        "code_library": [{"token": item["token"], "hex": item["hex"], "name": item["name"]} for item in code_library],
        "regions": [{"name": item["name"], "notes": item["notes"]} for item in regions],
        "schemes": schemes,
        "results": [{"name": item["name"], "delta_e": item["delta_e"], "region_delta_e": item["region_delta_e"]} for item in results],
    }

    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("source/base.jpg", image_to_jpg_bytes(base_image))
        zf.writestr("source/reference.jpg", image_to_jpg_bytes(reference_image))
        for item in reference_colors:
            zf.writestr(f"reference_colors/{slugify(item['token'])}_{item['hex'].replace('#', '')}.jpg", image_to_jpg_bytes(item["swatch"]))
        for item in code_library:
            zf.writestr(f"code_library/{slugify(item['token'])}_{item['hex'].replace('#', '')}.jpg", image_to_jpg_bytes(item["swatch"]))
        for index, item in enumerate(regions, start=1):
            mask_png = cv2.imencode(".png", item["mask"])[1].tobytes()
            zf.writestr(f"masks/{index:02d}_{slugify(item['name'])}.png", mask_png)
        for item in results:
            zf.writestr(f"previews/{slugify(item['name'])}.jpg", image_to_jpg_bytes(item["image"]))
        zf.writestr("previews/contact_sheet.jpg", image_to_jpg_bytes(build_contact_sheet(results)))
    memory.seek(0)
    return memory.getvalue()


def render_color_entries(title: str, entries: list[dict[str, Any]], show_ratio: bool = False, max_display: int = 24) -> None:
    st.markdown(f'<div class="compact-card"><div class="compact-title">{title}</div></div>', unsafe_allow_html=True)
    if not entries:
        st.info("当前没有可用颜色。")
        return
    if len(entries) > max_display:
        st.caption(f"共 {len(entries)} 个候选，仅展示前 {max_display} 个；下拉框里可搜索全量成品编码。")
    cols = st.columns(8)
    for index, entry in enumerate(entries[:max_display]):
        with cols[index % 8]:
            st.image(cv2.cvtColor(entry["swatch"], cv2.COLOR_BGR2RGB), width=52)
            label = f"{entry['token']} {entry['hex']}"
            if show_ratio:
                label += f"  {entry['ratio']:.0%}"
            st.caption(label)
            meta_parts = []
            if entry.get("color_code"):
                meta_parts.append(entry["color_code"])
            if entry.get("fabric_code"):
                meta_parts.append(entry["fabric_code"])
            if meta_parts:
                st.caption(" / ".join(meta_parts))


def render_code_library_overview(title: str, entries: list[dict[str, Any]], max_display: int = 18) -> None:
    st.markdown(f'<div class="compact-card"><div class="compact-title">{title}</div></div>', unsafe_allow_html=True)
    if not entries:
        st.info("当前没有可用成品编码。")
        return
    st.caption(f"共 {len(entries)} 个候选。页面只展示前 {max_display} 个示例；实际请在方案下拉框里直接搜索成品编码。")
    tokens = [entry["token"] for entry in entries[:max_display]]
    cols = st.columns(6)
    for index, token in enumerate(tokens):
        with cols[index % 6]:
            st.code(token, language=None)


def merge_region_masks(mask_images: list[np.ndarray], target_hw: tuple[int, int]) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for mask_image in mask_images:
        processed = preprocess_mask(mask_image, target_hw)
        if merged is None:
            merged = processed.astype(np.uint8)
        else:
            merged = np.maximum(merged, processed.astype(np.uint8))
    if merged is None:
        return None
    merged = cv2.medianBlur(merged, 5)
    return cv2.GaussianBlur(merged, (0, 0), 1.2)


def collect_regions(base_bgr: np.ndarray, region_count: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    st.markdown('<div class="compact-card"><div class="compact-title">3. 上传拆分区域</div><div class="mini-note">同一区域可以上传多张蒙版，系统会自动合并为一个区域。</div></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for index in range(region_count):
        with cols[index % 3]:
            st.markdown(f"**区域 {index + 1}**")
            mask_files = st.file_uploader(
                "上传蒙版",
                type=IMAGE_TYPES,
                key=f"region_mask_{index}",
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if not mask_files:
                st.caption("等待上传")
                continue
            mask_images: list[np.ndarray] = []
            for mask_file in mask_files:
                mask_raw = read_uploaded_image(mask_file)
                if mask_raw is not None:
                    mask_images.append(mask_raw)
            if not mask_images:
                st.warning("蒙版读取失败")
                continue
            mask_u8 = merge_region_masks(mask_images, base_bgr.shape[:2])
            if mask_u8 is None:
                st.warning("蒙版合并失败")
                continue
            preview = cv2.bitwise_and(base_bgr, base_bgr, mask=(mask_u8 > 20).astype(np.uint8) * 255)
            st.caption(f"已上传 {len(mask_images)} 张")
            sub_cols = st.columns(2)
            with sub_cols[0]:
                st.image(mask_u8, caption="蒙版", width=96)
            with sub_cols[1]:
                st.image(cv2.cvtColor(thumbnail_for_ui(preview, 96, 128), cv2.COLOR_BGR2RGB), caption="预览", width=96)
            regions.append({"name": f"区域{index + 1}", "notes": "", "mask": mask_u8})
    return regions


def collect_schemes(
    region_names: list[str],
    token_examples: list[str],
    option_labels: dict[str, str],
    token_map: dict[str, dict[str, Any]],
    scheme_count: int,
    nearby_count: int,
) -> list[dict[str, Any]]:
    schemes: list[dict[str, Any]] = []
    st.markdown('<div class="compact-card"><div class="compact-title">4. 配方案</div><div class="mini-note">先选参考色或成品编码，再手动挑“原色 / 偏亮 / 偏深 / 偏暖 / 偏冷”后重新生成。多个区域可以自由组合。</div></div>', unsafe_allow_html=True)
    cols = st.columns(min(max(scheme_count, 1), 4))
    for index in range(scheme_count):
        with cols[index % len(cols)]:
            st.markdown(f"**方案 {index + 1}**")
            assignments: dict[str, dict[str, str]] = {}
            for region_name in region_names:
                quick_pick = st.selectbox(
                    region_name,
                    token_examples,
                    key=f"scheme_pick_{index}_{slugify(region_name)}",
                    index=min(index, max(len(token_examples) - 1, 0)) if token_examples else 0,
                    format_func=lambda token: option_labels.get(token, token),
                )
                selected_entry = resolve_token_to_color(quick_pick, token_map)
                if selected_entry is not None:
                    previews = build_nearby_color_entries(selected_entry, count=max(nearby_count + 1, 1))
                    preview_labels = [preview.get("variant_label", "原色") for preview in previews]
                    preview_map = {preview.get("variant_label", "原色"): preview for preview in previews}
                    selected_label = st.radio(
                        f"{region_name}_variant",
                        preview_labels,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"scheme_variant_{index}_{slugify(region_name)}",
                    )
                    preview_cols = st.columns(len(previews))
                    for preview_index, preview in enumerate(previews):
                        with preview_cols[preview_index]:
                            st.image(cv2.cvtColor(preview["swatch"], cv2.COLOR_BGR2RGB), width=26)
                            st.caption(preview.get("variant_label", "原色"))
                    chosen_preview = preview_map[selected_label]
                    assignments[region_name] = {
                        "token": chosen_preview.get("base_token", selected_entry["token"]),
                        "hex": chosen_preview["hex"],
                        "name": chosen_preview["name"],
                        "variant_label": chosen_preview.get("variant_label", "原色"),
                        "base_token": chosen_preview.get("base_token", selected_entry["token"]),
                        "display": chosen_preview.get("display", chosen_preview["token"]),
                    }
                else:
                    assignments[region_name] = {"token": quick_pick}
            schemes.append({"name": f"图片{index + 1}", "assignments": assignments, "notes": "", "display_order": index})
    return schemes


def render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    st.markdown('<div class="compact-card"><div class="compact-title">5. 预览结果</div><div class="mini-note">每个方案都展示总 DeltaE 和分区域 DeltaE，方便快速判断哪一版更接近目标色。</div></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for index, item in enumerate(results):
        with cols[index % 4]:
            st.image(cv2.cvtColor(thumbnail_for_ui(item["image"], 150, 205), cv2.COLOR_BGR2RGB), caption=item["name"], use_container_width=False)
            st.markdown(f'<div class="delta-pill">总 DeltaE {item["delta_e"]:.2f}</div>', unsafe_allow_html=True)
            st.caption(item["summary"])
            if item["region_delta_e"]:
                parts = [f"{name}: {value:.2f}" for name, value in item["region_delta_e"].items()]
                st.caption("分区域 DeltaE: " + " / ".join(parts))
            st.download_button(
                f"下载 {item['name']} JPG",
                image_to_jpg_bytes(item["image"]),
                file_name=f"{slugify(item['name'])}.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key=f"download_result_{index}",
            )


def main() -> None:
    st.set_page_config(page_title="延展拆色修图台", layout="wide")
    inject_css()
    st.title("延展拆色修图台")
    st.caption("从目标参考图自动拆出衣服颜色，方案可按成品编码或参考色选色；默认基础面料编号 JL1，可切换。")

    full_code_library = load_internal_code_library(COLOR_TABLE_PATH)
    fabric_options = sorted({item.get("fabric_code", "") for item in full_code_library if item.get("fabric_code")})
    default_fabric_index = fabric_options.index("JL1") if "JL1" in fabric_options else 0

    with st.sidebar:
        st.header("配置")
        if fabric_options:
            selected_fabric_code = st.selectbox("基础面料编号", fabric_options, index=default_fabric_index)
        else:
            selected_fabric_code = "JL1"
            st.caption("未读取到本地颜色表，先使用内置样例编码。")
        region_count = int(st.number_input("区域数量", min_value=1, max_value=12, value=4, step=1))
        scheme_count = int(st.number_input("方案数量", min_value=1, max_value=12, value=2, step=1))
        max_colors = int(st.number_input("参考图最多提取颜色数", min_value=1, max_value=10, value=4, step=1))
        nearby_count = int(st.number_input("每个方案自动附带临近色数量", min_value=0, max_value=4, value=2, step=1))

    st.markdown('<div class="compact-card"><div class="compact-title">1. 基础图与目标参考图</div></div>', unsafe_allow_html=True)
    job_name = f"延展拆色任务_{selected_fabric_code}"
    upload_cols = st.columns(2)
    with upload_cols[0]:
        base_file = st.file_uploader("上传基础图", type=IMAGE_TYPES, key="base_image")
    with upload_cols[1]:
        reference_file = st.file_uploader("上传目标参考图", type=IMAGE_TYPES, key="reference_image")

    if base_file is None or reference_file is None:
        st.info("先上传基础图和目标参考图。")
        return

    base_raw = read_uploaded_image(base_file)
    reference_raw = read_uploaded_image(reference_file)
    if base_raw is None or reference_raw is None:
        st.error("图片读取失败，请重新上传。")
        return

    base_bgr = ensure_bgr(base_raw)
    reference_bgr = ensure_bgr(reference_raw)
    reference_mask = None

    top_cols = st.columns(2)
    with top_cols[0]:
        st.image(cv2.cvtColor(thumbnail_for_ui(base_bgr, 180, 240), cv2.COLOR_BGR2RGB), caption="基础图", use_container_width=False)
    with top_cols[1]:
        st.image(cv2.cvtColor(thumbnail_for_ui(reference_bgr, 180, 240), cv2.COLOR_BGR2RGB), caption="目标参考图", use_container_width=False)

    reference_colors = extract_palette_from_reference(reference_bgr, reference_mask, max_colors=max_colors)
    render_color_entries("2. 参考图拆出的衣服颜色", reference_colors, show_ratio=True, max_display=8)

    filtered_code_library = filter_code_library(full_code_library, selected_fabric_code) if full_code_library else []
    if not filtered_code_library:
        filtered_code_library = default_code_library()
        st.warning(f"颜色表里暂时没有基础面料编号 {selected_fabric_code} 的记录，已回退到内置样例编码。")

    token_map = build_token_map(reference_colors, filtered_code_library)
    option_labels = build_option_labels(reference_colors, filtered_code_library)
    token_examples = [item["token"] for item in reference_colors] + [item["token"] for item in filtered_code_library]

    regions = collect_regions(base_bgr, region_count)
    if not regions:
        st.warning("至少需要 1 个有效区域蒙版。")
        return

    schemes = collect_schemes(
        [item["name"] for item in regions],
        token_examples,
        option_labels,
        token_map,
        scheme_count,
        nearby_count,
    )

    if not st.button("生成方案预览", type="primary", use_container_width=True):
        return

    unresolved: list[str] = []
    for scheme in schemes:
        for region_name, payload in scheme["assignments"].items():
            token = payload.get("token", "").strip()
            if not token and not payload.get("hex", "").strip():
                continue
            if resolve_assignment_to_color(payload, token_map) is None:
                unresolved.append(f"{scheme['name']} / {region_name}: {token}")

    if unresolved:
        st.error("下面这些颜色标识无法识别，请改成“颜色1/颜色2”、已有搭配编码，或直接填 HEX：")
        for item in unresolved:
            st.write(f"- {item}")
        return

    results: list[dict[str, Any]] = []
    for scheme in schemes:
        rendered, region_delta_e = render_scheme(base_bgr, regions, scheme["assignments"], token_map)
        parts: list[str] = []
        for region_name, payload in scheme["assignments"].items():
            color_entry = resolve_assignment_to_color(payload, token_map)
            if color_entry is None:
                continue
            variant_label = payload.get("variant_label", "原色")
            base_token = payload.get("base_token", payload.get("token", ""))
            parts.append(f"{region_name}={base_token}[{variant_label}]({color_entry['hex']})")
        total_delta_e = float(np.mean(list(region_delta_e.values()))) if region_delta_e else 0.0
        results.append(
            {
                "name": scheme["name"],
                "display_order": scheme.get("display_order", 0),
                "summary": " / ".join(parts),
                "image": rendered,
                "delta_e": total_delta_e,
                "region_delta_e": region_delta_e,
            }
        )

    results.sort(key=lambda item: item.get("display_order", 0))
    render_results(results)

    bottom_cols = st.columns(1)
    bundle = build_export_zip(job_name, base_bgr, reference_bgr, reference_colors, filtered_code_library, regions, schemes, results)
    with bottom_cols[0]:
        st.download_button("下载整包 ZIP", bundle, file_name=f"{slugify(job_name)}_extend_bundle.zip", mime="application/zip", use_container_width=True)


if __name__ == "__main__":
    main()
