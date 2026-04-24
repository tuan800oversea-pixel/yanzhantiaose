from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from typing import Any

import cv2
import numpy as np
import streamlit as st

try:
    from skimage.color import deltaE_ciede2000
except Exception:  # pragma: no cover
    deltaE_ciede2000 = None


IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


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


def solid_swatch(color_hex: str, size: int = 76) -> np.ndarray:
    return np.full((size, size, 3), hex_to_bgr(color_hex), dtype=np.uint8)


def delta_e_lab(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    a = np.asarray(lab_a, dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(lab_b, dtype=np.float32).reshape(1, 1, 3)
    if deltaE_ciede2000 is not None:
        return float(deltaE_ciede2000(a, b)[0, 0])
    return float(np.linalg.norm(a - b))


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
    return token_map.get(normalize_token(text))


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
        token = assignments.get(spec["name"], {}).get("token", "").strip()
        if not token:
            continue
        color_entry = resolve_token_to_color(token, token_map)
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


def render_color_entries(title: str, entries: list[dict[str, Any]], show_ratio: bool = False) -> None:
    st.markdown(f'<div class="compact-card"><div class="compact-title">{title}</div></div>', unsafe_allow_html=True)
    if not entries:
        st.info("当前没有可用颜色。")
        return
    cols = st.columns(6)
    for index, entry in enumerate(entries):
        with cols[index % 6]:
            st.image(cv2.cvtColor(entry["swatch"], cv2.COLOR_BGR2RGB), width=76)
            label = f"{entry['token']} {entry['hex']}"
            if show_ratio:
                label += f"  {entry['ratio']:.0%}"
            st.caption(label)
            if entry.get("name") and entry["name"] != entry["token"]:
                st.caption(entry["name"])


def collect_regions(base_bgr: np.ndarray, region_count: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    st.markdown('<div class="compact-card"><div class="compact-title">3. 上传拆分区域</div><div class="mini-note">一个区域可以覆盖多个部位。如果两个部位同色，就放在同一张蒙版里。</div></div>', unsafe_allow_html=True)
    for index in range(region_count):
        with st.expander(f"区域 {index + 1}", expanded=index < 2):
            name = st.text_input("区域名", value=f"区域{index + 1}", key=f"region_name_{index}")
            notes = st.text_input("区域说明", value="比如：上衣主体 / 腰头包边 / 牙边", key=f"region_notes_{index}")
            mask_file = st.file_uploader("上传该区域蒙版", type=IMAGE_TYPES, key=f"region_mask_{index}")
            if mask_file is None:
                st.warning("还没上传蒙版。")
                continue
            mask_raw = read_uploaded_image(mask_file)
            if mask_raw is None:
                st.warning("蒙版读取失败。")
                continue
            mask_u8 = preprocess_mask(mask_raw, base_bgr.shape[:2])
            preview = cv2.bitwise_and(base_bgr, base_bgr, mask=(mask_u8 > 20).astype(np.uint8) * 255)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(mask_u8, caption="蒙版", width=220)
            with col2:
                st.image(cv2.cvtColor(thumbnail_for_ui(preview, 220, 220), cv2.COLOR_BGR2RGB), caption="覆盖预览", width=220)
            regions.append({"name": name.strip() or f"区域{index + 1}", "notes": notes.strip(), "mask": mask_u8})
    return regions


def collect_schemes(region_names: list[str], token_examples: list[str], scheme_count: int) -> list[dict[str, Any]]:
    schemes: list[dict[str, Any]] = []
    st.markdown('<div class="compact-card"><div class="compact-title">4. 配方案</div><div class="mini-note">每个区域可以输入“颜色1 / 颜色2”、已有搭配编码，或直接输入 HEX。</div></div>', unsafe_allow_html=True)
    example_text = "、".join(token_examples[:8]) if token_examples else "颜色1"
    for index in range(scheme_count):
        with st.expander(f"方案 {index + 1}", expanded=index < 2):
            name = st.text_input("方案名", value=f"修色{index + 1}", key=f"scheme_name_{index}")
            assignments: dict[str, dict[str, str]] = {}
            for region_name in region_names:
                left, right = st.columns([1, 1.2])
                with left:
                    quick_pick = st.selectbox(f"{region_name} 快速选", [""] + token_examples, key=f"scheme_pick_{index}_{slugify(region_name)}")
                with right:
                    manual_token = st.text_input(
                        f"{region_name} 输入标识",
                        value="",
                        placeholder=f"如：颜色1 / A102 / #8AA379。示例：{example_text}",
                        key=f"scheme_manual_{index}_{slugify(region_name)}",
                    )
                assignments[region_name] = {"token": manual_token.strip() or quick_pick}
            notes = st.text_area("备注", value="", placeholder="比如：牙边参考颜色2；主体用 A102。", key=f"scheme_notes_{index}", height=90)
            schemes.append({"name": name.strip() or f"修色{index + 1}", "assignments": assignments, "notes": notes})
    return schemes


def render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    st.markdown('<div class="compact-card"><div class="compact-title">5. 预览结果</div><div class="mini-note">每个方案都展示总 DeltaE 和分区域 DeltaE，方便快速判断哪一版更接近目标色。</div></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for index, item in enumerate(results):
        with cols[index % 2]:
            st.image(cv2.cvtColor(thumbnail_for_ui(item["image"], 430, 500), cv2.COLOR_BGR2RGB), caption=item["name"], use_container_width=False)
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
    st.caption("从目标参考图自动拆出颜色，区域方案直接引用“颜色1 / 搭配编码 / HEX”，并对每个方案计算色差。")

    with st.sidebar:
        st.header("配置")
        region_count = int(st.number_input("区域数量", min_value=1, max_value=12, value=4, step=1))
        scheme_count = int(st.number_input("方案数量", min_value=1, max_value=12, value=2, step=1))
        max_colors = int(st.number_input("参考图最多提取颜色数", min_value=1, max_value=10, value=4, step=1))

    st.markdown('<div class="compact-card"><div class="compact-title">1. 基础图与目标参考图</div></div>', unsafe_allow_html=True)
    job_name = st.text_input("任务名", value="延展拆色任务")
    base_file = st.file_uploader("上传基础图", type=IMAGE_TYPES, key="base_image")
    reference_file = st.file_uploader("上传目标参考图", type=IMAGE_TYPES, key="reference_image")
    reference_mask_file = st.file_uploader("可选：上传目标参考图的服装蒙版", type=IMAGE_TYPES, key="reference_mask")

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
    if reference_mask_file is not None:
        mask_raw = read_uploaded_image(reference_mask_file)
        if mask_raw is not None:
            reference_mask = preprocess_mask(mask_raw, reference_bgr.shape[:2])

    top_cols = st.columns([1, 1, 0.8])
    with top_cols[0]:
        st.image(cv2.cvtColor(thumbnail_for_ui(base_bgr, 300, 360), cv2.COLOR_BGR2RGB), caption="基础图", use_container_width=False)
    with top_cols[1]:
        st.image(cv2.cvtColor(thumbnail_for_ui(reference_bgr, 300, 360), cv2.COLOR_BGR2RGB), caption="目标参考图", use_container_width=False)
    with top_cols[2]:
        if reference_mask is not None:
            st.image(reference_mask, caption="参考图蒙版", width=240)
        else:
            st.info("如果想只从衣服区域拆色，建议上传一张参考图服装蒙版。")

    reference_colors = extract_palette_from_reference(reference_bgr, reference_mask, max_colors=max_colors)
    render_color_entries("2. 参考图拆出的颜色", reference_colors, show_ratio=True)

    st.markdown('<div class="compact-card"><div class="compact-title">已有搭配编码库</div><div class="mini-note">支持上传 CSV / TXT / JSON，或直接粘贴。推荐格式：编码,HEX,名称</div></div>', unsafe_allow_html=True)
    code_file = st.file_uploader("上传搭配编码库", type=["csv", "txt", "json"], key="code_library")
    manual_codes = st.text_area("或直接粘贴搭配编码", value="", height=120, placeholder="示例：\nA102,#859A72,军绿\nB305,#F7F7F2,米白")
    code_library = parse_code_library(code_file, manual_codes)
    render_color_entries("已有搭配编码", code_library, show_ratio=False)

    token_map = {normalize_token(item["token"]): item for item in reference_colors + code_library}
    token_examples = [item["token"] for item in reference_colors] + [item["token"] for item in code_library]

    regions = collect_regions(base_bgr, region_count)
    if not regions:
        st.warning("至少需要 1 个有效区域蒙版。")
        return

    schemes = collect_schemes([item["name"] for item in regions], token_examples, scheme_count)

    if not st.button("生成方案预览", type="primary", use_container_width=True):
        return

    unresolved: list[str] = []
    for scheme in schemes:
        for region_name, payload in scheme["assignments"].items():
            token = payload.get("token", "").strip()
            if not token:
                continue
            if resolve_token_to_color(token, token_map) is None:
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
            token = payload.get("token", "").strip()
            if not token:
                continue
            color_entry = resolve_token_to_color(token, token_map)
            if color_entry is None:
                continue
            parts.append(f"{region_name}={token}({color_entry['hex']})")
        total_delta_e = float(np.mean(list(region_delta_e.values()))) if region_delta_e else 0.0
        results.append(
            {
                "name": scheme["name"],
                "summary": " / ".join(parts),
                "image": rendered,
                "delta_e": total_delta_e,
                "region_delta_e": region_delta_e,
            }
        )

    results.sort(key=lambda item: item["delta_e"])
    render_results(results)

    bottom_cols = st.columns(3)
    bundle = build_export_zip(job_name, base_bgr, reference_bgr, reference_colors, code_library, regions, schemes, results)
    contact_sheet = build_contact_sheet(results)
    config_payload = {
        "job_name": job_name,
        "reference_colors": [{"token": item["token"], "hex": item["hex"], "ratio": item["ratio"]} for item in reference_colors],
        "code_library": [{"token": item["token"], "hex": item["hex"], "name": item["name"]} for item in code_library],
        "regions": [{"name": item["name"], "notes": item["notes"]} for item in regions],
        "schemes": schemes,
        "results": [{"name": item["name"], "delta_e": item["delta_e"], "region_delta_e": item["region_delta_e"]} for item in results],
    }
    with bottom_cols[0]:
        st.download_button("下载整包 ZIP", bundle, file_name=f"{slugify(job_name)}_extend_bundle.zip", mime="application/zip", use_container_width=True)
    with bottom_cols[1]:
        st.download_button("下载联系表 JPG", image_to_jpg_bytes(contact_sheet), file_name=f"{slugify(job_name)}_contact_sheet.jpg", mime="image/jpeg", use_container_width=True)
    with bottom_cols[2]:
        st.download_button(
            "下载配置 JSON",
            json.dumps(config_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{slugify(job_name)}_config.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
