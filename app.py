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


IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


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


def solid_swatch(color_hex: str, size: int = 88) -> np.ndarray:
    return np.full((size, size, 3), hex_to_bgr(color_hex), dtype=np.uint8)


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
    scale = 1.0
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

    if len(pixels_bgr) > 12000:
        rng = np.random.default_rng(42)
        choice = rng.choice(len(pixels_bgr), 12000, replace=False)
        pixels_bgr = pixels_bgr[choice]

    pixels_lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    cluster_count = max(1, min(max_colors + 2, len(pixels_lab)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pixels_lab, cluster_count, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    groups: list[dict[str, Any]] = []
    for index in range(cluster_count):
        cluster_pixels = pixels_lab[labels == index]
        if len(cluster_pixels) < 60:
            continue
        center_lab = centers[index]
        center_bgr = cv2.cvtColor(np.uint8([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
        groups.append(
            {
                "lab": center_lab.astype(np.float32),
                "bgr": np.array(center_bgr, dtype=np.uint8),
                "count": int(len(cluster_pixels)),
            }
        )

    groups.sort(key=lambda item: item["count"], reverse=True)

    merged: list[dict[str, Any]] = []
    for item in groups:
        matched = False
        for existing in merged:
            if float(np.linalg.norm(item["lab"] - existing["lab"])) < 14.0:
                total = existing["count"] + item["count"]
                existing["lab"] = (existing["lab"] * existing["count"] + item["lab"] * item["count"]) / total
                existing["bgr"] = cv2.cvtColor(np.uint8([[existing["lab"]]]), cv2.COLOR_LAB2BGR)[0, 0]
                existing["count"] = total
                matched = True
                break
        if not matched:
            merged.append(item)

    merged.sort(key=lambda item: item["count"], reverse=True)
    merged = merged[:max_colors]
    total_count = sum(item["count"] for item in merged) or 1

    colors: list[dict[str, Any]] = []
    for index, item in enumerate(merged, start=1):
        color_hex = bgr_to_hex(item["bgr"])
        colors.append(
            {
                "token": f"颜色{index}",
                "name": f"颜色{index}",
                "hex": color_hex,
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
                        rows.append(
                            {
                                "code": str(item.get("code", "")),
                                "hex": str(item.get("hex", "")),
                                "name": str(item.get("name", "")),
                            }
                        )
        else:
            sample = raw.splitlines()
            if sample:
                dialect = csv.Sniffer().sniff("\n".join(sample[:3]), delimiters=",\t;")
            else:
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
                    code = row[0].strip()
                    hex_value = row[1].strip()
                    name = row[2].strip() if len(row) > 2 else ""
                    rows.append({"code": code, "hex": hex_value, "name": name})

    for line in manual_text.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[,;\t|]", text)]
        if len(parts) >= 2:
            code = parts[0]
            hex_value = parts[1]
            name = parts[2] if len(parts) > 2 else ""
            rows.append({"code": code, "hex": hex_value, "name": name})

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
                "source": "code",
                "swatch": solid_swatch(hex_value),
            }
        )
    return entries


def recolor_region(base_bgr: np.ndarray, mask_u8: np.ndarray, target_hex: str) -> np.ndarray:
    base = ensure_bgr(base_bgr).astype(np.float32)
    mask = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    region = mask > 0.02
    if not np.any(region):
        return ensure_bgr(base_bgr)

    base_lab = cv2.cvtColor(ensure_bgr(base_bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(np.uint8([[list(hex_to_bgr(target_hex))]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    light = base_lab[:, :, 0]
    mean_light = float(light[region].mean())
    centered = light - mean_light

    new_lab = base_lab.copy()
    new_lab[:, :, 0] = np.clip(target_lab[0] * 0.60 + light * 0.40 + centered * 0.82, 0, 255)
    new_lab[:, :, 1] = np.clip(target_lab[1] * 0.94 + base_lab[:, :, 1] * 0.06, 0, 255)
    new_lab[:, :, 2] = np.clip(target_lab[2] * 0.94 + base_lab[:, :, 2] * 0.06, 0, 255)

    recolored = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    gray = cv2.cvtColor(ensure_bgr(base_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), 3.0)
    detail = gray - low
    recolored = np.clip(recolored + detail[:, :, None] * 0.16, 0, 255)

    alpha = cv2.GaussianBlur(mask, (0, 0), 1.1)[:, :, None]
    blended = base * (1.0 - alpha) + recolored * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_contact_sheet(results: list[dict[str, Any]], thumb_width: int = 320) -> np.ndarray:
    if not results:
        return np.full((400, 700, 3), 255, dtype=np.uint8)

    cards: list[np.ndarray] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for item in results:
        image = item["image"]
        scale = thumb_width / float(image.shape[1])
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        lines = [item["name"], item["summary"]]
        card_h = thumb.shape[0] + 88
        card = np.full((card_h, thumb_width, 3), 250, dtype=np.uint8)
        cv2.putText(card, lines[0], (16, 30), font, 0.8, (42, 51, 69), 2, cv2.LINE_AA)
        cv2.putText(card, lines[1][:40], (16, 60), font, 0.55, (92, 112, 140), 1, cv2.LINE_AA)
        card[88:, :] = thumb
        cards.append(card)

    gap = 24
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


def render_scheme(base_image: np.ndarray, mask_specs: list[dict[str, Any]], assignments: dict[str, dict[str, str]], token_map: dict[str, dict[str, Any]]) -> np.ndarray:
    output = ensure_bgr(base_image).copy()
    for spec in mask_specs:
        token = assignments.get(spec["name"], {}).get("token", "").strip()
        if not token:
            continue
        if re.fullmatch(r"#?[0-9A-Fa-f]{6}", token):
            target_hex = token if token.startswith("#") else "#" + token
        else:
            entry = token_map.get(normalize_token(token))
            if entry is None:
                continue
            target_hex = entry["hex"]
        output = recolor_region(output, spec["mask"], target_hex)
    return output


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
    st.subheader(title)
    if not entries:
        st.info("当前没有可用颜色。")
        return
    cols = st.columns(4)
    for index, entry in enumerate(entries):
        with cols[index % 4]:
            st.image(cv2.cvtColor(entry["swatch"], cv2.COLOR_BGR2RGB), use_container_width=False)
            caption = f"{entry['token']}  {entry['hex']}"
            if show_ratio:
                caption += f"  占比 {entry['ratio']:.0%}"
            st.caption(caption)
            if entry.get("name") and entry["name"] != entry["token"]:
                st.caption(entry["name"])


def collect_regions(base_bgr: np.ndarray, region_count: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    st.subheader("3. 上传拆分区域")
    st.caption("一个区域可以覆盖多个部位。比如两个部分同色，就把它们做进同一张蒙版里，一起上传。")
    for index in range(region_count):
        with st.expander(f"区域 {index + 1}", expanded=index < 3):
            name = st.text_input("区域名", value=f"区域{index + 1}", key=f"region_name_{index}")
            notes = st.text_input("区域说明", value="比如：外套主体 / 领口袖笼 / 腰头牙边", key=f"region_notes_{index}")
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
            col1, col2 = st.columns(2)
            with col1:
                st.image(mask_u8, caption="蒙版", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="覆盖预览", use_container_width=True)
            regions.append({"name": name.strip() or f"区域{index + 1}", "notes": notes.strip(), "mask": mask_u8})
    return regions


def collect_schemes(region_names: list[str], token_examples: list[str], scheme_count: int) -> list[dict[str, Any]]:
    schemes: list[dict[str, Any]] = []
    st.subheader("4. 配方案")
    st.caption("每个区域可以直接填“颜色1 / 颜色2”，也可以填你们自己的搭配编码；如果直接填 HEX，也支持。")
    example_text = "、".join(token_examples[:8]) if token_examples else "颜色1"
    for index in range(scheme_count):
        with st.expander(f"方案 {index + 1}", expanded=index < 2):
            name = st.text_input("方案名", value=f"修色{index + 1}", key=f"scheme_name_{index}")
            assignments: dict[str, dict[str, str]] = {}
            for region_name in region_names:
                left, right = st.columns([1, 1])
                with left:
                    quick_pick = st.selectbox(
                        f"{region_name} 快捷选择",
                        [""] + token_examples,
                        key=f"scheme_pick_{index}_{slugify(region_name)}",
                    )
                with right:
                    manual_token = st.text_input(
                        f"{region_name} 手动输入",
                        value="",
                        placeholder=f"如：颜色1 / A102 / #8AA379；可留空使用左侧快捷项。示例：{example_text}",
                        key=f"scheme_manual_{index}_{slugify(region_name)}",
                    )
                assignments[region_name] = {"token": manual_token.strip() or quick_pick}
            notes = st.text_area("备注", value="", placeholder="比如：所有黑色改绿；牙边改白；粉裤腰改黑。", key=f"scheme_notes_{index}")
            schemes.append({"name": name.strip() or f"修色{index + 1}", "assignments": assignments, "notes": notes})
    return schemes


def render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    st.subheader("5. 预览结果")
    cols = st.columns(2)
    for index, item in enumerate(results):
        with cols[index % 2]:
            st.image(cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB), caption=item["name"], use_container_width=True)
            st.caption(item["summary"])
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
    st.title("延展拆色修图台")
    st.write("先从目标参考图自动拆出衣服颜色，再让每个区域直接引用“颜色1 / 颜色2”或已有搭配编码，适合延展自己快速套方案。")

    with st.sidebar:
        st.header("配置")
        region_count = int(st.number_input("区域数量", min_value=1, max_value=12, value=4, step=1))
        scheme_count = int(st.number_input("方案数量", min_value=1, max_value=12, value=2, step=1))
        max_colors = int(st.number_input("参考图最多提取颜色数", min_value=1, max_value=10, value=4, step=1))

    st.subheader("1. 基础图与目标参考图")
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

    top_cols = st.columns(2)
    with top_cols[0]:
        st.image(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB), caption="基础图", use_container_width=True)
    with top_cols[1]:
        st.image(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB), caption="目标参考图", use_container_width=True)
        if reference_mask is not None:
            st.image(reference_mask, caption="参考图服装蒙版", use_container_width=True)

    reference_colors = extract_palette_from_reference(reference_bgr, reference_mask, max_colors=max_colors)
    render_color_entries("2. 参考图拆出的颜色", reference_colors, show_ratio=True)

    st.subheader("已有搭配编码库")
    st.caption("支持上传 CSV / TXT / JSON，或直接粘贴。推荐格式：编码,HEX,名称")
    code_file = st.file_uploader("上传搭配编码库", type=["csv", "txt", "json"], key="code_library")
    manual_codes = st.text_area(
        "或直接粘贴搭配编码",
        value="",
        height=140,
        placeholder="示例：\nA102,#859A72,军绿\nB305,#F7F7F2,米白",
    )
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

    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for scheme in schemes:
        for region_name, payload in scheme["assignments"].items():
            token = payload.get("token", "").strip()
            if not token:
                continue
            if re.fullmatch(r"#?[0-9A-Fa-f]{6}", token):
                continue
            if normalize_token(token) not in token_map:
                unresolved.append(f"{scheme['name']} / {region_name}: {token}")

    if unresolved:
        st.error("下面这些颜色标识无法识别，请改成“颜色1/颜色2”、已有搭配编码，或直接填 HEX：")
        for item in unresolved:
            st.write(f"- {item}")
        return

    for scheme in schemes:
        rendered = render_scheme(base_bgr, regions, scheme["assignments"], token_map)
        parts: list[str] = []
        for region_name, payload in scheme["assignments"].items():
            token = payload.get("token", "").strip()
            if not token:
                continue
            if re.fullmatch(r"#?[0-9A-Fa-f]{6}", token):
                hex_value = token if token.startswith("#") else "#" + token
            else:
                hex_value = token_map[normalize_token(token)]["hex"]
            parts.append(f"{region_name}={token}({hex_value})")
        results.append({"name": scheme["name"], "summary": " / ".join(parts), "image": rendered})

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
    }
    with bottom_cols[0]:
        st.download_button(
            "下载整包 ZIP",
            bundle,
            file_name=f"{slugify(job_name)}_extend_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with bottom_cols[1]:
        st.download_button(
            "下载联系表 JPG",
            image_to_jpg_bytes(contact_sheet),
            file_name=f"{slugify(job_name)}_contact_sheet.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
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
