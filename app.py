from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any

import cv2
import numpy as np
import streamlit as st


def slugify(value: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def read_uploaded_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    if not data:
        return None
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    return image


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def image_to_jpg_bytes(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPG 编码失败")
    return encoded.tobytes()


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        return (127, 127, 127)
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    return (b, g, r)


def bgr_to_hex(color: tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = [int(x) for x in color]
    return f"#{r:02X}{g:02X}{b:02X}"


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > target_h or image.shape[1] > target_w else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


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


def extract_representative_bgr(image: np.ndarray) -> tuple[int, int, int]:
    bgr = ensure_bgr(image)
    max_side = max(bgr.shape[:2])
    if max_side > 600:
        scale = 600.0 / float(max_side)
        bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * scale)), int(round(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    valid = ((sat > 16) & (val > 24) & (val < 245)) | ((gray > 24) & (gray < 235))
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = resize_to_match(image[:, :, 3], bgr.shape[:2])
        valid &= alpha > 10
    if int(valid.sum()) < 200:
        valid = gray < 245
    if int(valid.sum()) < 50:
        h, w = bgr.shape[:2]
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.2), int(w * 0.8)
        crop = bgr[y0:y1, x0:x1]
        return tuple(int(x) for x in np.median(crop.reshape(-1, 3), axis=0))
    pixels = bgr[valid]
    return tuple(int(x) for x in np.median(pixels, axis=0))


def solid_swatch(color_hex: str, size: int = 96) -> np.ndarray:
    swatch = np.full((size, size, 3), hex_to_bgr(color_hex), dtype=np.uint8)
    return swatch


def recolor_region(base_bgr: np.ndarray, mask_u8: np.ndarray, target_hex: str) -> np.ndarray:
    base = base_bgr.astype(np.float32)
    mask = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    region = mask > 0.02
    if not np.any(region):
        return base_bgr.copy()

    base_lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(np.uint8([[list(hex_to_bgr(target_hex))]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    light = base_lab[:, :, 0]
    mean_light = float(light[region].mean())
    centered = light - mean_light

    new_lab = base_lab.copy()
    new_lab[:, :, 0] = np.clip(target_lab[0] * 0.62 + light * 0.38 + centered * 0.85, 0, 255)
    new_lab[:, :, 1] = np.clip(target_lab[1] * 0.94 + base_lab[:, :, 1] * 0.06, 0, 255)
    new_lab[:, :, 2] = np.clip(target_lab[2] * 0.94 + base_lab[:, :, 2] * 0.06, 0, 255)

    recolored = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), 3.2)
    detail = gray - low
    recolored = np.clip(recolored + detail[:, :, None] * 0.18, 0, 255)

    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2)[:, :, None]
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
        card_h = thumb.shape[0] + 76
        card = np.full((card_h, thumb_width, 3), 250, dtype=np.uint8)
        cv2.putText(card, item["name"], (16, 28), font, 0.8, (42, 51, 69), 2, cv2.LINE_AA)
        cv2.putText(card, item["summary"], (16, 56), font, 0.62, (92, 112, 140), 2, cv2.LINE_AA)
        card[76:, :] = thumb
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


def build_export_zip(
    job_name: str,
    base_image: np.ndarray,
    board_image: np.ndarray | None,
    masks: list[dict[str, Any]],
    palettes: list[dict[str, Any]],
    schemes: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> bytes:
    manifest = {
        "job_name": job_name,
        "palettes": palettes,
        "regions": [{"name": item["name"], "color_role": item["color_role"]} for item in masks],
        "schemes": schemes,
        "result_files": [f"previews/{slugify(item['name'])}.jpg" for item in results],
    }
    contact_sheet = build_contact_sheet(results)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        lines = [
            f"任务名: {job_name}",
            "",
            "配色库:",
        ]
        for palette in palettes:
            lines.append(f"- {palette['name']}: {palette['hex']}")
        lines.append("")
        lines.append("方案:")
        for scheme in schemes:
            lines.append(f"- {scheme['name']}")
            for region_name, palette_name in scheme["assignments"].items():
                lines.append(f"  {region_name}: {palette_name}")
            if scheme["notes"].strip():
                lines.append(f"  备注: {scheme['notes'].strip()}")
        zf.writestr("README.txt", "\n".join(lines))
        zf.writestr("source/base.jpg", image_to_jpg_bytes(base_image))
        if board_image is not None:
            zf.writestr("source/requirement_board.jpg", image_to_jpg_bytes(ensure_bgr(board_image)))
        for index, mask in enumerate(masks, start=1):
            mask_png = cv2.imencode(".png", mask["mask"])[1].tobytes()
            zf.writestr(f"masks/{index:02d}_{slugify(mask['name'])}.png", mask_png)
        for palette in palettes:
            zf.writestr(f"palette/{slugify(palette['name'])}.jpg", image_to_jpg_bytes(solid_swatch(palette["hex"])))
            if palette.get("sample_image") is not None:
                zf.writestr(f"palette/{slugify(palette['name'])}_reference.jpg", image_to_jpg_bytes(palette["sample_image"]))
        for item in results:
            zf.writestr(f"previews/{slugify(item['name'])}.jpg", image_to_jpg_bytes(item["image"]))
        zf.writestr("previews/contact_sheet.jpg", image_to_jpg_bytes(contact_sheet))
    memory.seek(0)
    return memory.getvalue()


def render_scheme(base_image: np.ndarray, mask_specs: list[dict[str, Any]], assignments: dict[str, str], palette_map: dict[str, str]) -> np.ndarray:
    output = base_image.copy()
    for spec in mask_specs:
        palette_name = assignments.get(spec["name"], "")
        target_hex = palette_map.get(palette_name)
        if not target_hex:
            continue
        output = recolor_region(output, spec["mask"], target_hex)
    return output


def collect_palette_entries(palette_count: int) -> list[dict[str, Any]]:
    defaults = [
        ("黑色", "#1F1F1F"),
        ("白色", "#F7F7F2"),
        ("军绿", "#859A72"),
        ("粉色", "#F55AD8"),
        ("湖蓝", "#39D7E6"),
        ("黄色", "#FFF15A"),
    ]
    palettes: list[dict[str, Any]] = []
    st.subheader("1. 配色库")
    st.caption("先把常用颜色建成配色库，后面每个方案直接选，不用重复输。可以手动给 HEX，也可以上传参考图自动取色。")
    for index in range(palette_count):
        default_name, default_hex = defaults[index] if index < len(defaults) else (f"颜色{index + 1}", "#808080")
        with st.expander(f"颜色 {index + 1}", expanded=index < 3):
            name = st.text_input("颜色名", value=default_name, key=f"palette_name_{index}")
            sample_file = st.file_uploader("可选：上传该颜色的参考图", type=["jpg", "jpeg", "png"], key=f"palette_sample_{index}")
            sampled_hex = None
            sample_preview = None
            if sample_file is not None:
                sample_preview = ensure_bgr(read_uploaded_image(sample_file))
                sampled_hex = bgr_to_hex(extract_representative_bgr(sample_preview))
                st.image(cv2.cvtColor(sample_preview, cv2.COLOR_BGR2RGB), caption=f"自动取色 {sampled_hex}", use_container_width=True)
            color_value = st.color_picker("颜色值", value=sampled_hex or default_hex, key=f"palette_hex_{index}")
            palettes.append({
                "name": name.strip() or f"颜色{index + 1}",
                "hex": color_value,
                "sample_image": sample_preview,
            })
    return palettes


def collect_regions(base_bgr: np.ndarray, region_count: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    st.subheader("2. 改色区域")
    st.caption("每个区域对应一张蒙版。适合同版型、明确部位的延展图，比如外套主体、领口、腰头、牙边、裤脚。")
    for index in range(region_count):
        with st.expander(f"区域 {index + 1}", expanded=index < 3):
            name = st.text_input("区域名", value=f"区域{index + 1}", key=f"region_name_{index}")
            role = st.text_input("区域说明", value="比如：外套主体 / 领口包边 / 牙边", key=f"region_role_{index}")
            mask_file = st.file_uploader("上传该区域蒙版", type=["jpg", "jpeg", "png"], key=f"region_mask_{index}")
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
                st.image(mask_u8, caption="蒙版预览", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="区域覆盖预览", use_container_width=True)
            regions.append({
                "name": name.strip() or f"区域{index + 1}",
                "color_role": role.strip(),
                "mask": mask_u8,
            })
    return regions


def collect_schemes(region_names: list[str], palette_names: list[str], scheme_count: int) -> list[dict[str, Any]]:
    schemes: list[dict[str, Any]] = []
    st.subheader("3. 方案列表")
    st.caption("一个方案就是一张最终图。非常适合像你截图那样，一次把“修色十一、修色十二”这类方案都配好。")
    for index in range(scheme_count):
        with st.expander(f"方案 {index + 1}", expanded=index < 2):
            name = st.text_input("方案名", value=f"修色{index + 1}", key=f"scheme_name_{index}")
            assignments: dict[str, str] = {}
            for region_name in region_names:
                assignments[region_name] = st.selectbox(
                    f"{region_name} 用哪个颜色",
                    palette_names,
                    key=f"assign_{index}_{slugify(region_name)}",
                )
            notes = st.text_area("备注", value="", placeholder="比如：所有黑色改绿；粉蓝改白黑；牙边参考左图。", key=f"scheme_notes_{index}")
            schemes.append({
                "name": name.strip() or f"修色{index + 1}",
                "assignments": assignments,
                "notes": notes,
            })
    return schemes


def render_result_gallery(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    st.subheader("4. 结果预览")
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
                key=f"download_scheme_{index}",
            )


def main() -> None:
    st.set_page_config(page_title="延展批量修图台", layout="wide")
    st.title("延展批量修图台")
    st.write("适合同版型、多部位、多方案的快速改色。思路是先把区域蒙版和常用颜色建好，再让延展自己批量出预览图，不用每次都重新写长需求。")

    with st.sidebar:
        st.header("配置")
        region_count = int(st.number_input("区域数量", min_value=1, max_value=12, value=4, step=1))
        palette_count = int(st.number_input("配色数量", min_value=1, max_value=16, value=6, step=1))
        scheme_count = int(st.number_input("方案数量", min_value=1, max_value=12, value=2, step=1))

    st.subheader("基础素材")
    job_name = st.text_input("任务名", value="延展修图任务")
    base_file = st.file_uploader("上传基础图", type=["jpg", "jpeg", "png"], key="base_file")
    board_file = st.file_uploader("可选：上传需求拼图 / 标注图", type=["jpg", "jpeg", "png"], key="board_file")

    if base_file is None:
        st.info("先上传基础图，我再继续往下展示配置区。")
        return

    base_image_raw = read_uploaded_image(base_file)
    if base_image_raw is None:
        st.error("基础图读取失败。")
        return
    base_bgr = ensure_bgr(base_image_raw)
    board_image = read_uploaded_image(board_file) if board_file is not None else None

    top_cols = st.columns(2)
    with top_cols[0]:
        st.image(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB), caption="基础图", use_container_width=True)
    with top_cols[1]:
        if board_image is not None:
            st.image(cv2.cvtColor(ensure_bgr(board_image), cv2.COLOR_BGR2RGB), caption="需求拼图 / 标注图", use_container_width=True)
        else:
            st.info("可以上传你截图这种需求拼图，方便延展直接对照操作。")

    palettes = collect_palette_entries(palette_count)
    regions = collect_regions(base_bgr, region_count)
    if not regions:
        st.warning("至少需要 1 个有效区域蒙版。")
        return

    palette_names = [item["name"] for item in palettes]
    schemes = collect_schemes([item["name"] for item in regions], palette_names, scheme_count)

    if not st.button("生成所有方案预览", type="primary", use_container_width=True):
        return

    palette_map = {item["name"]: item["hex"] for item in palettes}
    results: list[dict[str, Any]] = []
    for scheme in schemes:
        rendered = render_scheme(base_bgr, regions, scheme["assignments"], palette_map)
        summary = " / ".join(f"{region}: {scheme['assignments'][region]}" for region in scheme["assignments"])
        results.append({
            "name": scheme["name"],
            "summary": summary,
            "image": rendered,
        })

    render_result_gallery(results)

    bundle = build_export_zip(job_name, base_bgr, board_image, regions, palettes, schemes, results)
    contact_sheet = build_contact_sheet(results)

    bottom_cols = st.columns(3)
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
            json.dumps(
                {
                    "job_name": job_name,
                    "palettes": [{"name": p["name"], "hex": p["hex"]} for p in palettes],
                    "regions": [{"name": r["name"], "color_role": r["color_role"]} for r in regions],
                    "schemes": schemes,
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name=f"{slugify(job_name)}_config.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
