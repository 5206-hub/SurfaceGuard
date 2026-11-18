"""Native-resolution prediction analysis and portable inspection reports.

The model always predicts at 200 x 200 pixels.  This module keeps that
inference contract separate from inspection outputs: label masks are resized
with nearest-neighbour interpolation, while probability maps use linear
interpolation and are re-normalised per pixel.
"""
from __future__ import annotations

import base64
import csv
import html
from datetime import datetime
from io import BytesIO, StringIO
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

from .vision import COLORS, NAMES, overlay


BACKGROUND_COLOR = (20, 25, 37)
MASK_COLORS = {0: BACKGROUND_COLOR, **COLORS}


def _target_size(shape: tuple[int, int]) -> tuple[int, int]:
    """Convert a NumPy (height, width) shape to an OpenCV (width, height)."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("Target dimensions must be positive")
    return width, height


def resize_labels_to_original(labels: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Return a class-ID mask at the original image resolution."""
    if labels.ndim != 2:
        raise ValueError("Expected a 2-D class-ID mask")
    unique = set(np.unique(labels).tolist())
    if not unique <= {0, 1, 2, 3}:
        raise ValueError("Class-ID mask must only contain labels 0..3")
    return cv2.resize(labels.astype(np.uint8), _target_size(original_shape), interpolation=cv2.INTER_NEAREST)


def resize_probabilities_to_original(probabilities: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Resize [4,H,W] softmax probabilities and restore a valid probability simplex."""
    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("Expected probabilities shaped [4,H,W]")
    target = _target_size(original_shape)
    resized = np.stack(
        [cv2.resize(channel.astype(np.float32), target, interpolation=cv2.INTER_LINEAR) for channel in probabilities]
    )
    resized = np.clip(resized, 0.0, None)
    normalizer = resized.sum(axis=0, keepdims=True)
    return np.divide(resized, normalizer, out=np.zeros_like(resized), where=normalizer > 0)


def colour_mask(labels: np.ndarray) -> np.ndarray:
    """Render every class ID, including background, with a fixed RGB colour."""
    if labels.ndim != 2:
        raise ValueError("Expected a 2-D class-ID mask")
    result = np.empty((*labels.shape, 3), dtype=np.uint8)
    for label, color in MASK_COLORS.items():
        result[labels == label] = color
    if np.any(~np.isin(labels, list(MASK_COLORS))):
        raise ValueError("Class-ID mask must only contain labels 0..3")
    return result


def probability_heatmap(probability: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Render a probability map using the matching defect colour and a dark base."""
    if probability.ndim != 2:
        raise ValueError("Expected a 2-D probability map")
    value = np.clip(probability.astype(np.float32), 0.0, 1.0)[..., None]
    base = np.asarray(BACKGROUND_COLOR, dtype=np.float32)
    tint = np.asarray(color, dtype=np.float32)
    return np.rint(base * (1.0 - value) + tint * value).astype(np.uint8)


def _class_name(class_id: int, class_names: Mapping[int, str] | None) -> str:
    return (class_names or NAMES).get(class_id, f"Label {class_id}")


def analyse_regions(
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
    class_names: Mapping[int, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Compute native-resolution per-class statistics and connected components."""
    if labels.ndim != 2:
        raise ValueError("Expected a 2-D class-ID mask")
    if probabilities is not None and probabilities.shape != (4, *labels.shape):
        raise ValueError("Probability-map dimensions must match the label mask")

    total_pixels = int(labels.size)
    class_summary: list[dict] = []
    regions: list[dict] = []
    for class_id in (1, 2, 3):
        binary = (labels == class_id).astype(np.uint8)
        components, component_ids, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        area = int(binary.sum())
        class_regions: list[dict] = []
        for component in range(1, components):
            x, y, width, height, pixels = (int(value) for value in stats[component])
            region = {
                "class_id": class_id,
                "defect_class": _class_name(class_id, class_names),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "defect_area_pixels": pixels,
                "percent_of_original_image": round(100.0 * pixels / total_pixels, 4),
            }
            if probabilities is not None:
                values = probabilities[class_id][component_ids == component]
                region["mean_class_probability"] = round(float(values.mean()), 4)
                region["max_class_probability"] = round(float(values.max()), 4)
            class_regions.append(region)
        regions.extend(class_regions)
        class_summary.append(
            {
                "class_id": class_id,
                "defect_class": _class_name(class_id, class_names),
                "region_count": len(class_regions),
                "area_pixels": area,
                "percent_of_original_image": round(100.0 * area / total_pixels, 4),
            }
        )
    return class_summary, regions


def draw_bounding_boxes(image: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Draw native-resolution region boxes on an RGB image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    result = image.copy()
    thickness = max(1, round(min(image.shape[:2]) / 240))
    for region in regions:
        color = MASK_COLORS[int(region["class_id"])]
        x, y = int(region["bbox_x"]), int(region["bbox_y"])
        width, height = int(region["bbox_width"]), int(region["bbox_height"])
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, thickness)
        label = str(region["defect_class"])
        baseline_y = max(12, y - 4)
        cv2.putText(result, label, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return result


def build_inspection(
    original_rgb: np.ndarray,
    labels_at_model_resolution: np.ndarray,
    probabilities_at_model_resolution: np.ndarray | None,
    image_filename: str,
    method: str,
    class_names: Mapping[int, str] | None = None,
) -> dict:
    """Build all native-resolution inspection images, statistics, and export data."""
    if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB original image")
    original_shape = original_rgb.shape[:2]
    labels = resize_labels_to_original(labels_at_model_resolution, original_shape)
    probabilities = (
        resize_probabilities_to_original(probabilities_at_model_resolution, original_shape)
        if probabilities_at_model_resolution is not None
        else None
    )
    class_summary, regions = analyse_regions(labels, probabilities, class_names)
    affected_pixels = int(np.count_nonzero(labels))
    total_pixels = int(labels.size)
    coloured_mask = colour_mask(labels)
    native_overlay = overlay(original_rgb, labels)
    annotated_overlay = draw_bounding_boxes(native_overlay, regions)
    return {
        "image_filename": image_filename,
        "method": method,
        "prediction_notice": "MODEL PREDICTION — not ground-truth annotation.",
        "inference_resolution": {
            "width": int(labels_at_model_resolution.shape[1]),
            "height": int(labels_at_model_resolution.shape[0]),
        },
        "visualization_resolution": {"width": int(original_shape[1]), "height": int(original_shape[0])},
        "affected_area_pixels": affected_pixels,
        "affected_area_percent": round(100.0 * affected_pixels / total_pixels, 4),
        "inspected_pixels": total_pixels,
        "class_summary": class_summary,
        "regions": regions,
        "original_rgb": original_rgb,
        "native_labels": labels,
        "native_probabilities": probabilities,
        "coloured_mask": coloured_mask,
        "native_overlay": native_overlay,
        "annotated_overlay": annotated_overlay,
    }


def export_payload(inspection: Mapping, severity: str | None = None) -> dict:
    """Return a JSON-safe version of an inspection, excluding image arrays."""
    payload = {
        key: value
        for key, value in inspection.items()
        if key not in {"original_rgb", "native_labels", "native_probabilities", "coloured_mask", "native_overlay", "annotated_overlay"}
    }
    payload["measurement"] = "Pixel-area estimate from a model prediction; not physical damage severity."
    if severity is not None:
        payload["severity_heuristic"] = severity
        payload["severity_notice"] = "Severity is a display heuristic based on affected image area and is not independently validated."
    return payload


def regions_csv(inspection: Mapping) -> bytes:
    """Create a UTF-8 region-level CSV from native-resolution results."""
    fields = [
        "image_filename",
        "defect_class",
        "defect_area_pixels",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "percent_of_original_image",
        "mean_class_probability",
        "max_class_probability",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for region in inspection["regions"]:
        writer.writerow({field: region.get(field, "") for field in fields} | {"image_filename": inspection["image_filename"]})
    return stream.getvalue().encode("utf-8")


def _image_data_uri(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _color_hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{value:02x}" for value in color)


def _legend_html(class_names: Mapping[int, str] | None = None) -> str:
    return "".join(
        f'<span class="legend"><i style="background:{_color_hex(MASK_COLORS[class_id])}"></i>{html.escape(_class_name(class_id, class_names))}</span>'
        for class_id in (0, 1, 2, 3)
    )


def _confidence_text(region: Mapping) -> str:
    value = region.get("mean_class_probability")
    return "" if value is None else f"{float(value):.4f}"


def inspection_html_report(
    inspection: Mapping,
    model_info: Mapping[str, str | float | int],
    severity: str | None = None,
    created_at: datetime | None = None,
    class_names: Mapping[int, str] | None = None,
) -> str:
    """Create a standalone HTML report with all images embedded as data URIs."""
    created_at = created_at or datetime.now().astimezone()
    safe_filename = html.escape(str(inspection["image_filename"]))
    class_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['defect_class']))}</td>"
        f"<td>{row['region_count']}</td><td>{row['area_pixels']:,}</td>"
        f"<td>{row['percent_of_original_image']:.4f}%</td></tr>"
        for row in inspection["class_summary"]
    )
    if inspection["regions"]:
        region_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['defect_class']))}</td>"
            f"<td>{row['defect_area_pixels']:,}</td>"
            f"<td>{row['bbox_x']}, {row['bbox_y']}, {row['bbox_width']}, {row['bbox_height']}</td>"
            f"<td>{row['percent_of_original_image']:.4f}%</td>"
            f"<td>{_confidence_text(row)}</td>"
            "</tr>"
            for row in inspection["regions"]
        )
    else:
        region_rows = '<tr><td colspan="5">No predicted defect regions.</td></tr>'
    model_rows = "".join(
        f"<tr><td>{html.escape(str(key).replace('_', ' ').title())}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in model_info.items()
    )
    severity_html = (
        f"<p><strong>Severity heuristic:</strong> {html.escape(severity)}. "
        "This area-based indicator is not an independently validated damage classification.</p>"
        if severity
        else ""
    )
    original_uri = _image_data_uri(inspection["original_rgb"])
    mask_uri = _image_data_uri(inspection["coloured_mask"])
    overlay_uri = _image_data_uri(inspection["annotated_overlay"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SurfaceGuard Inspection Report</title>
<style>
body {{ font-family: Arial, sans-serif; color: #172033; margin: 32px; }} h1 {{ color: #e55a1f; }}
.notice {{ background: #fff5ed; border-left: 4px solid #e55a1f; padding: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 12px; }} img {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }} th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; }} th {{ background: #f2f5f9; }}
.legend {{ display: inline-flex; align-items: center; margin: 4px 14px 4px 0; }} .legend i {{ width: 14px; height: 14px; margin-right: 6px; border: 1px solid #687385; }}
</style></head><body>
<h1>SurfaceGuard</h1><p><strong>Inspection date:</strong> {html.escape(created_at.isoformat(timespec='seconds'))}<br>
<strong>Image filename:</strong> {safe_filename}</p>
<div class="notice"><strong>MODEL PREDICTION.</strong> This report contains predicted segmentation results, not ground-truth annotations.</div>
<h2>Inspection summary</h2><p><strong>Affected area:</strong> {inspection['affected_area_percent']:.4f}% ({inspection['affected_area_pixels']:,} of {inspection['inspected_pixels']:,} pixels).<br>
<strong>Model inference resolution:</strong> {inspection['inference_resolution']['width']} × {inspection['inference_resolution']['height']} px.<br>
<strong>Native visualisation resolution:</strong> {inspection['visualization_resolution']['width']} × {inspection['visualization_resolution']['height']} px.</p>
{severity_html}
<h2>Class legend</h2>{_legend_html(class_names)}
<div class="grid"><section class="card"><h2>Original uploaded image</h2><img src="{original_uri}" alt="Original uploaded image"></section>
<section class="card"><h2>MODEL PREDICTION: segmentation mask</h2><img src="{mask_uri}" alt="Predicted segmentation mask"></section>
<section class="card"><h2>MODEL PREDICTION: overlay and regions</h2><img src="{overlay_uri}" alt="Predicted overlay with bounding boxes"></section></div>
<h2>Per-class predicted area</h2><table><thead><tr><th>Class</th><th>Regions</th><th>Area (px)</th><th>Image area</th></tr></thead><tbody>{class_rows}</tbody></table>
<h2>Predicted regions</h2><table><thead><tr><th>Class</th><th>Area (px)</th><th>Bounding box (x, y, w, h)</th><th>Image area</th><th>Mean class probability</th></tr></thead><tbody>{region_rows}</tbody></table>
<h2>Model and inference information</h2><table><tbody>{model_rows}</tbody></table>
</body></html>"""
