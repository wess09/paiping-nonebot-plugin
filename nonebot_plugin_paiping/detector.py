from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    is_screen_photo: bool
    score: float
    threshold: float
    features: dict[str, float]
    reasons: list[str]


def detect_screen_photo(
    image: np.ndarray,
    *,
    threshold: float = 0.62,
    min_size: int = 160,
) -> DetectionResult:
    """Detect whether an image is a camera photo of a display.

    The detector deliberately uses several weak OpenCV signals instead of one
    brittle rule. A real screen photo often has a mix of moire peaks, chroma
    sub-pixel noise, rolling shutter bands, uneven display illumination, and
    sometimes a screen-shaped border. Ordinary photos may trigger one signal,
    but they rarely trigger the same combination.
    """

    if image is None or image.size == 0:
        return _result(False, 0.0, threshold, {}, ["图片为空"])

    resized = _resize_for_analysis(image)
    height, width = resized.shape[:2]
    if min(height, width) < min_size:
        return _result(False, 0.0, threshold, {}, ["图片尺寸过小"])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    frequency = _frequency_peak_score(gray)
    local_frequency = _local_frequency_score(gray)
    chroma = _chroma_subpixel_score(resized, gray)
    banding = _banding_score(gray)
    rectangle = _screen_rectangle_score(gray)
    illumination = _illumination_score(gray)
    softness = _softness_score(gray)
    camera_artifact = _camera_artifact_score(gray, illumination, softness)
    display_content = _display_content_score(gray)
    white_clip = _white_clip_ratio(gray)
    screen_aspect = _screen_aspect_score(width, height)
    overexposed_display = _overexposed_display_score(
        white_clip=white_clip,
        screen_aspect=screen_aspect,
        illumination=illumination,
        softness=softness,
        camera_artifact=camera_artifact,
        display_content=display_content,
        banding=banding,
    )
    screenshot_penalty = _screenshot_like_penalty(gray, frequency, chroma, banding)

    features = {
        "frequency": frequency,
        "local_frequency": local_frequency,
        "chroma": chroma,
        "banding": banding,
        "rectangle": rectangle,
        "illumination": illumination,
        "softness": softness,
        "camera_artifact": camera_artifact,
        "display_content": display_content,
        "white_clip": white_clip,
        "screen_aspect": screen_aspect,
        "overexposed_display": overexposed_display,
        "screenshot_penalty": screenshot_penalty,
    }

    periodic = max(frequency, local_frequency, banding)
    capture_combo = _screen_capture_combo_score(
        frequency=frequency,
        local_frequency=local_frequency,
        banding=banding,
        illumination=illumination,
        softness=softness,
        camera_artifact=camera_artifact,
        display_content=display_content,
        white_clip=white_clip,
        overexposed_display=overexposed_display,
    )

    score = (
        0.27 * frequency
        + 0.18 * local_frequency
        + 0.23 * chroma
        + 0.13 * banding
        + 0.11 * rectangle
        + 0.05 * illumination
        + 0.03 * softness
    )
    if periodic > 0.55 and chroma > 0.42:
        score += 0.08
    if max(frequency, local_frequency) > 0.48 and banding > 0.45:
        score += 0.05
    if rectangle > 0.65 and (periodic > 0.35 or chroma > 0.35):
        score += 0.04
    if camera_artifact > 0.55 and display_content > 0.35 and softness > 0.30:
        score += 0.06
    score -= 0.14 * screenshot_penalty
    score = _clip01(max(score, capture_combo))

    reasons = _build_reasons(features)
    if not reasons:
        reasons.append("未发现稳定的拍屏特征组合")

    return _result(score >= threshold, score, threshold, features, reasons)


def detect_screen_photo_bytes(
    image_bytes: bytes,
    *,
    threshold: float = 0.62,
    min_size: int = 160,
) -> DetectionResult:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return _result(False, 0.0, threshold, {}, ["图片解码失败"])
    return detect_screen_photo(image, threshold=threshold, min_size=min_size)


def _result(
    is_screen_photo: bool,
    score: float,
    threshold: float,
    features: dict[str, float],
    reasons: list[str],
) -> DetectionResult:
    return DetectionResult(
        is_screen_photo=is_screen_photo,
        score=round(float(score), 4),
        threshold=round(float(threshold), 4),
        features={key: round(float(value), 4) for key, value in features.items()},
        reasons=reasons,
    )


def _resize_for_analysis(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _frequency_peak_score(gray: np.ndarray) -> float:
    work = _resize_gray(gray, max_side=640).astype(np.float32)
    height, width = work.shape
    blur_sigma = max(1.2, min(height, width) / 80.0)
    high_pass = work - cv2.GaussianBlur(work, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    high_pass -= float(np.mean(high_pass))

    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(high_pass * window))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    yy, xx = np.indices((height, width))
    cy, cx = height // 2, width // 2
    dy = yy - cy
    dx = xx - cx
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = max(1.0, float(min(height, width)) / 2.0)

    ring_mask = (radius > 0.10 * max_radius) & (radius < 0.88 * max_radius)
    ring_values = magnitude[ring_mask]
    if ring_values.size < 64:
        return 0.0

    base = float(np.percentile(ring_values, 70))
    spread = float(np.percentile(ring_values, 92) - np.percentile(ring_values, 35)) + 1e-6
    peak = float(np.percentile(ring_values, 99.82))
    peak_z = (peak - base) / spread

    local_max = magnitude == cv2.dilate(magnitude, np.ones((7, 7), np.uint8))
    candidates = magnitude[ring_mask & local_max]
    if candidates.size:
        top_count = min(12, candidates.size)
        top_mean = float(np.mean(np.partition(candidates, -top_count)[-top_count:]))
        peak_cluster = (top_mean - base) / spread
    else:
        peak_cluster = 0.0

    axis_width = max(2, int(min(height, width) * 0.008))
    axis_mask = ring_mask & ((np.abs(dx) <= axis_width) | (np.abs(dy) <= axis_width))
    diagonal_mask = ring_mask & (np.abs(np.abs(dx) - np.abs(dy)) <= axis_width)
    axis_ratio = _safe_mean(magnitude[axis_mask]) / (_safe_mean(magnitude[ring_mask]) + 1e-6)
    diagonal_ratio = _safe_mean(magnitude[diagonal_mask]) / (_safe_mean(magnitude[ring_mask]) + 1e-6)

    peak_score = _normalize(peak_z, 1.8, 5.4)
    cluster_score = _normalize(peak_cluster, 1.6, 4.8)
    axis_score = _normalize(max(axis_ratio, diagonal_ratio), 1.02, 1.55)
    return _clip01(0.48 * peak_score + 0.36 * cluster_score + 0.16 * axis_score)


def _local_frequency_score(gray: np.ndarray) -> float:
    work = gray
    height, width = work.shape
    if max(height, width) > 1400:
        scale = 1400 / float(max(height, width))
        work = cv2.resize(
            work,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = work.shape

    tile = min(192, max(96, min(height, width) // 4))
    if tile < 96 or height < tile or width < tile:
        return 0.0

    step = max(32, tile // 2)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    yy, xx = np.indices((tile, tile))
    center = tile // 2
    dx = xx - center
    dy = yy - center
    radius = np.sqrt(dx * dx + dy * dy)
    ring_mask = (radius > tile * 0.08) & (radius < tile * 0.46)
    axis_mask = ring_mask & (
        (np.abs(dx) < 3)
        | (np.abs(dy) < 3)
        | (np.abs(np.abs(dx) - np.abs(dy)) < 3)
    )

    peak_values: list[float] = []
    axis_values: list[float] = []
    energy_values: list[float] = []
    for y in range(0, height - tile + 1, step):
        for x in range(0, width - tile + 1, step):
            patch = work[y : y + tile, x : x + tile].astype(np.float32)
            if float(np.std(patch)) < 8.0:
                continue

            sigma = max(1.0, tile / 64.0)
            high_pass = patch - cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
            high_pass -= float(np.mean(high_pass))
            magnitude = np.log1p(
                np.abs(np.fft.fftshift(np.fft.fft2(high_pass * window)))
            ).astype(np.float32)

            ring_values = magnitude[ring_mask]
            if ring_values.size < 64:
                continue

            base = float(np.percentile(ring_values, 70))
            spread = float(np.percentile(ring_values, 92) - np.percentile(ring_values, 35)) + 1e-6
            peak = float(np.percentile(ring_values, 99.8))
            peak_values.append((peak - base) / spread)
            axis_values.append(_safe_mean(magnitude[axis_mask]) / (_safe_mean(ring_values) + 1e-6))
            energy_values.append(float(np.percentile(np.abs(high_pass), 85)))

    if not peak_values:
        return 0.0

    peaks = np.asarray(peak_values, dtype=np.float32)
    axis = np.asarray(axis_values, dtype=np.float32)
    energy = np.asarray(energy_values, dtype=np.float32)
    peak_score = _normalize(float(np.percentile(peaks, 90)), 1.55, 4.30)
    strong_peak_score = _normalize(float(np.percentile(peaks, 97)), 1.95, 5.80)
    axis_score = _normalize(float(np.percentile(axis, 85)), 1.00, 1.55)
    energy_score = _normalize(float(np.percentile(energy, 75)), 3.0, 18.0)
    return _clip01(
        0.42 * peak_score
        + 0.34 * strong_peak_score
        + 0.14 * axis_score
        + 0.10 * energy_score
    )


def _chroma_subpixel_score(image: np.ndarray, gray: np.ndarray) -> float:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    chroma_a = lab[:, :, 1] - 128.0
    chroma_b = lab[:, :, 2] - 128.0

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(sobel_x, sobel_y)
    flat_threshold = float(np.percentile(edge, 48))
    flat_mask = edge <= max(flat_threshold, 8.0)
    if int(np.count_nonzero(flat_mask)) < gray.size * 0.08:
        flat_mask = edge <= float(np.percentile(edge, 65))

    chroma_high = np.sqrt(
        _high_frequency_energy(chroma_a, flat_mask) ** 2
        + _high_frequency_energy(chroma_b, flat_mask) ** 2
    )
    luminance_high = _high_frequency_energy(luminance, flat_mask)
    chroma_ratio = chroma_high / (luminance_high + 0.75)

    absolute_score = _normalize(chroma_high, 1.2, 7.5)
    ratio_score = _normalize(chroma_ratio, 0.32, 1.15)
    return _clip01(0.62 * absolute_score + 0.38 * ratio_score)


def _banding_score(gray: np.ndarray) -> float:
    work = _resize_gray(gray, max_side=720).astype(np.float32)
    work = cv2.GaussianBlur(work, (0, 0), sigmaX=2.2, sigmaY=2.2)

    row_score = _profile_periodicity(np.mean(work, axis=1))
    col_score = _profile_periodicity(np.mean(work, axis=0))
    return _clip01(0.72 * row_score + 0.28 * col_score)


def _profile_periodicity(profile: np.ndarray) -> float:
    if profile.size < 80:
        return 0.0

    profile = profile.astype(np.float32)
    smooth_size = max(9, int(profile.size * 0.09) | 1)
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, smooth_size), 0).ravel()
    residual = profile - smooth
    residual -= float(np.mean(residual))

    window = np.hanning(residual.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(residual * window))
    if spectrum.size < 8:
        return 0.0

    start = 3
    stop = max(start + 1, int(spectrum.size * 0.42))
    values = spectrum[start:stop]
    median = float(np.median(values)) + 1e-6
    peak = float(np.percentile(values, 99.2))
    ratio = peak / median
    amplitude = float(np.std(residual)) / 255.0

    ratio_score = _normalize(ratio, 5.0, 22.0)
    amplitude_score = _normalize(amplitude, 0.006, 0.045)
    return _clip01(0.68 * ratio_score + 0.32 * amplitude_score)


def _screen_rectangle_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(height * width)
    best = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < 0.25 or area_ratio > 0.98:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(4, 2).astype(np.float32)
        angle_score = _quadrilateral_angle_score(points)
        fill_score = _normalize(area_ratio, 0.30, 0.82)
        border_score = _border_darkness_score(gray, points)
        candidate = _clip01(0.46 * angle_score + 0.34 * fill_score + 0.20 * border_score)
        best = max(best, candidate)

    return best


def _illumination_score(gray: np.ndarray) -> float:
    work = _resize_gray(gray, max_side=520).astype(np.float32)
    sigma = max(8.0, min(work.shape) / 9.0)
    low = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma, sigmaY=sigma)
    low = low / (float(np.mean(low)) + 1e-6)
    unevenness = float(np.std(low))

    top = float(np.percentile(work, 99.5))
    mid = float(np.percentile(work, 55)) + 1e-6
    glare_ratio = top / mid
    glare_area = float(np.mean(work >= top))

    uneven_score = _normalize(unevenness, 0.055, 0.24)
    glare_score = _normalize(glare_ratio, 1.18, 1.75) * _normalize(glare_area, 0.001, 0.02)
    return _clip01(0.75 * uneven_score + 0.25 * glare_score)


def _softness_score(gray: np.ndarray) -> float:
    work = _resize_gray(gray, max_side=620)
    lap_var = float(cv2.Laplacian(work, cv2.CV_32F).var())
    sharpness = np.log1p(lap_var)

    very_blurry = _normalize(5.2 - sharpness, 0.0, 1.6)
    moderately_soft = 1.0 - abs(float(sharpness) - 5.8) / 2.2
    return _clip01(0.34 * very_blurry + 0.66 * moderately_soft)


def _camera_artifact_score(gray: np.ndarray, illumination: float, softness: float) -> float:
    height, width = gray.shape
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.indices(small.shape)
    cy = cx = 47.5
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    center_mean = float(np.mean(small[radius < 22]))
    corner_mean = float(np.mean(small[radius > 52]))
    vignette = abs(center_mean - corner_mean) / 255.0

    strip = max(1, min(height, width) // 5)
    left = float(np.mean(gray[:, :strip]))
    right = float(np.mean(gray[:, width - strip :]))
    top = float(np.mean(gray[:strip, :]))
    bottom = float(np.mean(gray[height - strip :, :]))
    slope = max(abs(left - right), abs(top - bottom)) / 255.0

    bright_area = float(np.mean(gray > 245))
    highlight_ratio = float(np.percentile(gray, 99.5)) / (float(np.percentile(gray, 50)) + 1.0)
    highlight = _normalize(bright_area, 0.004, 0.035) * _normalize(highlight_ratio, 1.35, 3.20)

    vignette_score = _normalize(vignette, 0.04, 0.26)
    slope_score = _normalize(slope, 0.025, 0.16)
    return _clip01(
        0.30 * vignette_score
        + 0.22 * slope_score
        + 0.18 * highlight
        + 0.18 * illumination
        + 0.12 * softness
    )


def _white_clip_ratio(gray: np.ndarray) -> float:
    return _clip01(float(np.mean(gray > 245)))


def _screen_aspect_score(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    ratio = max(width / float(height), height / float(width))
    return _normalize(ratio, 1.55, 2.05)


def _overexposed_display_score(
    *,
    white_clip: float,
    screen_aspect: float,
    illumination: float,
    softness: float,
    camera_artifact: float,
    display_content: float,
    banding: float,
) -> float:
    if not 0.055 < white_clip <= 0.18:
        return 0.0
    if (
        screen_aspect < 0.55
        or illumination < 0.65
        or softness < 0.45
        or camera_artifact < 0.65
        or display_content < 0.35
        or banding < 0.25
    ):
        return 0.0

    evidence = float(
        np.mean(
            [
                _normalize(screen_aspect, 0.55, 1.0),
                _normalize(illumination, 0.65, 0.95),
                _normalize(softness, 0.45, 0.62),
                _normalize(camera_artifact, 0.65, 0.85),
                _normalize(display_content, 0.35, 0.65),
                _normalize(banding, 0.25, 0.55),
                1.0 - _normalize(white_clip, 0.16, 0.25),
            ]
        )
    )
    return _clip01(0.66 + 0.18 * evidence)


def _display_content_score(gray: np.ndarray) -> float:
    work = _resize_gray(gray, max_side=900)
    height, width = work.shape
    blurred = cv2.GaussianBlur(work, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_count = int(np.count_nonzero(edges))
    if edge_count == 0:
        return 0.0

    kernel_size = max(9, int(min(height, width) * 0.035))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    straight_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    straight_edges |= cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    straight_count = int(np.count_nonzero(straight_edges))
    straight_density = straight_count / float(edges.size)
    straight_ratio = straight_count / float(edge_count)

    sobel_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    angle = np.rad2deg(np.arctan2(np.abs(sobel_y), np.abs(sobel_x) + 1e-6))
    strong_mask = magnitude > float(np.percentile(magnitude, 80))
    if np.any(strong_mask):
        axis_gradient = float(np.mean((angle[strong_mask] < 12) | (angle[strong_mask] > 78)))
    else:
        axis_gradient = 0.0

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, int(min(height, width) * 0.08)),
        minLineLength=max(30, int(min(height, width) * 0.12)),
        maxLineGap=max(8, int(min(height, width) * 0.025)),
    )
    total_length = 0.0
    axis_length = 0.0
    line_count = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length <= 0:
                continue
            angle_degrees = abs(float(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))))
            if angle_degrees > 90:
                angle_degrees = 180 - angle_degrees
            total_length += length
            line_count += 1
            if min(angle_degrees, abs(angle_degrees - 90)) < 4:
                axis_length += length

    axis_line = axis_length / (total_length + 1e-6)
    line_density = line_count / (work.size / 10000.0)
    return _clip01(
        0.28 * _normalize(straight_ratio, 0.015, 0.16)
        + 0.24 * _normalize(straight_density, 0.0005, 0.018)
        + 0.22 * _normalize(axis_gradient, 0.48, 0.78)
        + 0.16 * _normalize(axis_line, 0.15, 0.70)
        + 0.10 * _normalize(line_density, 0.5, 8.0)
    )


def _screen_capture_combo_score(
    *,
    frequency: float,
    local_frequency: float,
    banding: float,
    illumination: float,
    softness: float,
    camera_artifact: float,
    display_content: float,
    white_clip: float,
    overexposed_display: float,
) -> float:
    has_supporting_screen_evidence = (
        frequency >= 0.025
        or local_frequency >= 0.12
        or display_content >= 0.18
        or camera_artifact >= 0.62
    )
    has_too_much_flat_white = white_clip > 0.055

    primary = 0.0
    if (
        not has_too_much_flat_white
        and has_supporting_screen_evidence
        and illumination >= 0.50
        and softness >= 0.34
        and banding >= 0.18
    ):
        primary = 0.64 + 0.18 * float(
            np.mean(
                [
                    _normalize(illumination, 0.50, 0.78),
                    _normalize(softness, 0.34, 0.55),
                    _normalize(banding, 0.18, 0.42),
                ]
            )
        )

    periodic_texture = max(frequency, local_frequency)
    secondary = 0.0
    if (
        not has_too_much_flat_white
        and illumination >= 0.30
        and softness >= 0.34
        and banding >= 0.52
        and (
            frequency >= 0.04
            or local_frequency >= 0.22
            or display_content >= 0.18
        )
    ):
        secondary = 0.62 + 0.20 * float(
            np.mean(
                [
                    _normalize(illumination, 0.30, 0.55),
                    _normalize(softness, 0.34, 0.55),
                    _normalize(banding, 0.52, 0.75),
                    _normalize(periodic_texture, 0.04, 0.24),
                    _normalize(max(display_content, camera_artifact), 0.18, 0.58),
                ]
            )
        )

    return _clip01(max(primary, secondary, overexposed_display))


def _screenshot_like_penalty(
    gray: np.ndarray,
    frequency: float,
    chroma: float,
    banding: float,
) -> float:
    lap_var = float(cv2.Laplacian(_resize_gray(gray, max_side=620), cv2.CV_32F).var())
    sharpness = np.log1p(lap_var)
    crisp_score = _normalize(sharpness, 6.2, 8.4)

    edges = cv2.Canny(gray, 60, 160)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))
    straight_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    straight_edges |= cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    straight_ratio = float(np.count_nonzero(straight_edges)) / float(gray.size)
    straight_score = _normalize(straight_ratio, 0.018, 0.085)

    no_camera_artifact = 1.0 - _clip01(0.46 * frequency + 0.36 * chroma + 0.18 * banding)
    return _clip01(0.42 * crisp_score + 0.28 * straight_score + 0.30 * no_camera_artifact)


def _high_frequency_energy(channel: np.ndarray, mask: np.ndarray) -> float:
    high = channel - cv2.GaussianBlur(channel, (0, 0), sigmaX=1.25, sigmaY=1.25)
    values = np.abs(high[mask])
    if values.size == 0:
        values = np.abs(high.ravel())
    return float(np.percentile(values, 82))


def _quadrilateral_angle_score(points: np.ndarray) -> float:
    ordered = _order_points(points)
    scores = []
    for index in range(4):
        prev_point = ordered[(index - 1) % 4]
        point = ordered[index]
        next_point = ordered[(index + 1) % 4]
        v1 = prev_point - point
        v2 = next_point - point
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_angle = abs(float(np.dot(v1, v2) / denom))
        scores.append(1.0 - _normalize(cos_angle, 0.08, 0.45))
    return _clip01(float(np.mean(scores)))


def _border_darkness_score(gray: np.ndarray, points: np.ndarray) -> float:
    height, width = gray.shape
    rect = cv2.boundingRect(points.astype(np.int32))
    x, y, w, h = rect
    pad = max(4, int(min(width, height) * 0.018))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)

    outer = gray[y0:y1, x0:x1]
    inner_x0 = min(width, max(0, x + pad))
    inner_y0 = min(height, max(0, y + pad))
    inner_x1 = max(0, min(width, x + w - pad))
    inner_y1 = max(0, min(height, y + h - pad))
    inner = gray[inner_y0:inner_y1, inner_x0:inner_x1]
    if outer.size == 0 or inner.size == 0:
        return 0.0

    outer_mean = float(np.mean(outer))
    inner_mean = float(np.mean(inner))
    dark_border = (inner_mean - outer_mean) / 255.0
    return _normalize(dark_border, 0.018, 0.18)


def _order_points(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _resize_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return gray
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip01((float(value) - low) / (high - low))


def _clip01(value: float | np.floating[Any]) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _build_reasons(features: dict[str, float]) -> list[str]:
    reason_map = [
        ("frequency", "存在疑似屏幕像素网格或摩尔纹频谱峰"),
        ("local_frequency", "局部区域存在疑似屏幕周期纹理"),
        ("chroma", "平坦区域存在异常高频彩色子像素噪声"),
        ("banding", "存在疑似滚动快门或刷新率条纹"),
        ("rectangle", "检测到较大的屏幕形矩形边界"),
        ("illumination", "亮度分布存在拍摄屏幕常见的不均匀性"),
        ("softness", "清晰度接近拍摄屏幕常见的轻微软化"),
        ("camera_artifact", "存在镜头阴影、亮斑或拍摄角度造成的光照痕迹"),
        ("display_content", "画面中存在屏幕内容常见的横竖边缘结构"),
        ("overexposed_display", "存在局部过曝但仍符合拍摄屏幕的宽幅显示器特征"),
    ]
    reasons = [text for key, text in reason_map if features.get(key, 0.0) >= 0.55]
    if features.get("screenshot_penalty", 0.0) >= 0.60:
        reasons.append("同时具有部分截图特征，已在总分中降权")
    return reasons
