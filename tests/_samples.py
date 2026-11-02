"""合成面单生成与破损模拟.

用途
----
1. **验收测试**: 生成"完好面单"与"破损面单"两组数据集, 量化识别率
   (任务书第八节验收标准).
2. **界面演示**: Streamlit 里一键生成示例面单, 无需准备真实素材即可演示全流程.

⚠️ 重要说明
-----------
合成数据**不能替代真实拍摄的面单**。它验证的是算法在已知退化类型下的
鲁棒性, 而真实场景还包含印刷工艺、材质反光、曲面形变等合成难以复现的因素。
因此: 合成集上的识别率**只作参考基线**, 最终验收必须在真实面单上复测。

所有随机性均由 ``seed`` 控制, 保证实验可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import zxingcpp

__all__ = [
    "GroundTruth",
    "make_waybill",
    "apply_damage",
    "put_text_cn",
    "DAMAGE_TYPES",
    "INTACT_PROFILE",
    "DAMAGED_PROFILE",
]

# --------------------------------------------------------------------------
# 面单模板
# --------------------------------------------------------------------------
_CARRIERS = [
    ("SF EXPRESS", "顺丰速运"),
    ("YTO EXPRESS", "圆通速递"),
    ("ZTO EXPRESS", "中通快递"),
    ("STO EXPRESS", "申通快递"),
    ("YUNDA", "韵达速递"),
    ("JD LOGISTICS", "京东物流"),
]
_CITIES = [
    ("北京市", "朝阳区", "建国路 88 号"),
    ("上海市", "浦东新区", "张江路 100 号"),
    ("广州市", "天河区", "体育西路 12 号"),
    ("深圳市", "南山区", "科技园南路 5 号"),
    ("杭州市", "西湖区", "文三路 45 号"),
    ("成都市", "武侯区", "人民南路 3 段"),
]
_SURNAMES = ["张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴"]


@dataclass
class GroundTruth:
    """一张合成面单的真值."""

    tracking_no: str
    barcodes: List[Dict[str, Any]] = field(default_factory=list)
    damage: List[str] = field(default_factory=list)
    carrier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_no": self.tracking_no,
            "carrier": self.carrier,
            "damage": list(self.damage),
            "barcodes": [
                {
                    "text": b["text"],
                    "format": b["format"],
                    "quad": [[float(p[0]), float(p[1])] for p in b["quad"]],
                }
                for b in self.barcodes
            ],
        }


def put_text_cn(
    img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
    color: Tuple[int, int, int] = (35, 35, 35), thick: int = 2,
) -> None:
    """在图像上绘制文本, **支持中文**.

    ``cv2.putText`` 只支持 ASCII (Hershey 字体), 中文会全部变成 ``?``。
    因此含中文的文本改走 PIL + 系统字体 (微软雅黑/黑体/宋体),
    纯 ASCII 文本仍用 ``cv2.putText`` (更快且更清晰)。

    注意坐标语义: ``org`` 的 y 是**文字基线**, 与 ``cv2.putText`` 保持一致。
    """
    if all(ord(c) < 128 for c in text):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        return
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = None
        for path in (
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ):
            try:
                font = ImageFont.truetype(path, max(10, int(scale * 32)))
                break
            except Exception:
                continue
        if font is None:
            cv2.putText(img, text.encode("ascii", "replace").decode(), org,
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            return
        draw.text((org[0], org[1] - int(scale * 26)), text, font=font,
                  fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text.encode("ascii", "replace").decode(), org,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# 向后兼容的私有别名 (早期版本使用 _put_text)
_put_text = put_text_cn


def _render_barcode(text: str, fmt, target_w: int, add_hrt: bool = True) -> np.ndarray:
    """渲染条码/二维码并缩放到目标宽度 (灰度 uint8)."""
    bc = zxingcpp.create_barcode(text, fmt)
    scale = 3
    img = np.asarray(zxingcpp.write_barcode_to_image(
        bc, scale=scale, add_hrt=add_hrt, add_quiet_zones=True
    ))
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return np.full((10, 10), 255, np.uint8)
    k = target_w / float(w)
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)


def make_waybill(
    seed: int = 0,
    size: Tuple[int, int] = (1000, 1500),
    carrier: Optional[str] = None,
    include_qr: bool = True,
    include_code128: bool = True,
) -> Tuple[np.ndarray, GroundTruth]:
    """渲染一张"完好"合成面单.

    Returns
    -------
    (BGR 图像, GroundTruth)
        真值中的 ``quad`` 为条码四点在**图像坐标系**下的坐标, 供评估 IoU/定位误差.
    """
    rng = np.random.default_rng(seed)
    W, H = size

    # --- 纸张底色 + 轻微纹理 (真实纸张不是纯白) ---
    base = int(rng.integers(246, 254))
    img = np.full((H, W, 3), base, np.uint8)
    texture = rng.normal(0, 1.6, (H, W, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    # --- 单号 ---
    prefix = ["SF", "YT", "ZT", "ST", "YD", "JD"][seed % 6]
    tracking = f"{prefix}{rng.integers(10**11, 10**12 - 1)}"
    carrier_en, carrier_cn = (
        (carrier, carrier) if carrier else _CARRIERS[seed % len(_CARRIERS)]
    )
    (pc, dc, street) = _CITIES[seed % len(_CITIES)]
    name = _SURNAMES[seed % len(_SURNAMES)] + "小明"
    phone = f"1{rng.integers(3, 9)}{rng.integers(10**8, 10**9 - 1)}"

    gt = GroundTruth(tracking_no=tracking, carrier=carrier_en)

    # --- 顶栏 ---
    cv2.rectangle(img, (0, 0), (W, 110), (60, 60, 62), -1)
    _put_text(img, carrier_en, (36, 72), 1.5, (255, 255, 255), 3)
    _put_text(img, carrier_cn, (W - 300, 72), 1.3, (230, 230, 230), 2)

    y = 150
    # --- 条形码 ---
    if include_code128:
        code = _render_barcode(tracking, zxingcpp.BarcodeFormat.Code128, target_w=int(W * 0.60))
        ch, cw = code.shape[:2]
        x0 = 50
        img[y:y + ch, x0:x0 + cw] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
        gt.barcodes.append({
            "text": tracking, "format": "Code128",
            "quad": np.array([[x0, y], [x0 + cw, y], [x0 + cw, y + ch], [x0, y + ch]], float),
        })

    # --- 二维码 ---
    if include_qr:
        qr_payload = f"https://ex.com/t?no={tracking}"
        qr = _render_barcode(qr_payload, zxingcpp.BarcodeFormat.QRCode,
                             target_w=int(W * 0.22), add_hrt=False)
        qh, qw = qr.shape[:2]
        qx, qy = W - qw - 50, y
        img[qy:qy + qh, qx:qx + qw] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        gt.barcodes.append({
            "text": qr_payload, "format": "QRCode",
            "quad": np.array([[qx, qy], [qx + qw, qy], [qx + qw, qy + qh], [qx, qy + qh]], float),
        })

    # --- 字段区 ---
    y2 = 470
    cv2.line(img, (40, y2), (W - 40, y2), (120, 120, 120), 3)
    rows = [
        ("收件人", f"{name}    {phone}"),
        ("地  址", f"{pc}{dc}{street}"),
        ("寄件人", f"王大力    13900001111"),
        ("目的地", f"{pc}{dc}"),
        ("单  号", tracking),
    ]
    for i, (k, v) in enumerate(rows):
        yy = y2 + 60 + i * 58
        _put_text(img, k, (55, yy), 0.85, (90, 90, 90), 2)
        _put_text(img, v, (190, yy), 0.90, (30, 30, 30), 2)

    # --- 条码下方的分拣码 / 格口 ---
    _put_text(img, f"SORT-{rng.integers(100,999)}", (55, H - 120), 0.8, (70, 70, 70), 2)
    segments = "".join(str(d) for d in rng.integers(0, 10, 4))
    for i, d in enumerate(f"{segments}-{rng.integers(10,99)}"):
        _put_text(img, d, (W - 260 + i * 30, H - 120), 0.8, (60, 60, 60), 2)

    return img, gt


# --------------------------------------------------------------------------
# 破损模拟
# --------------------------------------------------------------------------
DAMAGE_TYPES = (
    "blur", "motion_blur", "noise", "glare", "shadow", "stain", "crease",
    "tear", "low_contrast", "jpeg", "downscale", "rotate", "perspective",
    "occlusion", "exposure",
)


def _d_blur(img: np.ndarray, rng, sev: float) -> np.ndarray:
    k = int(3 + round(sev * 12))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _d_motion_blur(img: np.ndarray, rng, sev: float) -> np.ndarray:
    k = int(5 + sev * 25)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def _d_noise(img: np.ndarray, rng, sev: float) -> np.ndarray:
    sigma = 4 + sev * 22
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255)
    # 椒盐
    p = 0.001 + sev * 0.02
    mask = rng.random(img.shape[:2])
    out[mask < p / 2] = 0
    out[mask > 1 - p / 2] = 255
    return out.astype(np.uint8)


def _d_glare(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """反光: 椭圆高光叠加, 局部过曝把条码对比压平."""
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    ax, ay = int(w * (0.18 + 0.30 * sev)), int(h * (0.10 + 0.22 * sev))
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx) / max(1, ax)) ** 2 + ((yy - cy) / max(1, ay)) ** 2
    glow = np.clip(1.0 - d, 0, 1) ** 0.6
    strength = 0.55 + 0.45 * sev
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] += glow * 255.0 * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _d_shadow(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """阴影: 一侧变暗 (手持遮挡 / 光照不均)."""
    h, w = img.shape[:2]
    ang = rng.uniform(0, math.pi * 2)
    yy, xx = np.mgrid[0:h, 0:w]
    proj = np.cos(ang) * xx / w + np.sin(ang) * yy / h
    proj = (proj - proj.min()) / max(1e-6, proj.max() - proj.min())
    factor = 1.0 - sev * 0.75 * proj
    return np.clip(img.astype(np.float32) * factor[:, :, None], 0, 255).astype(np.uint8)


def _d_stain(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """污渍: 随机墨迹/水渍斑块."""
    h, w = img.shape[:2]
    layer = img.copy()
    n = int(2 + sev * 8)
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        axes = (int(rng.integers(w // 30, w // 8)), int(rng.integers(h // 40, h // 12)))
        ang = float(rng.uniform(0, 180))
        col = int(rng.integers(60, 150))
        cv2.ellipse(layer, (cx, cy), axes, ang, 0, 360, (col, col, col), -1)
    alpha = (0.25 + 0.55 * sev)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def _d_crease(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """褶皱: 明暗条带 (纸张折痕).

    真实折痕**不是等间距的**: 这里给投影距离叠一层低频抖动 (wobble),
    再叠一层幅度调制, 避免生成看起来像"规则条纹"的假图。
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(-0.9, 0.9)
    proj = xx * math.cos(ang) + yy * math.sin(ang)

    # 低频抖动: 让折痕间距不规则
    proj = proj + np.sin(proj * rng.uniform(0.004, 0.013) + rng.uniform(0, 6.283)) \
        * rng.uniform(4.0, 16.0)

    freq = rng.uniform(0.03, 0.11)
    amp = 10.0 + 38.0 * sev
    band = np.sin(proj * freq) * amp
    # 幅度调制: 折痕深浅不一
    band = band * (0.60 + 0.40 * np.sin(proj * freq * 0.31 + rng.uniform(0, 6.283)))

    out = np.clip(img.astype(np.float32) + band[:, :, None], 0, 255).astype(np.uint8)
    if sev > 0.5:  # 严重褶皱造成条码局部断裂
        mask = (np.sin(proj * freq * 1.7) > 0.90).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        out[mask > 0] = int(rng.integers(180, 255))
    return out


def _d_tear(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """撕裂/缺角: 用白色多边形遮挡一角."""
    h, w = img.shape[:2]
    corner = int(rng.integers(0, 4))
    cw = int(w * (0.12 + 0.32 * sev))
    chh = int(h * (0.10 + 0.30 * sev))
    pts = {
        0: [[0, 0], [cw, 0], [0, chh]],
        1: [[w, 0], [w - cw, 0], [w, chh]],
        2: [[w, h], [w - cw, h], [w, h - chh]],
        3: [[0, h], [cw, h], [0, h - chh]],
    }[corner]
    out = img.copy()
    cv2.fillPoly(out, [np.array(pts, np.int32)], (250, 250, 250))
    return out


def _d_low_contrast(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """低对比度: 整体压平动态范围 (曝光不准 / 褪色)."""
    lo = 70 + sev * 70
    hi = 255 - sev * 60
    out = lo + (img.astype(np.float32) / 255.0) * (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _d_jpeg(img: np.ndarray, rng, sev: float) -> np.ndarray:
    q = int(max(4, 60 - sev * 52))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _d_downscale(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """分辨率损失: 缩小再放大 (远距离/低像素拍摄)."""
    h, w = img.shape[:2]
    f = max(0.12, 1.0 - sev * 0.80)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _d_rotate(img: np.ndarray, rng, sev: float) -> np.ndarray:
    ang = float(rng.uniform(3, 6 + sev * 30) * (1 if rng.random() < 0.5 else -1))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(250, 250, 250))


def _d_perspective(img: np.ndarray, rng, sev: float) -> np.ndarray:
    h, w = img.shape[:2]
    d = 0.03 + sev * 0.16
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * rng.uniform(0, d), h * rng.uniform(0, d)],
        [w * (1 - rng.uniform(0, d)), h * rng.uniform(0, d)],
        [w * (1 - rng.uniform(0, d)), h * (1 - rng.uniform(0, d))],
        [w * rng.uniform(0, d), h * (1 - rng.uniform(0, d))],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(250, 250, 250))


def _d_occlusion(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """遮挡: 模拟胶带/贴纸压住条码局部."""
    h, w = img.shape[:2]
    n = int(1 + sev * 4)
    out = img.copy()
    for _ in range(n):
        x = int(rng.integers(0, max(1, w - 40)))
        y = int(rng.integers(0, max(1, h - 40)))
        bw = int(rng.integers(w // 25, w // 7))
        bh = int(rng.integers(h // 60, h // 22))
        col = int(rng.integers(200, 256))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (col, col, col), -1)
    return out


def _d_exposure(img: np.ndarray, rng, sev: float) -> np.ndarray:
    """过曝 / 欠曝."""
    gamma = 1.0 + (1.6 * sev if rng.random() < 0.5 else -0.55 * sev)
    gamma = max(0.25, gamma)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


_DAMAGE_FUNCS = {
    "blur": _d_blur,
    "motion_blur": _d_motion_blur,
    "noise": _d_noise,
    "glare": _d_glare,
    "shadow": _d_shadow,
    "stain": _d_stain,
    "crease": _d_crease,
    "tear": _d_tear,
    "low_contrast": _d_low_contrast,
    "jpeg": _d_jpeg,
    "downscale": _d_downscale,
    "rotate": _d_rotate,
    "perspective": _d_perspective,
    "occlusion": _d_occlusion,
    "exposure": _d_exposure,
}

#: 完好面单: 只做极轻微退化 (模拟正常拍摄), 用于验证"清晰码必须能解"
INTACT_PROFILE: Sequence[Tuple[str, Tuple[float, float]]] = (
    ("noise", (0.02, 0.10)),
    ("jpeg", (0.03, 0.15)),
    ("shadow", (0.02, 0.08)),
)

#: 破损面单: 从这些类型中随机组合 2~4 种, 严重度较高。
#: 刻意包含 rotate / perspective / tear 等**几何破坏** —— 真实破损面单
#: 往往是"褶皱 + 污渍 + 拍歪"叠加出现, 只做单一退化会严重高估识别率。
DAMAGED_PROFILE: Sequence[Tuple[str, Tuple[float, float]]] = (
    ("blur", (0.45, 0.90)),
    ("motion_blur", (0.45, 0.95)),
    ("noise", (0.45, 0.95)),
    ("glare", (0.50, 1.00)),
    ("shadow", (0.55, 1.00)),
    ("stain", (0.50, 1.00)),
    ("crease", (0.55, 1.00)),
    ("low_contrast", (0.50, 0.95)),
    ("jpeg", (0.55, 1.00)),
    ("downscale", (0.45, 0.90)),
    ("exposure", (0.50, 1.00)),
    ("occlusion", (0.40, 0.85)),
    ("rotate", (0.35, 0.85)),
    ("perspective", (0.35, 0.85)),
    ("tear", (0.25, 0.60)),
)


def apply_damage(
    img: np.ndarray,
    damage: Sequence[str],
    severities: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> np.ndarray:
    """按名称序列依次施加退化 (顺序会影响效果, 例如先去噪后加噪毫无意义)."""
    rng = np.random.default_rng(seed + 7919)
    out = img.copy()
    for i, name in enumerate(damage):
        fn = _DAMAGE_FUNCS.get(name)
        if fn is None:
            raise ValueError(f"未知破损类型 {name!r}, 可选: {sorted(_DAMAGE_FUNCS)}")
        sev = float(severities[i]) if severities is not None and i < len(severities) else 0.6
        out = fn(out, rng, float(np.clip(sev, 0.0, 1.0)))
    return out
