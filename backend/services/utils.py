import base64
import io
from PIL import Image

def b64_to_image(b64: str, mode: str = "RGB") -> Image.Image:
    if "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img.convert(mode)

def pil_to_b64(img, fmt="PNG", quality=95) -> str:
    buf = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        img = img.convert("RGB")  # JPEG는 알파 채널 불가
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
