from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

from vector_formats import TextureData, ModelData, parse_texture_asset, parse_model_asset

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.tga', '.bmp', '.dds'}


def _texture_from_image(path: Path) -> TextureData:
    if Image is None:
        raise RuntimeError('Pillow is required for image skin import.')
    with Image.open(path) as im:
        im = im.convert('RGBA')
        rgba = np.asarray(im, dtype=np.uint8).copy()
        w, h = im.size
    return TextureData(path.stem, w, h, 1, -10, rgba, f'External {path.suffix.upper()}')


def _texture_from_bin(path: Path) -> TextureData:
    data = path.read_bytes()
    # VuTextureAsset .bin payloads use the same Vector texture container that
    # the APF decoder already understands.  Do not confuse VuAnimatedModelAsset
    # skin/model .bin files with textures: those fail the texture parser and are
    # deliberately ignored by package discovery.
    return parse_texture_asset(data, path.stem, '')


def load_texture_file(path: str | Path) -> TextureData:
    p = Path(path)
    if p.suffix.lower() == '.bin':
        return _texture_from_bin(p)
    if p.suffix.lower() in IMAGE_EXTS:
        return _texture_from_image(p)
    raise ValueError(f'Unsupported skin texture format: {p.suffix}')


def _key(value: str) -> str:
    s = str(value or '').replace('\\', '/').strip().lower()
    return Path(s).stem.lower()


def load_skin_package(path: str | Path):
    """Load image/texture BINs and, when present, animated-model skin BINs.

    Texture BINs are decoded as VuTextureAsset payloads. If a BIN is not a
    texture, it is tested as a VuAnimatedModelAsset payload. This is important
    for extracted BBR character folders where e.g. Cyber_SkinA.bin exists in
    VuAnimatedModelAsset while Cyber_SkinA.bin also exists separately under
    VuTextureAsset.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.is_file() and
                        (x.suffix.lower() in IMAGE_EXTS or x.suffix.lower() == '.bin')])
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(str(p))

    textures: Dict[str, TextureData] = {}
    models: Dict[str, ModelData] = {}
    model_bytes: Dict[str, bytes] = {}
    texture_templates: Dict[str, bytes] = {}
    ignored = []
    errors = []
    for f in files:
        if f.suffix.lower() in IMAGE_EXTS:
            try:
                textures[_key(f.name)] = _texture_from_image(f)
            except Exception as exc:
                ignored.append(f.name); errors.append(f'{f.name}: {exc}')
            continue
        try:
            textures[_key(f.name)] = _texture_from_bin(f)
            texture_templates[_key(f.name)] = f.read_bytes()
            continue
        except Exception as tex_exc:
            pass
        try:
            raw_model = f.read_bytes()
            models[_key(f.name)] = parse_model_asset(raw_model, f.stem, str(f))
            model_bytes[_key(f.name)] = raw_model
        except Exception as model_exc:
            ignored.append(f.name)
            errors.append(f'{f.name}: texture={tex_exc}; model={model_exc}')

    if not textures and not models:
        detail = '\n'.join(errors[:8])
        raise ValueError('No decodable BBR skin asset/image was found.' + ('\n' + detail if detail else ''))
    return textures, {
        'source': str(p), 'count': len(textures), 'model_count': len(models),
        'models': models, 'model_bytes': model_bytes, 'texture_templates': texture_templates,
        'ignored': ignored, 'errors': errors,
    }

def match_texture(textures: Dict[str, TextureData], mesh_name: str = '',
                  material_texture: str = '', default: Optional[TextureData] = None):
    candidates = []
    for value in (material_texture, mesh_name):
        if value:
            k = _key(value)
            if k:
                candidates.append(k)
                # Common BBR suffix variants.
                if k.endswith('_skina'):
                    candidates.append(k[:-6])
    for k in candidates:
        if k in textures:
            return textures[k]
    # A single imported texture is a valid explicit override for the whole
    # model and is the expected workflow for a custom atlas.
    if len(textures) == 1:
        return next(iter(textures.values()))
    return default
