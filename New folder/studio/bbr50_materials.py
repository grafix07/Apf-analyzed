from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import re
import struct
import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord
from vector_formats import MaterialData, TextureData, CubeTextureData


@dataclass
class MaterialBinding:
    mesh_name: str
    material_record: Optional[AssetRecord]
    material: MaterialData
    diffuse_texture_record: Optional[AssetRecord] = None
    diffuse_texture: Optional[TextureData] = None
    # General shader mask (Character DiffuseAdditiveMask*, etc.). This is
    # separate from the vehicle CarPaint decal mask below.
    mask_texture_record: Optional[AssetRecord] = None
    mask_texture: Optional[TextureData] = None
    mask_cache_key: Optional[Tuple[Any, ...]] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)
    texture_cache_key: Optional[Tuple[Any, ...]] = None
    detail_texture_record: Optional[AssetRecord] = None
    detail_texture: Optional[TextureData] = None
    detail_cache_key: Optional[Tuple[Any, ...]] = None
    detail_uv_scale: float = 1.0
    alpha_texture_record: Optional[AssetRecord] = None
    alpha_texture: Optional[TextureData] = None
    env_cube_record: Optional[AssetRecord] = None
    env_cube: Optional[CubeTextureData] = None
    env_cube_cache_key: Optional[Tuple[Any, ...]] = None
    additive_env_cube_record: Optional[AssetRecord] = None
    additive_env_cube: Optional[CubeTextureData] = None
    additive_env_cube_cache_key: Optional[Tuple[Any, ...]] = None
    normal_texture_record: Optional[AssetRecord] = None
    normal_texture: Optional[TextureData] = None
    normal_cache_key: Optional[Tuple[Any, ...]] = None
    fresnel_texture_record: Optional[AssetRecord] = None
    fresnel_texture: Optional[TextureData] = None
    fresnel_cache_key: Optional[Tuple[Any, ...]] = None
    dynamic_environment: bool = False

    # Exact three-texture input used by Art/CarDriver/CarPaint*.  These remain
    # separate because the game samples the two colour ramps by incident angle
    # and the decal mask by mesh UV; baking them into one flat bitmap loses that
    # information (and was the main paint error in v0.51).
    carpaint: bool = False
    paint_texture_record: Optional[AssetRecord] = None
    paint_texture: Optional[TextureData] = None
    decal_color_texture_record: Optional[AssetRecord] = None
    decal_color_texture: Optional[TextureData] = None
    decal_mask_texture_record: Optional[AssetRecord] = None
    decal_mask_texture: Optional[TextureData] = None
    paint_cache_key: Optional[Tuple[Any, ...]] = None
    decal_color_cache_key: Optional[Tuple[Any, ...]] = None
    decal_mask_cache_key: Optional[Tuple[Any, ...]] = None
    # generic/custom use a colour ramp + colour ramp + red-channel decal mask.
    # custom_decal uses a colour ramp + authored RGBA decal texture.
    carpaint_mode: str = ""
    # Whether the actual VuShaderAsset declares aColor/COLOR0.  The renderer
    # must not multiply arbitrary packed vertex bytes into textures for shaders
    # that do not consume vertex colour.
    uses_vertex_color: bool = False

    @property
    def alpha(self) -> float:
        try:
            return float(self.material.diffuse_color[3])
        except Exception:
            return 1.0

    @property
    def alpha_test(self) -> bool:
        # Vector Unit's *1bit* shaders are cutout materials (hair/feathers,
        # foliage-style cards, etc.).  Treating those alpha bits as ordinary
        # transparency produces the dark/ghosted polygons seen on Tribal_SkinA.
        shader = str(self.material.shader or "").strip().lower()
        return ("1bit" in shader or "alpha_test" in shader or "alphatest" in shader)

    @property
    def transparent(self) -> bool:
        if self.alpha_test:
            return False
        if self.alpha < 0.995:
            return True
        # Do not infer blending from arbitrary texture alpha alone. Several BBR2
        # vehicle materials (notably HWDonutDrifter/HWSquiggle using
        # Art/CarDriver/DiffuseEnvmapMask) store useful mask/data in the alpha
        # channel even though the shader is opaque. Treating every non-255 alpha
        # texel as transparency punched holes through otherwise solid bodywork.
        shader = str(self.material.shader or "").strip().lower()
        alpha_shader = any(token in shader for token in (
            "alpha", "transparent", "transluc", "blend", "glass",
            "lens", "cloud", "water", "fade",
        ))
        if not alpha_shader:
            return False
        tex = self.diffuse_texture
        if tex is not None and getattr(tex, "rgba", None) is not None and tex.rgba.size:
            a = tex.rgba[..., 3]
            return bool(np.min(a) < 250)
        return False


def _representative_rgb(tex: Optional[TextureData], fallback=(0.55, 0.55, 0.55)) -> np.ndarray:
    """v0.43/v0.44 representative colour for 1-D paint ramps.

    The known-good preview used the median of visible ramp pixels rather than
    sampling an incident-angle position.  Keep that stable behaviour for the
    fixed-function compatibility renderer restored in v0.56.
    """
    if tex is None or tex.rgba is None or not tex.rgba.size:
        return np.asarray(fallback, dtype=np.float32)
    rgba = np.asarray(tex.rgba, dtype=np.float32) / 255.0
    rgb = rgba[..., :3].reshape((-1, 3))
    if rgba.shape[2] >= 4:
        alpha = rgba[..., 3].reshape(-1)
        keep = alpha > 0.05
        if np.any(keep):
            rgb = rgb[keep]
    if not len(rgb):
        return np.asarray(fallback, dtype=np.float32)
    return np.clip(np.median(rgb, axis=0), 0.0, 1.0).astype(np.float32)


class MaterialResolver50:
    """Resolve BBR materials and authored vehicle skin inputs.

    Target CarPaint GLSL establishes the exact core operation::

        paint = texture2D(VehiclePaintColor, incidentAngle).rgb
        decal = texture2D(VehicleDecalColor, incidentAngle).rgb
        mask  = texture2D(VehicleDecalTexture, uv).r
        diffuse.rgb *= mix(paint, decal, mask)

    Only the two generic material slots ``Car/Paint/Paint`` and
    ``Car/Paint/Paint_skinned`` receive the per-vehicle Vehicle Skins row.

    v0.56 deliberately restores the v0.43 *preview* path for vehicle paint:
    paint/decal inputs are composited to a conventional 2-D preview texture and
    drawn through the ordinary material renderer.  That was the last vehicle
    path the supplied BBR2 package mapped correctly in the Studio.  The newer
    experimental incident-angle CarPaint GLSL remains available for non-vehicle
    material inspection, but is not used for chassis or wheel preview.
    """

    def __init__(self, db: AssetDatabase):
        self.db = db
        self._cache: Dict[str, MaterialBinding] = {}
        # Mobile BBR2 uses "Vehicle Skins"; Island Adventure uses the
        # Vehicle rows in "Vehicle Configs" for the same paint/decal inputs.
        self._vehicle_skin_rows = db.spreadsheet_by_name("Vehicle Skins")
        if not self._vehicle_skin_rows and getattr(db, "is_island_adventure", False):
            self._vehicle_skin_rows = {
                str(row.get("Name")): row
                for row in db.spreadsheet("Vehicle Configs")
                if isinstance(row, dict)
                and str(row.get("Type") or "").lower() == "vehicle"
                and str(row.get("Name") or "")
            }
        self._wheel_rows = db.spreadsheet_by_name("Wheels")
        self._vehicle_name = ""
        self._vehicle_skin_key = ""
        self._vehicle_skin: Dict[str, Any] = {}
        self._vehicle_skin_texture: Optional[TextureData] = None
        self._vehicle_skin_note = ""
        self._skin_assets_cache = None
        self._shader_vertex_color_cache: Dict[str, bool] = {}
        # Runtime cube-image macros (Proxy_cube / Proxy_additive) are authored
        # by each track setting variant through VuSetCubeImageMacroEntity.
        self._cube_macros: Dict[str, str] = {}

    _GENERIC_CARPAINT_SHADERS = {
        "art/cardriver/carpaint",
        "art/cardriver/carpaintskinned",
        "art/cardriver/carpaint_matte",
        "art/cardriver/carpaintskinned_matte",
    }
    _CUSTOM_CARPAINT_SHADERS = {
        "art/cardriver/carpaint_custom",
        "art/cardriver/carpaint_customskinned",
    }
    _CUSTOM_DECAL_SHADERS = {
        "art/cardriver/carpaint_customdecal",
        "art/cardriver/carpaint_customdecal_skinned",
    }

    def set_cube_macros(self, mapping: Optional[Dict[str, str]] = None):
        clean={str(k):str(v) for k,v in dict(mapping or {}).items() if str(k) and str(v)}
        if clean == self._cube_macros:
            return
        self._cube_macros=clean
        # Environment bindings cache resolved cube records, so changing track
        # settings must invalidate them before display-list compilation.
        self._cache.clear()

    def _resolved_cube_ref(self, logical_ref: str) -> str:
        ref=str(logical_ref or "").strip()
        return self._cube_macros.get(ref, ref)

    def set_vehicle(self, vehicle_name: str, skin_key: str = ""):
        vehicle_name = str(vehicle_name or "")
        skin_key = str(skin_key or vehicle_name)
        if vehicle_name == self._vehicle_name and skin_key == self._vehicle_skin_key:
            return
        self._vehicle_name = vehicle_name
        self._vehicle_skin_key = skin_key
        # Exact variant rows win.  On Island Adventure, alternate SkinA
        # projects often have no Vehicle Config row; in that case do not fall
        # back to the base vehicle paint, because the variant project carries
        # its own authored material setup.
        row = self._vehicle_skin_rows.get(skin_key)
        if row is None and (not getattr(self.db, "is_island_adventure", False) or skin_key == vehicle_name):
            row = self._vehicle_skin_rows.get(vehicle_name)
        self._vehicle_skin = dict(row or {})
        self._vehicle_skin_texture = None
        self._vehicle_skin_note = ""
        self._skin_assets_cache = None
        self._cache.clear()

    @staticmethod
    def _mesh_candidates(mesh_name: str):
        name = str(mesh_name or "").strip().strip("/")
        out = []
        if name:
            out.append(name)
            clean = re.sub(r"_LOD\d+$", "", name, flags=re.I)
            if clean != name:
                out.append(clean)
        seen = set()
        for x in out:
            if x not in seen:
                seen.add(x)
                yield x

    def _load_texture(self, logical_ref: str) -> Tuple[Optional[AssetRecord], Optional[TextureData]]:
        if not logical_ref:
            return None, None
        rec = self.db.resolve_texture(logical_ref)
        if not rec:
            return None, None
        try:
            return rec, self.db.texture(rec)
        except Exception:
            return rec, None

    def _load_cube_texture(self, logical_ref: str) -> Tuple[Optional[AssetRecord], Optional[CubeTextureData]]:
        if not logical_ref:
            return None, None
        resolved=self._resolved_cube_ref(logical_ref)
        rec = self.db.resolve_cube_texture(resolved)
        if not rec:
            return None, None
        try:
            return rec, self.db.cube_texture(rec)
        except Exception:
            return rec, None

    def _load_vehicle_decal_texture(self, logical_ref: str) -> Tuple[Optional[AssetRecord], Optional[TextureData]]:
        """Load a vehicle decal using the last known-good v0.43/v0.44 mask path.

        The newer v0.50+ texture path reclassified Android format code 1 as an
        ETC2A8 image.  The vehicle preview that the user verified before the BIN
        work treated these Car/Decal assets as the stored single-channel mask.
        Reproduce that behaviour *only* for Car/Decal format-1 assets so the
        rollback does not disturb character, environment or effect textures.
        """
        if not logical_ref:
            return None, None
        rec = self.db.resolve_texture(logical_ref)
        if not rec:
            return None, None
        try:
            raw = self.db.decode(rec).data
            if (str(logical_ref).strip('/').lower().startswith('car/decal/')
                    and len(raw) >= 42):
                fmt = struct.unpack_from('<H', raw, 0x16)[0]
                width = struct.unpack_from('<H', raw, 0x1A)[0]
                height = struct.unpack_from('<H', raw, 0x1E)[0]
                mips = struct.unpack_from('<H', raw, 0x22)[0]
                payload_size = struct.unpack_from('<I', raw, 0x26)[0]
                if fmt == 1 and width > 0 and height > 0 and payload_size <= len(raw)-42:
                    payload = raw[42:42+payload_size]
                    top_size = int(width) * int(height)
                    if len(payload) >= top_size:
                        ch = np.frombuffer(payload[:top_size], dtype=np.uint8).reshape((height, width)).copy()
                        rgba = np.empty((height, width, 4), dtype=np.uint8)
                        rgba[..., 0] = ch
                        rgba[..., 1] = ch
                        rgba[..., 2] = ch
                        rgba[..., 3] = 255
                        return rec, TextureData(
                            name=str(logical_ref), width=width, height=height,
                            mip_count=mips, format_code=fmt, rgba=rgba,
                            format_name='v0.44 vehicle decal R8 compatibility mask',
                        )
            return rec, self.db.texture(rec)
        except Exception:
            return rec, None

    @staticmethod
    def _black_mask(name="VehicleSkin/NoDecal") -> TextureData:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return TextureData(name=name, width=4, height=4, mip_count=1,
                           format_code=-52, rgba=rgba, format_name="SyntheticBlackMask")

    @staticmethod
    def _transparent_decal(name="Material/NoCustomDecal") -> TextureData:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        return TextureData(name=name, width=4, height=4, mip_count=1,
                           format_code=-54, rgba=rgba, format_name="SyntheticTransparentDecal")

    def _paint_assets_from_row(self, row: Dict[str, Any], note_prefix: str):
        row = dict(row or {})
        paint_name = str(row.get("Paint Color") or "")
        decal_color_name = str(row.get("Decal Color") or "")
        decal_name = str(row.get("Decal") or "")

        paint_rec, paint_tex = self._load_texture("Car/Paint/" + paint_name) if paint_name and paint_name.lower() != "none" else (None, None)
        dcol_rec, dcol_tex = self._load_texture("Car/Paint/" + decal_color_name) if decal_color_name and decal_color_name.lower() != "none" else (None, None)
        if dcol_tex is None:
            dcol_rec, dcol_tex = paint_rec, paint_tex

        if decal_name and decal_name.lower() != "none":
            dmask_rec, dmask_tex = self._load_vehicle_decal_texture("Car/Decal/" + decal_name)
        else:
            dmask_rec, dmask_tex = None, self._black_mask(note_prefix + "/NoDecal")

        note = f"{note_prefix}: paint={paint_name or 'None'}, decalColor={decal_color_name or 'None'}, decal={decal_name or 'None'}"
        return paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex, note

    def _effective_vehicle_skin(self) -> Dict[str, Any]:
        row = dict(self._vehicle_skin or {})
        # Screamin' Jack/Pumpkin ships a dedicated pumpkin-face decal even
        # though its Vehicle Skins row says Decal=None.
        if (self._vehicle_name == "Pumpkin" and row
                and str(row.get("Decal") or "None").lower() == "none"
                and self.db.resolve_texture("Car/Decal/Pumpkin")):
            row["Decal"] = "Pumpkin"
            row["Decal Color"] = row.get("Decal Color") or "Orange_Light"

        return row

    def _shader_uses_vertex_color(self, shader: str) -> bool:
        shader = str(shader or "").strip().strip("/")
        if not shader:
            return False
        if shader in self._shader_vertex_color_cache:
            return self._shader_vertex_color_cache[shader]
        rec = self.db.get("VuShaderAsset/" + shader)
        uses = False
        if rec is not None:
            try:
                data = self.db.decode(rec).data
                # Shader assets contain the generated GLSL text.  Checking the
                # declared input is safer than guessing from mesh stride.
                uses = (b"attribute vec4 aColor" in data or b"in.var.COLOR0" in data)
            except Exception:
                uses = False
        self._shader_vertex_color_cache[shader] = bool(uses)
        return bool(uses)

    def _vehicle_skin_assets(self):
        if self._skin_assets_cache is not None:
            return self._skin_assets_cache
        row = self._effective_vehicle_skin()
        if not row:
            self._skin_assets_cache = (None, None, None, None, None, None, "")
            return self._skin_assets_cache
        sheet = "Vehicle Configs" if getattr(self.db, "is_island_adventure", False) else "Vehicle Skins"
        self._skin_assets_cache = self._paint_assets_from_row(
            row, f"{sheet}/{self._vehicle_skin_key or self._vehicle_name}"
        )
        return self._skin_assets_cache

    def _wheel_skin_assets(self, wheel_name: str):
        row = dict(self._wheel_rows.get(str(wheel_name or ""), {}) or {})
        paint_name = row.get("Render Paint Color")
        decal_color = row.get("Render Decal Color")
        decal = row.get("Render Decal")

        # The default Candy Coupe uses peppermint red/white wheels in the
        # authored holiday presentation. Keep the wheel model's StripeA
        # coverage but use the same red/white palette as the fixed livery.
        if self._vehicle_name == "CandyCane_Default" and self._vehicle_skin_key == "CandyCane_Default":
            paint_name = "Red_Bright"
            decal_color = "White"
            decal = decal or "StripeA"

        if not paint_name:
            return None
        paint_row = {
            "Paint Color": paint_name,
            "Decal Color": decal_color or paint_name,
            "Decal": decal or "None",
        }
        return self._paint_assets_from_row(
            paint_row, f"Wheels/{wheel_name or 'Unknown'}"
        )

    def _build_vehicle_skin(self) -> Tuple[Optional[TextureData], str]:
        """v0.43/v0.44-compatible paint/decal composite for vehicle preview.

        Vehicle decal assets are loaded through the legacy single-channel mask
        path above, then composited with the authored paint/decal colours.
        """
        if self._vehicle_skin_texture is not None:
            return self._vehicle_skin_texture, self._vehicle_skin_note
        pre, ptex, cre, ctex, mre, mtex, note = self._vehicle_skin_assets()
        if ptex is None:
            return None, note
        base_rgb = _representative_rgb(ptex, (0.55, 0.55, 0.55))
        decal_rgb = _representative_rgb(ctex, base_rgb)
        if mtex is None:
            mask = np.zeros((4, 4, 1), dtype=np.float32)
        else:
            # v0.43 preview semantics: decals are grayscale/red-channel masks,
            # but when the decoded asset carries meaningful alpha it also gates
            # the coverage.  The later red-only shader path produced the
            # scattered paint fragments visible on Motorcycle_Dirt.
            src = np.asarray(mtex.rgba, dtype=np.float32) / 255.0
            mask2 = np.clip(src[..., 0], 0.0, 1.0)
            if src.shape[2] >= 4 and np.any(src[..., 3] < 0.999):
                mask2 = mask2 * np.clip(src[..., 3], 0.0, 1.0)
            mask = mask2[..., None]
        h, w = mask.shape[:2]
        rgb = base_rgb.reshape((1,1,3)) * (1.0-mask) + decal_rgb.reshape((1,1,3)) * mask
        rgba = np.empty((h,w,4), dtype=np.uint8)
        rgba[...,:3] = np.clip(np.round(rgb*255.0),0,255).astype(np.uint8)
        rgba[...,3] = 255
        tex = TextureData(name=f"VehicleSkin/{self._vehicle_name}", width=w, height=h,
                          mip_count=1, format_code=-51, rgba=rgba,
                          format_name="VehicleSkinFallbackComposite")
        self._vehicle_skin_texture = tex
        self._vehicle_skin_note = note
        return tex, note

    @staticmethod
    def _resize_rgba_nearest(rgba, width: int, height: int):
        src = np.asarray(rgba, dtype=np.uint8)
        if src.shape[1] == width and src.shape[0] == height:
            return src.copy()
        yi = np.minimum(src.shape[0]-1, (np.arange(height) * src.shape[0] / max(1, height)).astype(np.int32))
        xi = np.minimum(src.shape[1]-1, (np.arange(width) * src.shape[1] / max(1, width)).astype(np.int32))
        return src[yi[:, None], xi[None, :]].copy()

    def _candycane_default_texture(self) -> Optional[TextureData]:
        """Recreate the v0.43 CandyCane_Default fixed-livery preview.

        The default project model has a generic paint shell, but the package
        also ships CandyCane_SkinRainbow whose geometry/UVs contain the exact
        stripe layout.  VehicleResolver50 swaps to that authored shell in
        v0.56; here we recolor its Rainbow stripe bands to peppermint
        red/white.  Do not resize/bake the generic Snow customization mask onto
        this fixed-livery UV set; that was the remaining v0.56 Candy mismatch.
        """
        rrec, rainbow = self._load_texture("Car/Vehicles/CandyCane/CandyCane_Rainbow")
        if rainbow is None or rainbow.rgba is None:
            return None
        src = np.asarray(rainbow.rgba, dtype=np.float32) / 255.0
        rgb = src[..., :3]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = np.max(rgb, axis=2); mn = np.min(rgb, axis=2); d = mx - mn
        h = np.zeros_like(mx, dtype=np.float32); nz = d > 1e-6
        mr = nz & (mx == r); mg = nz & (mx == g); mb = nz & (mx == b)
        h[mr] = ((g[mr] - b[mr]) / d[mr]) % 6.0
        h[mg] = ((b[mg] - r[mg]) / d[mg]) + 2.0
        h[mb] = ((r[mb] - g[mb]) / d[mb]) + 4.0
        h /= 6.0
        sector = np.floor(h * 6.0 + 0.5).astype(np.int32) % 6
        sat = np.zeros_like(mx, dtype=np.float32); valid = mx > 1e-6
        sat[valid] = d[valid] / mx[valid]
        white_band = np.isin(sector, (2, 3, 4)) | (sat < 0.14)
        red = np.asarray((0.93, 0.055, 0.025), dtype=np.float32)
        white = np.asarray((0.97, 0.97, 0.95), dtype=np.float32)
        out = np.empty_like(rgb); out[:] = red; out[white_band] = white
        shade = np.clip(0.45 + mx * 0.55, 0.35, 1.0)
        out *= shade[..., None]
        dark = mx < 0.16
        if np.any(dark):
            out[dark] = rgb[dark]

        rgba = np.empty((rainbow.height, rainbow.width, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
        rgba[..., 3] = 255
        return TextureData(
            name="CandyCane_Default_Christmas", width=rainbow.width, height=rainbow.height,
            mip_count=1, format_code=-56, rgba=rgba,
            format_name="v0.43 CandyCane fixed-livery reconstruction",
        )

    @classmethod
    def _carpaint_mode(cls, mat: MaterialData) -> str:
        shader = str(mat.shader or "").strip().lower()
        if shader in cls._GENERIC_CARPAINT_SHADERS:
            return "generic"
        if shader in cls._CUSTOM_CARPAINT_SHADERS:
            return "custom"
        if shader in cls._CUSTOM_DECAL_SHADERS:
            return "custom_decal"
        return ""


    @staticmethod
    def _sample_repeat_rgba(tex: TextureData, width: int, height: int, scale: float = 1.0) -> np.ndarray:
        src=np.asarray(tex.rgba,dtype=np.uint8)
        if src.ndim!=3 or src.shape[2]<4 or width<=0 or height<=0:
            return np.zeros((max(1,height),max(1,width),4),dtype=np.uint8)
        u=(np.arange(width,dtype=np.float32)+0.5)/float(width)
        v=(np.arange(height,dtype=np.float32)+0.5)/float(height)
        u=np.mod(u*float(scale),1.0); v=np.mod(v*float(scale),1.0)
        xi=np.minimum(src.shape[1]-1,(u*src.shape[1]).astype(np.int32))
        yi=np.minimum(src.shape[0]-1,(v*src.shape[0]).astype(np.int32))
        return src[yi[:,None],xi[None,:]].copy()

    @staticmethod
    def _cube_representative_rgb(cube: Optional[CubeTextureData]) -> Optional[np.ndarray]:
        if cube is None or not getattr(cube,'faces',None):
            return None
        chunks=[]
        for face in cube.faces[:6]:
            rgba=np.asarray(face.rgba,dtype=np.float32)
            if rgba.ndim==3 and rgba.shape[2]>=3 and rgba.size:
                step=max(1,int(max(rgba.shape[0],rgba.shape[1])/64))
                chunks.append(rgba[::step,::step,:3].reshape(-1,3))
        if not chunks:
            return None
        a=np.concatenate(chunks,axis=0)/255.0
        return np.clip(np.median(a,axis=0),0.0,1.0).astype(np.float32)

    def _environment_preview_texture(self, base: Optional[TextureData],
                                     env_cube: Optional[CubeTextureData], mask_tex: Optional[TextureData],
                                     fresnel_tex: Optional[TextureData], name: str) -> Optional[TextureData]:
        """Approximate environment reflection while retaining the authored base alpha.

        Detail maps must be sampled from their own tiled UVs by the renderer. The
        game's 1-bit shaders discard from DiffuseTexture.a, so the base alpha
        must not be replaced by OneBitAlphaTexture's red colour channel.
        """
        if base is None or getattr(base,'rgba',None) is None:
            return base
        env_rgb=self._cube_representative_rgb(env_cube)
        if env_rgb is None:
            return base
        rgba=np.asarray(base.rgba,dtype=np.uint8).copy()
        h,w=rgba.shape[:2]
        rgb=rgba[...,:3].astype(np.float32)/255.0
        if mask_tex is not None and getattr(mask_tex,'rgba',None) is not None:
            m=self._sample_repeat_rgba(mask_tex,w,h,1.0)[...,0].astype(np.float32)/255.0
        else:
            m=np.full((h,w),0.35,dtype=np.float32)
        fresnel_gain=1.0
        if fresnel_tex is not None and getattr(fresnel_tex,'rgba',None) is not None:
            f=np.asarray(fresnel_tex.rgba,dtype=np.float32)[...,:3]/255.0
            fresnel_gain=float(np.clip(np.median(f),0.2,1.0))
        amount=np.clip(m*(0.10+0.18*fresnel_gain),0.0,0.28)[...,None]
        rgb=np.clip(rgb*(1.0-amount)+env_rgb.reshape(1,1,3)*amount,0.0,1.0)
        rgba[...,:3]=np.clip(np.round(rgb*255.0),0,255).astype(np.uint8)
        return TextureData(name=name,width=w,height=h,mip_count=1,format_code=-685,rgba=rgba,
                           format_name='Environment preview composite')

    def _material_carpaint_assets(self, mat: MaterialData, mode: str):
        """Resolve texture inputs exactly as the target CarPaint variants name them."""
        paint_rec, paint_tex = self._load_texture(mat.paint_texture)

        if mode == "custom_decal":
            # Target CarPaint_CustomDecal:
            #   paint = texture2D(PaintColor, incident).rgb
            #   decal = texture2D(DecalTexture, uv)
            #   rgb *= mix(paint, decal.rgb, decal.a)
            mask_rec, mask_tex = self._load_texture(mat.decal_texture)
            if mask_tex is None:
                mask_rec, mask_tex = None, self._transparent_decal("Material/NoCustomDecal")
            return paint_rec, paint_tex, paint_rec, paint_tex, mask_rec, mask_tex

        # Target CarPaint_Custom has the same three-channel semantics as the
        # generic vehicle paint shader, but textures come from the material.
        dcol_rec, dcol_tex = self._load_texture(mat.decal_color_texture)
        if dcol_tex is None:
            dcol_rec, dcol_tex = paint_rec, paint_tex
        mask_rec, mask_tex = self._load_texture(mat.decal_texture)
        if mask_tex is None:
            mask_rec, mask_tex = None, self._black_mask("Material/NoDecal")
        return paint_rec, paint_tex, dcol_rec, dcol_tex, mask_rec, mask_tex

    @staticmethod
    def _fallback_carpaint_texture(paint_tex, decal_color_tex, decal_tex, mode: str, name: str):
        """CPU fallback for systems where the GLSL compatibility shader fails."""
        if paint_tex is None:
            return None
        paint_rgb = _representative_rgb(paint_tex, (0.55, 0.55, 0.55))
        if decal_tex is None:
            rgba = np.empty((4, 4, 4), dtype=np.uint8)
            rgba[..., :3] = np.clip(np.round(paint_rgb * 255.0), 0, 255).astype(np.uint8)
            rgba[..., 3] = 255
            return TextureData(name=name, width=4, height=4, mip_count=1,
                               format_code=-53, rgba=rgba, format_name="CarPaintFallback")

        src = np.asarray(decal_tex.rgba, dtype=np.float32) / 255.0
        if mode == "custom_decal":
            # CustomDecal carries authored decal colour and alpha in one map.
            mask = np.clip(src[..., 3:4], 0.0, 1.0)
            decal_rgb = np.clip(src[..., :3], 0.0, 1.0)
        else:
            # v0.43 preview semantics: red-channel mask optionally gated by
            # authored alpha.  This is intentionally different from the later
            # experimental live GLSL path and matches the last correct Studio
            # vehicle rendering pipeline.
            mask = np.clip(src[..., 0:1], 0.0, 1.0)
            if src.shape[2] >= 4 and np.any(src[..., 3] < 0.999):
                mask = mask * np.clip(src[..., 3:4], 0.0, 1.0)
            dcol = _representative_rgb(decal_color_tex, paint_rgb)
            decal_rgb = np.empty((*src.shape[:2], 3), dtype=np.float32)
            decal_rgb[:] = dcol.reshape((1, 1, 3))
        rgb = paint_rgb.reshape((1, 1, 3)) * (1.0 - mask) + decal_rgb * mask
        rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        rgba[..., 3] = 255
        return TextureData(name=name, width=rgba.shape[1], height=rgba.shape[0], mip_count=1,
                           format_code=-53, rgba=rgba, format_name="CarPaintFallback")

    def resolve_mesh(self, mesh_name: str, context: str = "generic", wheel_name: str = "") -> MaterialBinding:
        key = str(mesh_name or "")
        context = str(context or "generic").lower()
        wheel_name = str(wheel_name or "")
        cache_key = self._vehicle_name + "\0" + context + "\0" + wheel_name + "\0" + key
        if cache_key in self._cache:
            return self._cache[cache_key]

        rec = None
        for cand in self._mesh_candidates(mesh_name):
            rec = self.db.resolve_material(cand)
            if rec:
                break
        mat = self.db.material(rec) if rec else None
        notes = []
        if mat is None:
            mat = MaterialData(name=key)
            notes.append("No VuMaterialAsset resolved; using neutral diffuse material.")

        tex_rec = None
        tex = None
        texture_cache_key = None
        carpaint = False
        carpaint_mode = ""
        paint_rec = paint_tex = dcol_rec = dcol_tex = dmask_rec = dmask_tex = None

        # CandyCane_Default uses the fixed-livery stripe shell above. Its
        # material is not a generic customizable CarPaint slot, so supply the
        # red/white reconstructed stripe texture directly.
        if (context == "vehicle" and self._vehicle_name == "CandyCane_Default"
                and self._vehicle_skin_key == "CandyCane_Default"
                and rec is not None and rec.path.endswith("/CandyCane/CandyCane_SkinRainbow")):
            candy = self._candycane_default_texture()
            if candy is not None:
                tex = candy
                texture_cache_key = ("candycane-default-peppermint-v058",)
                notes.append("Peppermint fixed-livery reconstruction from authored CandyCane_Rainbow stripe UVs")

        carpaint_mode = self._carpaint_mode(mat)
        if carpaint_mode == "generic" and context == "wheel":
            wheel_assets = self._wheel_skin_assets(wheel_name)
            # v0.56: restored v0.43 fixed-function vehicle preview.
            carpaint = False
            if wheel_assets is not None:
                paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex, note = wheel_assets
                notes.append(note)
                tex = self._fallback_carpaint_texture(
                    paint_tex, dcol_tex, dmask_tex, "generic",
                    f"WheelPaint/{self._vehicle_name}/{wheel_name}/{key}",
                )
                if tex is not None:
                    texture_cache_key = ("wheel-paint-fallback", self._vehicle_name, wheel_name, key)
            else:
                # Most wheel presets do not define render paint overrides.
                # Keep the material's authored generic paint instead of
                # incorrectly borrowing the current vehicle body's skin.
                paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex = self._material_carpaint_assets(mat, "generic")
                tex = self._fallback_carpaint_texture(
                    paint_tex, dcol_tex, dmask_tex, "generic",
                    f"WheelMaterialPaint/{wheel_name}/{key}",
                )
                notes.append("Wheel material-authored CarPaint (no Wheels render override)")
                if tex is not None:
                    texture_cache_key = ("wheel-material-paint-fallback", wheel_name, key)

        elif carpaint_mode == "generic" and context == "vehicle" and self._vehicle_skin:
            # v0.56: restored v0.43 fixed-function vehicle preview.  Do NOT
            # run the experimental incident-angle GLSL for chassis paint.
            carpaint = False
            paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex, note = self._vehicle_skin_assets()
            notes.append(note)
            # Retain a correct red-channel CPU composite as shader fallback.
            tex, _ = self._build_vehicle_skin()
            if tex is not None:
                effective = self._effective_vehicle_skin()
                texture_cache_key = ("vehicle-skin-fallback", self._vehicle_name,
                                     str(effective.get("Paint Color")),
                                     str(effective.get("Decal Color")),
                                     str(effective.get("Decal")))

        elif carpaint_mode in ("custom", "custom_decal"):
            # Custom CarPaint is materially different from the generic vehicle
            # skin slot: Hot Wheels liveries such as RipRod author their own
            # paint ramp/decal textures in VuMaterialAsset.  Use the live
            # incident-angle shader for these authored materials only.  The
            # generic chassis paint path above remains on the proven v0.43
            # fixed-function/CPU route, so this does not reintroduce the v0.55
            # generic paint regression.
            carpaint = (context == "vehicle")
            paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex = self._material_carpaint_assets(mat, carpaint_mode)
            tex = self._fallback_carpaint_texture(
                paint_tex, dcol_tex, dmask_tex, carpaint_mode,
                f"MaterialCarPaint/{key}",
            )
            if tex is not None:
                texture_cache_key = ("material-carpaint-fallback", key, carpaint_mode)
            if carpaint:
                notes.append(
                    "Live material-authored CarPaint_CustomDecal" if carpaint_mode == "custom_decal"
                    else "Live material-authored CarPaint_Custom"
                )
            else:
                notes.append(
                    "Material-authored CarPaint_CustomDecal" if carpaint_mode == "custom_decal"
                    else "Material-authored CarPaint_Custom"
                )

        elif carpaint_mode == "generic":
            # Generic CarPaint used outside a vehicle body (attachments/effects)
            # keeps its material-authored texture inputs, but uses the same
            # fixed-function fallback as the corrected vehicle paint path.
            carpaint = False
            paint_rec, paint_tex, dcol_rec, dcol_tex, dmask_rec, dmask_tex = self._material_carpaint_assets(mat, "generic")
            tex = self._fallback_carpaint_texture(
                paint_tex, dcol_tex, dmask_tex, "generic", f"MaterialCarPaint/{key}"
            )
            notes.append("Material-authored generic CarPaint")
            if tex is not None:
                texture_cache_key = ("generic-material-carpaint-fallback", key)

        if tex is None and mat.diffuse_texture:
            tex_rec = self.db.resolve_texture(mat.diffuse_texture)
            if tex_rec:
                try:
                    tex = self.db.texture(tex_rec)
                except Exception as exc:
                    notes.append(f"Diffuse texture decode failed: {exc}")
            else:
                notes.append(f"Diffuse texture not found: {mat.diffuse_texture}")

        mask_rec = None
        mask_tex = None
        if mat.mask_texture:
            mask_rec = self.db.resolve_texture(mat.mask_texture)
            if mask_rec:
                try:
                    mask_tex = self.db.texture(mask_rec)
                except Exception as exc:
                    notes.append(f"Mask texture decode failed: {exc}")
            else:
                notes.append(f"Mask texture not found: {mat.mask_texture}")

        detail_rec = detail_tex = None
        if mat.detail_texture:
            detail_rec = self.db.resolve_texture(mat.detail_texture)
            if detail_rec:
                try: detail_tex = self.db.texture(detail_rec)
                except Exception as exc: notes.append(f"Detail texture decode failed: {exc}")
            else: notes.append(f"Detail texture not found: {mat.detail_texture}")

        alpha_rec = alpha_tex = None
        if mat.one_bit_alpha_texture:
            alpha_rec = self.db.resolve_texture(mat.one_bit_alpha_texture)
            if alpha_rec:
                try: alpha_tex = self.db.texture(alpha_rec)
                except Exception as exc: notes.append(f"1-bit alpha texture decode failed: {exc}")

        normal_rec = normal_tex = None
        if mat.normal_texture:
            normal_rec = self.db.resolve_texture(mat.normal_texture)
            if normal_rec:
                try: normal_tex = self.db.texture(normal_rec)
                except Exception as exc: notes.append(f"Normal texture decode failed: {exc}")
            else: notes.append(f"Normal texture not found: {mat.normal_texture}")

        fresnel_rec = fresnel_tex = None
        if mat.fresnel_texture:
            fresnel_rec = self.db.resolve_texture(mat.fresnel_texture)
            if fresnel_rec:
                try: fresnel_tex = self.db.texture(fresnel_rec)
                except Exception as exc: notes.append(f"Fresnel texture decode failed: {exc}")
            else: notes.append(f"Fresnel texture not found: {mat.fresnel_texture}")

        env_rec, env_cube = self._load_cube_texture(mat.env_texture) if mat.env_texture else (None, None)
        add_env_rec, add_env_cube = self._load_cube_texture(mat.additive_env_texture) if mat.additive_env_texture else (None, None)
        if mat.env_texture and env_rec is None: notes.append(f"Environment cubemap not found: {mat.env_texture}")
        if mat.additive_env_texture and add_env_rec is None: notes.append(f"Additive environment cubemap not found: {mat.additive_env_texture}")

        if tex is None and not mat.diffuse_texture and env_cube is None and detail_tex is not None:
            tex_rec, tex = detail_rec, detail_tex
            texture_cache_key = ("detail-as-base", detail_rec.archive_index, detail_rec.entry_index) if detail_rec else None
            notes.append("DetailTexture used as base fallback because material has no diffuse/cubemap colour source")
            detail_tex = None

        dynamic_environment=(context == "environment" and env_cube is not None and
            str(mat.shader or "").strip().lower() in (
                "art/world/emissiveenvmapnodetail", "art/world/envmapnodetail",
                "art/color/vcenvmap", "art/nofog/vcenvmap"))
        if context == "environment" and tex is not None and not dynamic_environment:
            combined = self._environment_preview_texture(
                tex, env_cube or add_env_cube, mask_tex, fresnel_tex,
                f"EnvironmentPreview/{key}",
            )
            if combined is not tex:
                tex = combined
                texture_cache_key = ("environment-preview", key,
                    getattr(env_rec or add_env_rec,"archive_index",-1), getattr(env_rec or add_env_rec,"entry_index",-1))
                notes.append("Environment cubemap approximated in fixed-function preview")

        source = rec.path if rec else "editor neutral fallback"
        binding = MaterialBinding(
            mesh_name=key, material_record=rec, material=mat,
            diffuse_texture_record=tex_rec, diffuse_texture=tex,
            mask_texture_record=mask_rec, mask_texture=mask_tex,
            mask_cache_key=(("asset", mask_rec.archive_index, mask_rec.entry_index) if mask_rec else None),
            source=source, notes=notes, texture_cache_key=texture_cache_key,
            detail_texture_record=detail_rec, detail_texture=detail_tex,
            detail_cache_key=(("asset", detail_rec.archive_index, detail_rec.entry_index) if detail_rec else None),
            detail_uv_scale=float(getattr(mat, "detail_uv_scale", 1.0) or 1.0),
            alpha_texture_record=alpha_rec, alpha_texture=alpha_tex,
            env_cube_record=env_rec, env_cube=env_cube,
            env_cube_cache_key=(("cube", env_rec.archive_index, env_rec.entry_index) if env_rec else None),
            additive_env_cube_record=add_env_rec, additive_env_cube=add_env_cube,
            additive_env_cube_cache_key=(("cube", add_env_rec.archive_index, add_env_rec.entry_index) if add_env_rec else None),
            normal_texture_record=normal_rec, normal_texture=normal_tex,
            normal_cache_key=(("asset", normal_rec.archive_index, normal_rec.entry_index) if normal_rec else None),
            fresnel_texture_record=fresnel_rec, fresnel_texture=fresnel_tex,
            fresnel_cache_key=(("asset", fresnel_rec.archive_index, fresnel_rec.entry_index) if fresnel_rec else None),
            dynamic_environment=dynamic_environment,
            carpaint=carpaint,
            paint_texture_record=paint_rec, paint_texture=paint_tex,
            decal_color_texture_record=dcol_rec, decal_color_texture=dcol_tex,
            decal_mask_texture_record=dmask_rec, decal_mask_texture=dmask_tex,
            paint_cache_key=(("asset", paint_rec.archive_index, paint_rec.entry_index) if paint_rec else ("vehicle-paint", self._vehicle_name)),
            decal_color_cache_key=(("asset", dcol_rec.archive_index, dcol_rec.entry_index) if dcol_rec else ("vehicle-decal-color", self._vehicle_name)),
            decal_mask_cache_key=(("asset", dmask_rec.archive_index, dmask_rec.entry_index) if dmask_rec else ("vehicle-no-decal", self._vehicle_name)),
            carpaint_mode=carpaint_mode if carpaint else "",
            uses_vertex_color=(self._shader_uses_vertex_color(mat.shader) if context not in ("vehicle", "wheel") else False),
        )
        self._cache[cache_key] = binding
        return binding

    def report(self, mesh_name: str) -> str:
        b = self.resolve_mesh(mesh_name)
        m = b.material
        lines = [
            f"Mesh: {mesh_name}",
            f"Material source: {b.source}",
            f"Shader: {m.shader or '(none)'}",
            f"Diffuse color: {np.asarray(m.diffuse_color).tolist()}",
            f"Diffuse texture: {m.diffuse_texture or '(none)'}",
        ]
        if b.carpaint:
            lines.append(f"CarPaint mode: {b.carpaint_mode}")
            if b.carpaint_mode == "generic":
                lines.append("CarPaint path: VehiclePaintColor + VehicleDecalColor + VehicleDecalTexture.r")
                lines.append(f"Vehicle skin row: {self._vehicle_skin}")
            elif b.carpaint_mode == "custom":
                lines.append("CarPaint path: material PaintColor + DecalColor + DecalTexture.r")
            elif b.carpaint_mode == "custom_decal":
                lines.append("CarPaint path: material PaintColor + DecalTexture.rgb/a")
            for label, rec, tex in (
                ("Paint ramp", b.paint_texture_record, b.paint_texture),
                ("Decal color ramp", b.decal_color_texture_record, b.decal_color_texture),
                ("Decal mask", b.decal_mask_texture_record, b.decal_mask_texture),
            ):
                if rec:
                    lines.append(f"{label}: {rec.path} [{rec.source_name}]")
                elif tex:
                    lines.append(f"{label}: {tex.name}")
        if b.diffuse_texture_record:
            t = b.diffuse_texture
            lines.append(f"Texture source: {b.diffuse_texture_record.path} [{b.diffuse_texture_record.source_name}]")
            if t:
                lines.append(f"Texture format: {t.format_name} ({t.width}x{t.height}, {t.mip_count} mips)")
        if b.notes:
            lines += ["Notes:"] + ["  " + x for x in b.notes]
        return "\n".join(lines)
