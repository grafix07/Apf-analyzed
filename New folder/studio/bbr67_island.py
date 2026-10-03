from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord
from bbr50_entities import TemplateExpander, walk_entities
from vector_formats import ModelData, TextureData


@dataclass
class IslandSceneModel:
    name: str
    model_ref: str
    model: ModelData
    matrix: np.ndarray
    source: str
    animated: bool = False
    layer: str = "Environment"
    skybox: bool = False


@dataclass
class IslandScenePoint:
    name: str
    entity_type: str
    position: np.ndarray
    matrix: np.ndarray
    source: str
    label: str = ""


@dataclass
class IslandSceneBox:
    name: str
    matrix: np.ndarray
    source: str


@dataclass
class IslandWaterSurface:
    name: str
    matrix: np.ndarray
    size_x: float
    size_y: float
    source: str
    water_map_ref: str = ""
    water_map: Optional[TextureData] = None
    decal_texture_ref: str = ""
    decal_texture: Optional[TextureData] = None
    foam_texture_ref: str = ""
    foam_texture: Optional[TextureData] = None
    surface_type: str = "Standard"
    surface_mask: int = 1
    reflection_cube_ref: str = ""
    procedural_reflection: bool = True
    fog_enabled: bool = False
    foam_ramp_speed: float = 0.08
    foam_ramp_frequency: float = 3.0
    wave_height_multiplier: float = 0.04
    wave_direction_deg: float = 90.0
    wave_wind_speed: float = 8.0
    wave_complexity: float = 6.0
    wave_suppression: float = 0.0


@dataclass
class IslandTrackDecal:
    name: str
    matrix: np.ndarray
    material_ref: str
    texture_record: AssetRecord
    texture: TextureData
    projected_triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))


@dataclass
class IslandSceneSpec:
    name: str
    kind: str
    project_path: str
    project_source: str
    setting_name: str = ""
    cube_macros: Dict[str, str] = field(default_factory=dict)
    clear_color: Optional[np.ndarray] = None
    ambient_color: Optional[np.ndarray] = None
    directional_color: Optional[np.ndarray] = None
    models: List[IslandSceneModel] = field(default_factory=list)
    track_segments: List[List[np.ndarray]] = field(default_factory=list)
    collision_segments: List[List[np.ndarray]] = field(default_factory=list)
    out_of_bounds_boxes: List[IslandSceneBox] = field(default_factory=list)
    spawn_points: List[IslandScenePoint] = field(default_factory=list)
    powerup_points: List[IslandScenePoint] = field(default_factory=list)
    pfx_points: List[IslandScenePoint] = field(default_factory=list)
    camera_points: List[IslandScenePoint] = field(default_factory=list)
    sector_points: List[IslandScenePoint] = field(default_factory=list)
    water_surfaces: List[IslandWaterSurface] = field(default_factory=list)
    track_decals: List[IslandTrackDecal] = field(default_factory=list)
    total_renderable_entities: int = 0
    skipped_detail_entities: int = 0
    unresolved_models: List[str] = field(default_factory=list)
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    start_focus: Optional[np.ndarray] = None

    def model_count(self, layer: str) -> int:
        return sum(1 for x in self.models if x.layer == layer)

    def report(self) -> str:
        span = self.bounds_max - self.bounds_min
        lines = [
            f"Map scene: {self.name}",
            f"Kind: {self.kind}",
            f"Project: {self.project_path}",
            f"Source archive: {self.project_source}",
            f"Track setting: {self.setting_name or '(project default)'}",
            f"Cube-image macros: " + (", ".join(f"{k}→{v}" for k,v in self.cube_macros.items()) if self.cube_macros else "(none)"),
            f"Renderable model instances: {len(self.models)} / {self.total_renderable_entities}",
            f"  Environment / track: {self.model_count('Environment')}",
            f"  Camera-follow skyboxes: {sum(1 for x in self.models if x.skybox)}",
            f"  Animated props: {self.model_count('Animated')}",
            f"  Gameplay / alternate-mode: {self.model_count('Gameplay')}",
            f"Track-path segments: {len(self.track_segments)}",
            f"Collision-wall polylines: {len(self.collision_segments)}",
            f"Out-of-bounds volumes: {len(self.out_of_bounds_boxes)}",
            f"Starting-grid spawns: {len(self.spawn_points)}",
            f"Power-up positions: {len(self.powerup_points)}",
            f"Static PFX markers: {len(self.pfx_points)}",
            f"Camera/cinematic markers: {len(self.camera_points)}",
            f"Track sectors: {len(self.sector_points)}",
            f"Water surfaces: {len(self.water_surfaces)}",
            f"Water profiles: " + (", ".join(sorted({f"{w.surface_type}/mask{w.surface_mask}" for w in self.water_surfaces})) if self.water_surfaces else "(none)"),
            f"Hot Wheels track decals: {len(self.track_decals)}",
            f"Bounds min: {self.bounds_min.tolist()}",
            f"Bounds max: {self.bounds_max.tolist()}",
            f"Scene span: {span.tolist()}",
        ]
        if self.start_focus is not None:
            lines.append(f"Initial focus: {self.start_focus.tolist()}")
        if self.skipped_detail_entities:
            lines.append(f"Small-detail career-map model instances skipped for overview: {self.skipped_detail_entities}")
        if self.unresolved_models:
            uniq=[]
            for x in self.unresolved_models:
                if x not in uniq:
                    uniq.append(x)
            lines += ["", f"Unresolved model references ({len(uniq)} unique):"]
            lines += ["  " + x for x in uniq[:40]]
            if len(uniq) > 40:
                lines.append(f"  ... {len(uniq)-40} more")
        return "\n".join(lines)


class IslandAdventureResolver:
    """Read-only Island Adventure project/track scene resolver.

    v0.68.1 expands the complete authored project and separates the result into
    renderable layers and developer overlays.  Unlike v0.67, model-bearing
    entities under Modes/HiddenItems are retained as an optional Gameplay layer
    instead of being discarded before the map reaches the viewport.
    """

    _DRAW_COMPONENTS = (
        "Vu3dDrawStaticModelComponent",
        "Vu3dDrawAnimatedModelComponent",
        "Vu3dDrawRagdollComponent",
        "Vu3dDrawBreakableModelComponent",
    )

    def __init__(self, db: AssetDatabase):
        self.db = db
        self.expander = TemplateExpander(db)
        self.track_rows = db.spreadsheet_by_name("Tracks")

    @property
    def available(self) -> bool:
        return bool(getattr(self.db, "has_map_tracks", False))

    def tracks(self) -> List[str]:
        prefix = "VuProjectAsset/Tracks/"
        out = []
        for path in self.db.records_by_path:
            if path.startswith(prefix):
                out.append(path[len(prefix):])
        return sorted(out, key=str.lower)

    def has_career_map(self) -> bool:
        return self.db.get("VuProjectAsset/Screens_Premium/Career_Map") is not None

    @staticmethod
    def _natural_key(text: str):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]

    @staticmethod
    def _component_model_ref(view) -> Tuple[str, bool]:
        comps = view.data.get("Components") if isinstance(view.data, dict) else None
        if isinstance(comps, dict):
            for comp_name in IslandAdventureResolver._DRAW_COMPONENTS:
                comp = comps.get(comp_name)
                if not isinstance(comp, dict):
                    continue
                props = comp.get("Properties")
                if not isinstance(props, dict):
                    props = {}
                ref = props.get("Model Asset") or props.get("Model")
                if isinstance(ref, str) and ref:
                    return ref, comp_name == "Vu3dDrawAnimatedModelComponent"
        # Several environment entities (notably VuSkyBoxEntity in lighting
        # setting templates) store Model Asset directly in entity Properties
        # and have no Components dictionary.  The old early return skipped
        # those models entirely.
        p = view.properties
        ref = p.get("Model Asset") or p.get("Model")
        return (str(ref), False) if isinstance(ref, str) and ref else ("", False)

    @staticmethod
    def _model_layer(path: str, animated: bool) -> str:
        p = "/" + str(path or "").replace("\\", "/").strip("/") + "/"
        # Skybox geometry is commonly nested under Lighting/<setting>; it is
        # authored environment, not optional gameplay content.
        if "skybox" in p.lower():
            return "Environment"
        if any(x in p for x in ("/Modes/", "/HiddenItems/", "/HiddenItems_HW/", "/lights/")):
            return "Gameplay"
        if animated:
            return "Animated"
        return "Environment"

    @staticmethod
    def _is_skybox_entity(view, model_ref: str) -> bool:
        """Return True for authored camera-follow sky/background geometry.

        VuSkyBoxEntity frequently points at an ordinary cube/dome model, so the
        model reference alone is not reliable. Some packages instead name the
        template/path SkyBox or SkyDome. Keep the test deliberately narrow so
        scenery with generic "sky" material names is not reclassified.
        """
        typ = str(getattr(view, "type", "") or "").lower()
        name = str(getattr(view, "name", "") or "").lower()
        path = "/" + str(getattr(view, "path", "") or "").replace("\\", "/").strip("/").lower() + "/"
        ref = "/" + str(model_ref or "").replace("\\", "/").strip("/").lower() + "/"
        if "vuskyboxentity" in typ:
            return True
        tokens = ("skybox", "sky_box", "skydome", "sky_dome")
        return any(t in name or t in path or t in ref for t in tokens)

    @staticmethod
    def _model_world_bounds(model: ModelData, matrix: np.ndarray):
        mn, mx = model.bounds()
        corners = np.asarray([
            (x, y, z, 1.0)
            for x in (float(mn[0]), float(mx[0]))
            for y in (float(mn[1]), float(mx[1]))
            for z in (float(mn[2]), float(mx[2]))
        ], dtype=np.float32)
        q = (np.asarray(matrix, dtype=np.float32) @ corners.T).T[:, :3]
        return q.min(axis=0), q.max(axis=0)

    def _collect_track_segments(self, views) -> List[List[np.ndarray]]:
        groups: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        all_nodes = []
        preferred = []
        for v in views:
            if v.type != "VuTrackNodeEntity":
                continue
            path = str(v.path or "").replace("\\", "/")
            row = (str(v.parent_path or ""), v.name, v.world_matrix[:3, 3].astype(np.float32).copy())
            all_nodes.append(row)
            if "/Modes/Track/Track/" in path:
                preferred.append(row)
        for parent, name, pos in (preferred or all_nodes):
            groups.setdefault(parent, []).append((name, pos))
        out = []
        for parent in sorted(groups, key=self._natural_key):
            rows = sorted(groups[parent], key=lambda x: self._natural_key(x[0]))
            pts = [p for _, p in rows]
            if len(pts) >= 2:
                out.append(pts)
        return out

    def _collect_collision_segments(self, views) -> List[List[np.ndarray]]:
        groups: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for v in views:
            if v.type not in ("VuCollisionWallEdgeEntity", "VuCollisionCeilingEdgeEntity"):
                continue
            parent = str(v.parent_path or "")
            groups.setdefault(parent, []).append((v.name, v.world_matrix[:3, 3].astype(np.float32).copy()))
        out=[]
        for parent in sorted(groups, key=self._natural_key):
            rows=sorted(groups[parent], key=lambda x:self._natural_key(x[0]))
            pts=[p for _,p in rows]
            if len(pts)>=2:
                out.append(pts)
        return out

    @staticmethod
    def _point(v, label: str = "") -> IslandScenePoint:
        return IslandScenePoint(
            v.name or v.type,
            v.type,
            v.world_matrix[:3, 3].astype(np.float32).copy(),
            v.world_matrix.astype(np.float32).copy(),
            str(v.path or ""),
            label,
        )

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _collect_overlays(self, views, spec: IslandSceneSpec):
        # The game authors water motion separately from the water surface.
        # VuInfiniteOceanWaveEntity supplies a profile selected by Surface Mask:
        # 1 = standard ocean, 2 = small/pool water, 4 = lava.  Earlier Studio
        # builds ignored this and used one hard-coded sine wave on every map.
        wave_profiles: Dict[int, Dict[str, float]] = {}
        for v in views:
            typ = str(v.type or "")
            if typ != "VuInfiniteOceanWaveEntity":
                continue
            p = v.properties
            try:
                mask = int(float(p.get("Surface Mask", 1)))
            except Exception:
                mask = 1
            wave_profiles[mask] = {
                "height": self._safe_float(p.get("Height Multiplier"), 0.04),
                "direction": self._safe_float(p.get("Wave Direction"), 90.0),
                "wind": self._safe_float(p.get("Wind Speed"), 8.0),
                "complexity": self._safe_float(p.get("Complexity"), 6.0),
                "suppression": self._safe_float(p.get("Suppression Wave Length"), 0.0),
            }

        for v in views:
            path = str(v.path or "").replace("\\", "/")
            typ = str(v.type or "")
            if typ == "VuOutOfBoundsEntity":
                spec.out_of_bounds_boxes.append(IslandSceneBox(v.name or typ, v.world_matrix.astype(np.float32).copy(), path))
            elif typ == "VuSpawnPointEntity" and "/StartingGrid" in path:
                spec.spawn_points.append(self._point(v, "Starting grid"))
            elif typ == "VuPowerUpEntity":
                spec.powerup_points.append(self._point(v, "Power-up"))
            elif typ == "VuStaticPfxEntity":
                label=str(v.properties.get("Effect Name") or "PFX")
                spec.pfx_points.append(self._point(v, label))
            elif typ in ("VuTestCameraEntity", "VuCinematicEntity") or typ.endswith("CameraEntity"):
                spec.camera_points.append(self._point(v, typ))
            elif typ == "VuTrackSectorEntity" and "/Modes/Track/Track/" in path:
                spec.sector_points.append(self._point(v, "Track sector"))
            elif typ == "VuDecalEntity":
                material_ref=str(v.properties.get("Material") or "").strip()
                # HW track arrows are material-driven projector decals. They
                # have no model/mesh and were absent from every map preview.
                if not material_ref.startswith("Building/Decals/HW_TrackArrow"):
                    continue
                try:
                    material=self.db.material(material_ref)
                    texture_rec=(self.db.resolve_texture(material.diffuse_texture)
                                 if material is not None else None)
                    texture=self.db.texture(texture_rec) if texture_rec else None
                except (ValueError, RuntimeError, OSError):
                    continue
                if texture_rec is not None and texture is not None:
                    spec.track_decals.append(IslandTrackDecal(
                        v.name or typ, v.world_matrix.astype(np.float32).copy(),
                        material_ref, texture_rec, texture))
            elif typ == "VuGameWaterSurfaceEntity":
                p=v.properties
                map_ref=str(p.get("WaterMap") or "").strip()
                decal_ref=str(p.get("DecalTextureAsset") or "").strip()
                foam_ref=str(p.get("FoamTextureAsset") or "Water/Foam").strip()
                try: water_map=self.db.water_map(map_ref) if map_ref else None
                except Exception: water_map=None
                try: decal=self.db.texture(decal_ref) if decal_ref else None
                except Exception: decal=None
                try: foam=self.db.texture(foam_ref) if foam_ref else None
                except Exception: foam=None
                sy=max(1.0,self._safe_float(p.get("Y Size"),1.0))
                # CityC omits X Size but has a 256x128 water map and Y Size 128.
                # Reconstruct its width from the authored map aspect ratio.
                default_x=sy*water_map.width/water_map.height if water_map else sy
                sx=max(1.0,self._safe_float(p.get("X Size"),default_x))

                surface_type=str(p.get("Type") or "Standard").strip() or "Standard"
                default_mask={"standard":1,"small":2,"lava":4}.get(surface_type.lower(),1)
                try: surface_mask=int(float(p.get("Surface Mask",default_mask)))
                except Exception: surface_mask=default_mask
                wave=(wave_profiles.get(surface_mask) or wave_profiles.get(1) or {
                    "height":0.04,"direction":90.0,"wind":8.0,"complexity":6.0,"suppression":0.0
                })
                reflection_ref=str(p.get("ReflectionCubeTextureAsset") or "").strip()
                procedural=p.get("ProceduralReflection",True)
                if isinstance(procedural,str):
                    procedural=procedural.strip().lower() not in ("0","false","no","off")
                fog=p.get("FogEnabled",False)
                if isinstance(fog,str):
                    fog=fog.strip().lower() in ("1","true","yes","on")

                spec.water_surfaces.append(IslandWaterSurface(
                    v.name or typ,v.world_matrix.astype(np.float32).copy(),sx,sy,path,
                    map_ref,water_map,decal_ref,decal,foam_ref,foam,
                    surface_type=surface_type, surface_mask=surface_mask,
                    reflection_cube_ref=reflection_ref, procedural_reflection=bool(procedural),
                    fog_enabled=bool(fog),
                    foam_ramp_speed=self._safe_float(p.get("Foam Ramp Speed"),0.08),
                    foam_ramp_frequency=self._safe_float(p.get("Foam Ramp Frequency"),3.0),
                    wave_height_multiplier=float(wave.get("height",0.04)),
                    wave_direction_deg=float(wave.get("direction",90.0)),
                    wave_wind_speed=float(wave.get("wind",8.0)),
                    wave_complexity=float(wave.get("complexity",6.0)),
                    wave_suppression=float(wave.get("suppression",0.0)),
                ))

    @staticmethod
    def _clip_decal_polygon(polygon, axis: int, bound: float, upper: bool):
        """Clip a road triangle against one side of an authored decal box."""
        result=[]
        for a,b in zip(polygon, polygon[1:]+polygon[:1]):
            da=(bound-a[axis]) if upper else (a[axis]-bound)
            db=(bound-b[axis]) if upper else (b[axis]-bound)
            inside_a=da>=0.0; inside_b=db>=0.0
            if inside_a:result.append(a)
            if inside_a!=inside_b:
                result.append(a+(b-a)*(da/(da-db)))
        return result

    def _project_track_decals(self, spec: IslandSceneSpec, model_bounds):
        """Follow track curves/slopes instead of floating a flat arrow quad."""
        # The short upside-down arrow projectors have a 0.68 Z scale while
        # the track's lower road face is ~1.35 world units below the upper one.
        # Their local clip depth therefore has to reach roughly 2.0.
        depth=2.25
        corners=np.asarray([(x,y,z) for x in (-.5,.5) for y in (-.5,.5)
                            for z in (-depth,depth)],dtype=np.float32)
        for decal in spec.track_decals:
            matrix=decal.matrix
            try:inverse=np.linalg.inv(matrix[:3,:3])
            except np.linalg.LinAlgError:continue
            box=corners@matrix[:3,:3].T+matrix[:3,3]
            box_min=box.min(axis=0);box_max=box.max(axis=0)
            projected=[]
            for item,mn,mx in model_bounds:
                if np.any(mx<box_min) or np.any(mn>box_max):continue
                for mesh in item.model.meshes:
                    if not mesh.name.startswith(("Building/HW/HW_Track_","Building/HW/HW_Booster")):
                        continue
                    world=mesh.positions@item.matrix[:3,:3].T+item.matrix[:3,3]
                    local=(world-matrix[:3,3])@inverse.T
                    triangles=np.asarray(mesh.indices,dtype=np.int64).reshape((-1,3))
                    if not len(triangles):continue
                    q=local[triangles]
                    keep=np.all(q.min(axis=1)<=np.asarray((.5,.5,depth)),axis=1)
                    keep &=np.all(q.max(axis=1)>=np.asarray((-.5,-.5,-depth)),axis=1)
                    for indices,tri in zip(triangles[keep],q[keep]):
                        p=world[indices]
                        if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),matrix[:3,2])<=0.0:
                            continue
                        polygon=list(tri)
                        for axis,low,high in ((0,-.5,.5),(1,-.5,.5),(2,-depth,depth)):
                            polygon=self._clip_decal_polygon(polygon,axis,low,False)
                            if len(polygon)<3:break
                            polygon=self._clip_decal_polygon(polygon,axis,high,True)
                            if len(polygon)<3:break
                        for i in range(1,len(polygon)-1):
                            projected.extend((polygon[0],polygon[i],polygon[i+1]))
            if projected:
                decal.projected_triangles=np.asarray(projected,dtype=np.float32).reshape((-1,3))

    def _collect_environment_settings(self, views, spec: IslandSceneSpec):
        """Recover the authored track lighting and cube-image macro state.

        Vector track-setting variants contain VuSetCubeImageMacroEntity nodes
        that replace Proxy_cube / Proxy_additive with the real environment
        cubemaps.  Without applying those macros reflective materials render as
        generic/white placeholders in the Studio.
        """
        ambient_candidates=[]
        directional_candidates=[]
        for v in views:
            typ=str(v.type or "")
            props=v.properties if isinstance(v.properties,dict) else {}
            if typ == "VuSetCubeImageMacroEntity":
                macro=str(props.get("Macro Name") or "").strip()
                cube=str(props.get("Cube Texture") or "").strip()
                if macro and cube:
                    spec.cube_macros[macro]=cube
            elif typ == "VuGlobalGfxSettingsEntity":
                c=props.get("Clear Color")
                if isinstance(c,(list,tuple)) and len(c)>=4:
                    spec.clear_color=np.asarray(c[:4],dtype=np.float32)
            elif typ == "VuAmbientLightEntity":
                c=props.get("Color")
                if isinstance(c,(list,tuple)) and len(c)>=4:
                    # High/default light wins over the explicitly Low variant.
                    priority=0 if "low" in str(v.name or "").lower() else 1
                    ambient_candidates.append((priority,np.asarray(c[:4],dtype=np.float32)))
            elif typ == "VuDirectionalLightEntity":
                c=props.get("Front Color") or props.get("Color")
                if isinstance(c,(list,tuple)) and len(c)>=4:
                    priority=0 if "low" in str(v.name or "").lower() else 1
                    directional_candidates.append((priority,np.asarray(c[:4],dtype=np.float32)))
        if ambient_candidates:
            spec.ambient_color=sorted(ambient_candidates,key=lambda x:x[0])[-1][1]
        if directional_candidates:
            spec.directional_color=sorted(directional_candidates,key=lambda x:x[0])[-1][1]

    def _resolve(self, rec: AssetRecord, kind: str, display_name: str,
                 career_overview_min_size: float = 100.0, setting_name: str = "") -> IslandSceneSpec:
        root = self.expander.expand_project(rec)
        views = list(walk_entities(root))
        spec = IslandSceneSpec(display_name, kind, rec.path, rec.source_name, setting_name=setting_name)
        self._collect_environment_settings(views, spec)
        model_cache: Dict[str, Optional[ModelData]] = {}
        world_mins=[]; world_maxs=[]; model_bounds=[]

        for v in views:
            ref, animated = self._component_model_ref(v)
            if not ref:
                continue
            spec.total_renderable_entities += 1

            if ref not in model_cache:
                mrec = self.db.resolve_model(ref)
                try:
                    model_cache[ref] = self.db.model(mrec) if mrec else None
                except Exception:
                    model_cache[ref] = None
            model = model_cache[ref]
            if model is None or not model.meshes:
                spec.unresolved_models.append(ref)
                continue

            if kind == "Career Map" and career_overview_min_size > 0:
                mn, mx = model.bounds()
                diag = float(np.linalg.norm(np.asarray(mx) - np.asarray(mn)))
                if diag < career_overview_min_size:
                    spec.skipped_detail_entities += 1
                    continue

            skybox = self._is_skybox_entity(v, ref)
            item = IslandSceneModel(
                v.name or ref.rsplit("/", 1)[-1], ref, model,
                v.world_matrix.astype(np.float32).copy(), v.path, animated,
                self._model_layer(v.path, animated), skybox,
            )
            spec.models.append(item)

            # Skyboxes are camera-follow backgrounds in the game. Treating
            # their authored cube/dome as world geometry caused the blue
            # rounded block seen in every track and also polluted Fit Full Map
            # bounds. Keep them renderable, but out of scene bounds/decal
            # projection so only real track/map geometry drives the camera.
            if skybox:
                continue
            wmn, wmx = self._model_world_bounds(model, item.matrix)
            world_mins.append(wmn); world_maxs.append(wmx)
            model_bounds.append((item,wmn,wmx))

        if world_mins:
            spec.bounds_min = np.min(np.asarray(world_mins), axis=0).astype(np.float32)
            spec.bounds_max = np.max(np.asarray(world_maxs), axis=0).astype(np.float32)
        if kind == "Track":
            spec.track_segments = self._collect_track_segments(views)
            spec.collision_segments = self._collect_collision_segments(views)
        self._collect_overlays(views, spec)
        if spec.track_decals:
            self._project_track_decals(spec,model_bounds)

        # Start the camera where the player actually starts rather than fitting
        # a kilometre-scale scene into one distant view.  Fall back to the first
        # normal-route node, then the scene centre.
        if spec.spawn_points:
            a=np.asarray([x.position for x in spec.spawn_points],dtype=np.float32)
            spec.start_focus=a.mean(axis=0).astype(np.float32)
        elif spec.track_segments and spec.track_segments[0]:
            spec.start_focus=np.asarray(spec.track_segments[0][0],dtype=np.float32).copy()
        elif world_mins:
            spec.start_focus=((spec.bounds_min+spec.bounds_max)*0.5).astype(np.float32)
        return spec

    def resolve_track(self, name: str) -> IslandSceneSpec:
        requested=str(name or "")
        rec = self.db.resolve_project("Tracks/" + requested)
        if rec is None:
            raise KeyError(f"Track project not found: {name}")

        # When the user chooses the canonical/base track, load the authored
        # setting variant used by the game (for example AlienA_NightB).  Those
        # projects include skybox/lighting and the exact Proxy_cube mappings.
        setting_name=""
        row=self.track_rows.get(requested) if isinstance(self.track_rows,dict) else None
        if isinstance(row,dict):
            setting_name=str(row.get("Settings") or "").strip()
            if setting_name:
                variant=self.db.get(f"VuProjectAsset/Tracks/{requested}_{setting_name}")
                if variant is not None:
                    rec=variant
        return self._resolve(rec, "Track", requested, setting_name=setting_name)

    def resolve_career_map(self) -> IslandSceneSpec:
        rec = self.db.get("VuProjectAsset/Screens_Premium/Career_Map")
        if rec is None:
            raise KeyError("Island Adventure Career Map project is not present in this package")
        return self._resolve(rec, "Career Map", "Career Map", career_overview_min_size=100.0)
