from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord
from bbr50_entities import EntityView, TemplateExpander, Transform, walk_entities
from vector_formats import ModelData, MeshData, bind_globals


@dataclass
class SourceValue:
    value: Any
    source: str


@dataclass
class ModelSpec:
    name: str
    role: str
    model_ref: str
    model: ModelData
    matrix: np.ndarray
    source: str
    visible: bool = True


@dataclass
class WheelSpec:
    name: str
    axle: str
    side: str
    wheel_name: str
    wheel_bone: str
    diameter: float
    width: float
    offset_x: float
    pfx_ref: str
    matrix: np.ndarray
    model: ModelData
    component_refs: List[str]
    source: str


@dataclass
class DriverAttachmentSpec:
    name: str
    model_ref: str
    model: ModelData
    bone: str
    matrix: np.ndarray
    source: str


@dataclass
class EmitterSpec:
    name: str
    kind: str
    effect_ref: str
    matrix: np.ndarray
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture_name: str = ""
    texture_size: float = 1.0
    source: str = ""
    enabled: bool = True
    alt_effect_ref: str = ""


@dataclass
class VehicleSpec:
    name: str
    project_path: str
    project_source: str
    expanded_root: Dict[str, Any]
    models: List[ModelSpec] = field(default_factory=list)
    wheels: List[WheelSpec] = field(default_factory=list)
    emitters: List[EmitterSpec] = field(default_factory=list)
    driver_attachments: List[DriverAttachmentSpec] = field(default_factory=list)
    tuning: Dict[str, Any] = field(default_factory=dict)
    skin: Dict[str, Any] = field(default_factory=dict)
    skin_name: str = "Default"
    skin_key: str = ""
    engine: Dict[str, Any] = field(default_factory=dict)
    suspension: Dict[str, Any] = field(default_factory=dict)
    hull: Dict[str, Any] = field(default_factory=dict)
    rigid_body: Dict[str, Any] = field(default_factory=dict)
    driver_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    driving_anim_set: str = "Standard"
    driver_source: str = ""
    wheelbase: float = 0.0
    track_width: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def source_report(self) -> str:
        lines = [
            f"Vehicle: {self.name}",
            f"Project: {self.project_path}",
            f"Project source: {self.project_source}",
            "",
            "Authored vehicle tuning:",
        ]
        for k in (
            "Speed", "Accel Factor", "Steering Lag", "Steering", "Mass", "Traction",
            "Drag Coeff", "Induced Power Slide Traction Factor", "Induced Power Slide Coeff",
            "Power Slide Coeff", "Buffed Power Slide Coeff", "Upper Spring Coeff",
            "Lower Spring Coeff", "Compression Damping Coeff", "Rebound Damping Coeff",
            "Rollover Resistance", "Wheelie Resistance",
        ):
            if k in self.tuning:
                lines.append(f"  {k}: {self.tuning[k]}")
        if self.skin:
            lines += ["", f"Vehicle skin: {self.skin_name}", f"  Paint Color: {self.skin.get('Paint Color')}", f"  Decal Color: {self.skin.get('Decal Color')}", f"  Decal: {self.skin.get('Decal')}"]
        if self.engine:
            lines += ["", "Engine params:"] + [f"  {k}: {v}" for k, v in self.engine.items()]
        if self.suspension:
            lines += ["", "Suspension params:"] + [f"  {k}: {v}" for k, v in self.suspension.items()]
        lines += ["", f"Driver mount: {self.driver_matrix[:3,3].tolist()} [{self.driver_source or 'template'}]",
                  f"Driving animation set: {self.driving_anim_set}",
                  f"Wheelbase from authored wheel bones: {self.wheelbase:.4f}",
                  f"Track width from authored wheel bones: {self.track_width:.4f}"]
        lines += ["", "Scene models:"]
        for m in self.models:
            p = m.matrix[:3, 3]
            lines.append(f"  {m.role}: {m.model_ref} @ ({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) [{m.source}]")
        lines += ["", "Wheels:"]
        for w in self.wheels:
            p = w.matrix[:3, 3]
            lines.append(
                f"  {w.name}: bone={w.wheel_bone} diameter={w.diameter:g} width={w.width:g} "
                f"pfx={w.pfx_ref} @ ({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})"
            )
        if self.warnings:
            lines += ["", "Warnings:"] + ["  " + x for x in self.warnings]
        return "\n".join(lines)


def _f(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _mask_semantics(props: Dict[str, Any], name: str) -> Tuple[str, str]:
    mask = props.get("Mask") if isinstance(props.get("Mask"), dict) else {}
    nl = (name or "").lower()
    if mask.get("Front") or "front" in nl:
        axle = "front"
    elif mask.get("Middle") or "middle" in nl or "mid" in nl:
        axle = "middle"
    elif mask.get("Rear") or "rear" in nl:
        axle = "rear"
    else:
        axle = "unknown"
    if mask.get("Left") or "left" in nl:
        side = "left"
    elif mask.get("Right") or "right" in nl:
        side = "right"
    else:
        side = "center"
    return axle, side


def _mesh_component_model(models: List[Tuple[str, ModelData]], name: str) -> Optional[ModelData]:
    meshes: List[MeshData] = []
    bones = []
    for ref, m in models:
        if not m:
            continue
        # Wheel component assets are static; preserve mesh material names and
        # geometry exactly and do not bake any per-component scaling here.
        meshes.extend(m.meshes)
        if not bones and m.bones:
            bones = m.bones
    if not meshes:
        return None
    return ModelData(name=name, meshes=meshes, bones=bones, source_path=" + ".join(x[0] for x in models))


def _model_radius_and_halfwidth(model: ModelData) -> Tuple[float, float]:
    mn, mx = model.bounds()
    radius = max(abs(float(mn[1])), abs(float(mx[1])), abs(float(mn[2])), abs(float(mx[2])))
    halfw = max(abs(float(mn[0])), abs(float(mx[0])))
    return radius, halfw


def _scale_matrix(xyz) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[0,0], M[1,1], M[2,2] = [float(x) for x in xyz]
    return M


def _rot_y_180() -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[0,0] = -1.0
    M[2,2] = -1.0
    return M


class VehicleResolver50:
    """Strict vehicle resolver driven by expanded Project/Template BIN data."""

    def __init__(self, db: AssetDatabase):
        self.db = db
        self.expander = TemplateExpander(db)
        self.vehicle_rows = db.spreadsheet_by_name("Vehicles")
        self.wheel_rows = db.spreadsheet_by_name("Wheels")
        # BBR2 mobile stores paint/decal presets in "Vehicle Skins".
        # Island Adventure moved the same per-vehicle paint data into
        # "Vehicle Configs" and mixes Driver + Vehicle rows in one sheet.
        # Prefer Vehicle Skins when present; otherwise select only Type=Vehicle
        # records from Vehicle Configs so driver presets never leak onto cars.
        self.skin_rows = db.spreadsheet_by_name("Vehicle Skins")
        if not self.skin_rows and getattr(db, "is_island_adventure", False):
            self.skin_rows = {
                str(row.get("Name")): row
                for row in db.spreadsheet("Vehicle Configs")
                if isinstance(row, dict)
                and str(row.get("Type") or "").lower() == "vehicle"
                and str(row.get("Name") or "")
            }
        # Index customizable-attachment projects by their authored basename.
        # Vehicle projects store only names such as Hood_LightBasic.
        self.attachment_projects = {}
        for path, recs in db.records_by_path.items():
            if path.startswith("VuProjectAsset/CustomizableAttachments/"):
                rec = db.get(path)
                if rec is not None:
                    self.attachment_projects.setdefault(path.rsplit("/", 1)[-1], rec)

    def vehicles(self) -> List[str]:
        names = []
        for name in self.vehicle_rows:
            if self.db.resolve_project("Cars/" + name):
                names.append(name)
        # Include projects not listed in spreadsheet, but keep spreadsheet order
        # as the primary authored list.
        return names

    def vehicle_skins(self, vehicle_name: str) -> List[str]:
        """Return selectable authored skin suffixes for one vehicle.

        ``Vehicles.Skins`` stores suffixes such as SkinA/HWGreen.  The current
        default may itself be an alternate project (e.g. HWRipRod HWDecal), so
        expose it once as ``Default`` and list only the remaining variants.
        """
        row=self.vehicle_rows.get(str(vehicle_name or ""), {}) or {}
        default_skin=str(row.get("Default Skin") or "")
        out=["Default"]
        for x in row.get("Skins") or []:
            x=str(x or "").strip()
            if not x or x==default_skin:
                continue
            project_ref="Cars/"+str(vehicle_name)+"_"+x
            row_key=str(vehicle_name)+"_"+x
            if self.db.resolve_project(project_ref) or row_key in self.skin_rows:
                out.append(x)
        return out

    def attachment_names(self) -> List[str]:
        """All authored customizable attachment projects in the package."""
        return sorted(self.attachment_projects, key=str.lower)

    def _attachment_config_views(self, attachment_name: str):
        rec = self.attachment_projects.get(str(attachment_name or ""))
        if rec is None:
            return None, [], []
        try:
            root = self.expander.expand_project(rec)
        except Exception:
            return rec, [], []
        views = list(walk_entities(root))
        configs = [v for v in views if v.type == "VuAttachmentConfigurationFolderEntity"]
        return rec, views, configs

    def compatible_attachment_mounts(self, spec: VehicleSpec, attachment_name: str) -> List[Tuple[str, str]]:
        """Return exact vehicle sockets allowed by an attachment's Mount Mask.

        The first tuple item is the stable expanded-entity path used when adding
        the preview; the second is a human-readable socket label.
        """
        _rec, _views, configs = self._attachment_config_views(attachment_name)
        if not configs:
            return []
        allowed = set()
        for cfg in configs:
            mask = cfg.properties.get("Mount Mask")
            if isinstance(mask, dict):
                allowed.update(str(k) for k, v in mask.items() if bool(v))
        out = []
        for mv in walk_entities(spec.expanded_root):
            # Expanded customizable sockets retain the source template ref as
            # their semantic type, e.g. #Tools/CustomizableAttachments/Side.
            if not str(mv.type or "").startswith("#Tools/CustomizableAttachments/"):
                continue
            mount_type = str(mv.properties.get("Type") or "")
            if mount_type not in allowed:
                continue
            label = str(mv.properties.get("Name") or mv.name or mv.path)
            out.append((mv.path, f"{label} [{mount_type}]"))
        return out

    def append_custom_attachment(self, spec: VehicleSpec, attachment_name: str,
                                 mount_path: str = "") -> str:
        """Add one user-selected authored attachment to a resolved VehicleSpec.

        Placement uses the exact customizable socket transform plus the
        attachment project's matching configuration transform. No bounds-based
        placement guesses are used.
        """
        rec, views, configs = self._attachment_config_views(attachment_name)
        if rec is None:
            raise KeyError(f"Attachment project not found: {attachment_name}")
        mounts = {v.path: v for v in walk_entities(spec.expanded_root)
                  if str(v.type or "").startswith("#Tools/CustomizableAttachments/")}
        candidates = self.compatible_attachment_mounts(spec, attachment_name)
        if not candidates:
            raise ValueError(f"{attachment_name} has no compatible authored mount on {spec.name}")
        if mount_path not in {p for p, _ in candidates}:
            mount_path = candidates[0][0]
        mount = mounts.get(mount_path)
        if mount is None:
            raise ValueError(f"Attachment mount no longer resolves: {mount_path}")
        mount_type = str(mount.properties.get("Type") or "")

        folder = None
        for cfg in configs:
            mask = cfg.properties.get("Mount Mask")
            if isinstance(mask, dict) and bool(mask.get(mount_type)):
                folder = cfg
                break
        if folder is None:
            raise ValueError(f"No {attachment_name} configuration accepts {mount_type}")

        prefix = folder.path + "/"
        added = 0
        source_tag = f"[CustomAttachment:{attachment_name}:{mount.path}]"
        for av in views:
            if not av.path.startswith(prefix):
                continue
            if av.type in ("VuVehicleModelParams", "VuVehicleLooseMountParams",
                           "VuVehicleLegacyLooseMountParams", "VuVehicleRagdollParams"):
                ref = av.properties.get("Model")
                if not isinstance(ref, str) or not ref:
                    continue
                mrec = self.db.resolve_model(ref)
                model = self.db.model(mrec) if mrec else None
                if model is None:
                    continue
                spec.models.append(ModelSpec(
                    f"Preview:{attachment_name}:{av.name}", "AttachmentPreview", ref,
                    model, mount.world_matrix @ av.world_matrix,
                    f"{source_tag} {rec.path}/{folder.name}"
                ))
                added += 1
            elif av.type in ("VuVehicleHeadlightCoronaParams", "VuVehicleBrakelightCoronaParams"):
                pp = av.properties
                c = pp.get("Texture Color", [1, 1, 1, 1])
                if not isinstance(c, (list, tuple)) or len(c) < 4:
                    c = [1, 1, 1, 1]
                spec.emitters.append(EmitterSpec(
                    f"Preview:{attachment_name}:{av.name}",
                    "headlight_corona" if "Headlight" in av.type else "brakelight_corona",
                    pp.get("Texture Name", ""), mount.world_matrix @ av.world_matrix,
                    tuple(float(x) for x in c[:4]), str(pp.get("Texture Name") or ""),
                    _f(pp.get("Texture Size"), 1.0), source_tag
                ))
        if not added:
            raise ValueError(f"{attachment_name} configuration {folder.name} has no renderable model")
        return next((label for p, label in candidates if p == mount_path), mount_path)

    def _resolve_wheel_models(self, wheel_name: str, axle: str, motorcycle: bool = False):
        row = self.wheel_rows.get(wheel_name, {})
        if motorcycle:
            hub = row.get("Motorcycle Hub") or row.get("Hub")
            rubber = row.get("Motorcycle Rubber") or row.get("Rubber")
            brake = (row.get("Motorcycle Front Brake") if axle == "front" else row.get("Motorcycle Rear Brake")) or row.get("Brake")
        else:
            hub, rubber, brake = row.get("Hub"), row.get("Rubber"), row.get("Brake")
        refs = []
        loaded = []
        for folder, logical in (("Hub", hub), ("Rubber", rubber), ("Brake", brake)):
            if not isinstance(logical, str) or not logical:
                continue
            ref = f"Car/Wheels/{folder}/{logical}"
            rec = self.db.resolve_model(ref)
            if not rec:
                continue
            m = self.db.model(rec)
            if m:
                refs.append(rec.path)
                loaded.append((rec.path, m))
        return row, refs, _mesh_component_model(loaded, "Wheel/" + wheel_name)

    def _append_default_attachment(self, spec: VehicleSpec, mount_view: EntityView, attachment_name: str):
        rec = self.attachment_projects.get(str(attachment_name or ""))
        if rec is None:
            spec.warnings.append(f"Default attachment project not found: {attachment_name} ({mount_view.path})")
            return
        try:
            root = self.expander.expand_project(rec)
        except Exception as exc:
            spec.warnings.append(f"Default attachment expand failed: {attachment_name}: {exc}")
            return
        views = list(walk_entities(root))
        mount_type = str(mount_view.properties.get("Type") or "")
        folder = None
        for av in views:
            if av.type != "VuAttachmentConfigurationFolderEntity":
                continue
            mask = av.properties.get("Mount Mask")
            if isinstance(mask, dict) and bool(mask.get(mount_type)):
                folder = av
                break
        if folder is None:
            # Do not render the Icon configuration.  If no exact mask was found,
            # prefer the first real attachment configuration.
            folder = next((av for av in views if av.type == "VuAttachmentConfigurationFolderEntity"), None)
        prefix = (folder.path + "/") if folder is not None else ""
        for av in views:
            if prefix and not av.path.startswith(prefix):
                continue
            if av.type == "VuVehicleModelParams":
                ref = av.properties.get("Model")
                if not isinstance(ref, str) or not ref:
                    continue
                mrec = self.db.resolve_model(ref)
                model = self.db.model(mrec) if mrec else None
                if model is None:
                    spec.warnings.append(f"Attachment model not found: {ref} ({attachment_name})")
                    continue
                M = mount_view.world_matrix @ av.world_matrix
                spec.models.append(ModelSpec(
                    f"{mount_view.name}:{attachment_name}:{av.name}", "Attachment", ref, model, M,
                    f"{mount_view.path} -> {rec.path}/{folder.name if folder else ''}"
                ))
            elif av.type in ("VuVehicleHeadlightCoronaParams", "VuVehicleBrakelightCoronaParams"):
                p = av.properties
                c = p.get("Texture Color", [1,1,1,1])
                if not isinstance(c, (list, tuple)) or len(c) < 4:
                    c = [1,1,1,1]
                spec.emitters.append(EmitterSpec(
                    f"{mount_view.name}:{attachment_name}:{av.name}",
                    "headlight_corona" if "Headlight" in av.type else "brakelight_corona",
                    p.get("Texture Name", ""), mount_view.world_matrix @ av.world_matrix,
                    tuple(float(x) for x in c[:4]), str(p.get("Texture Name") or ""),
                    _f(p.get("Texture Size"), 1.0), f"{mount_view.path} -> {rec.path}"
                ))

    def resolve(self, vehicle_name: str, skin_name: str = "Default") -> VehicleSpec:
        tuning = copy.deepcopy(self.vehicle_rows.get(vehicle_name, {}))
        selected_skin=str(skin_name or "Default")
        project = self.db.resolve_project("Cars/" + vehicle_name)
        if not project:
            raise KeyError(f"No vehicle project for {vehicle_name}")

        # Several Hot Wheels vehicles ship a plain editable project plus an
        # authored alternate project that is explicitly selected as the
        # vehicle's Default Skin.  v0.50/v0.51 always expanded the plain
        # project, so the baked HW livery, matching wheel presets and loose
        # mounts never appeared.  Follow Vehicles.Default Skin + Alternate
        # exactly when that alternate is itself a project.
        alternate = str(tuning.get("Alternate") or "")
        default_skin = str(tuning.get("Default Skin") or "")
        is_default = selected_skin.lower() == "default"
        if is_default and alternate and default_skin:
            alt_project = self.db.resolve_project("Cars/" + alternate)
            if alt_project:
                project = alt_project
        elif not is_default:
            variant_project = self.db.resolve_project("Cars/" + vehicle_name + "_" + selected_skin)
            if variant_project:
                project = variant_project

        root = self.expander.expand_project(project)
        spec = VehicleSpec(vehicle_name, project.path, project.source_name, root)
        spec.tuning = tuning
        spec.skin_name = "Default" if is_default else selected_skin
        spec.skin_key = vehicle_name if is_default else vehicle_name + "_" + selected_skin
        # For alternate Island Adventure projects (SkinA, HW variants, etc.),
        # use an exact authored config when one exists.  If it does not, leave
        # the skin row empty so the variant project's own materials are used
        # instead of incorrectly reapplying the base vehicle paint.
        if is_default:
            skin_row = self.skin_rows.get(vehicle_name, {})
        else:
            skin_row = self.skin_rows.get(spec.skin_key, {})
        spec.skin = copy.deepcopy(skin_row)

        entities = list(walk_entities(root))
        suspension_view: Optional[EntityView] = None
        wheel_views: List[EntityView] = []
        spare_views: List[EntityView] = []
        default_attachment_views: List[Tuple[EntityView, str]] = []
        suspension_attachment_views: List[EntityView] = []

        # Core project/template entities.
        for v in entities:
            typ = v.type
            p = v.properties
            if typ == "VuVehicleEngineParams":
                spec.engine.update(copy.deepcopy(p))
            elif typ == "VuVehicleSuspensionParams":
                spec.suspension.update(copy.deepcopy(p))
                suspension_view = v
            elif typ == "VuVehicleHullParams":
                spec.hull.update(copy.deepcopy(p))
            elif typ == "VuVehicleRigidBodyParams":
                spec.rigid_body.update(copy.deepcopy(p))
            elif typ == "VuVehicleDriverParams":
                spec.driver_matrix = v.world_matrix.copy()
                spec.driver_source = v.path
                spec.driving_anim_set = str(p.get("Driving Anim Set") or "Standard")
            elif typ == "VuVehicleWheelParams":
                wheel_views.append(v)
            elif typ == "VuVehicleSpareWheelParams":
                spare_views.append(v)
            elif typ == "VuVehicleSuspensionAttachmentParams":
                suspension_attachment_views.append(v)
            elif typ == "VuVehicleDriverAttachmentParams":
                ref = p.get("Model")
                bone = str(p.get("Bone") or "")
                if isinstance(ref, str) and ref:
                    rec = self.db.resolve_model(ref)
                    model = self.db.model(rec) if rec else None
                    if model:
                        spec.driver_attachments.append(DriverAttachmentSpec(
                            v.name or "DriverAttachment", ref, model, bone,
                            v.world_matrix.copy(), v.path
                        ))
                    else:
                        spec.warnings.append(f"Driver attachment model not found: {ref} ({v.path})")

            default_attachment = p.get("Default Attachment") if isinstance(p, dict) else None
            if isinstance(default_attachment, str) and default_attachment:
                default_attachment_views.append((v, default_attachment))

            # Base vehicle-visible models only.  Mount-point placeholder models
            # are intentionally not displayed until their power-up/effect is
            # active; older builds showed some of them permanently.
            if typ in ("VuVehicleModelParams", "VuVehicleLegacyLooseMountParams", "VuVehicleRagdollParams"):
                ref = p.get("Model")
                if isinstance(ref, str):
                    # Candy Coupe's normal customizable paint shell does not
                    # contain the broad wrapped peppermint stripe UV layout seen
                    # on the authored default holiday livery. The package's
                    # CandyCane_SkinRainbow shell carries that exact stripe
                    # coverage, so use its geometry for CandyCane_Default and
                    # recolor only its livery texture in MaterialResolver50.
                    if (typ == "VuVehicleModelParams" and v.name.lower() == "chassis"
                            and vehicle_name == "CandyCane_Default" and is_default):
                        candy_ref = "Car/Vehicles/CandyCane/CandyCane_SkinRainbow"
                        if self.db.resolve_model(candy_ref):
                            ref = candy_ref

                    # A small number of vehicles (notably HWRodgerDodger) use
                    # a direct alternate model instead of an alternate project.
                    # Apply it only to the chassis when Vehicles says that
                    # alternate is the default skin and the exact model exists.
                    if typ == "VuVehicleModelParams" and v.name.lower() == "chassis" and is_default and alternate and default_skin:
                        alt_ref = ref.rsplit("/", 1)[0] + "/" + alternate if "/" in ref else alternate
                        if self.db.resolve_model(alt_ref):
                            ref = alt_ref
                    rec = self.db.resolve_model(ref)
                    model = self.db.model(rec) if rec else None
                    if model:
                        if typ == "VuVehicleModelParams":
                            role = "Chassis"
                        elif typ == "VuVehicleRagdollParams":
                            # Detachable props such as PizzaWagon's tree freshener
                            # and MonsterTruck_Fire's hose are visible parts while
                            # attached; they only become ragdolls after gameplay
                            # interaction. Keep their authored project transform.
                            role = "Attachment"
                        else:
                            role = "LooseMount"
                        # Legacy loose mounts can author a rest-pose Angular Offset
                        # in addition to the entity transform. Hotweiler's hood, for
                        # example, has +20 deg in VuTransformComponent and -20 deg
                        # Angular Offset. Applying only the entity rotation leaves the
                        # mouth/bumper pitched upward and visually detached. Apply the
                        # authored offset to the rendered rest pose while leaving
                        # Center Of Mass Offset / Angular Limit as physics-only data.
                        model_matrix = v.world_matrix.copy()
                        if typ == "VuVehicleLegacyLooseMountParams":
                            angular_offset = p.get("Angular Offset")
                            if isinstance(angular_offset, (list, tuple)) and len(angular_offset) >= 3:
                                try:
                                    ao = np.asarray([float(angular_offset[0]), float(angular_offset[1]), float(angular_offset[2])], dtype=np.float32)
                                    if np.any(np.abs(ao) > 1e-6):
                                        model_matrix = model_matrix @ Transform(rotation_deg=ao).matrix()
                                except (TypeError, ValueError):
                                    pass
                        spec.models.append(ModelSpec(v.name or role, role, ref, model, model_matrix, v.path))
                    else:
                        spec.warnings.append(f"Model not found: {ref} ({v.path})")

            if typ in ("VuVehicleHeadlightCoronaParams", "VuVehicleBrakelightCoronaParams"):
                c = p.get("Texture Color", [1,1,1,1])
                if not isinstance(c, (list, tuple)) or len(c) < 4:
                    c = [1,1,1,1]
                spec.emitters.append(EmitterSpec(
                    v.name, "headlight_corona" if "Headlight" in typ else "brakelight_corona",
                    p.get("Texture Name", ""), v.world_matrix.copy(), tuple(float(x) for x in c[:4]),
                    str(p.get("Texture Name") or ""), _f(p.get("Texture Size"), 1.0), v.path
                ))
            elif typ == "VuVehiclePfxAttachmentParams":
                # Always-on authored vehicle PFX.  Screamin' Jack/Pumpkin uses
                # four Car/PumpkinCandle mounts plus Car/PumpkinFlameB in the
                # jack-o-lantern body.  Earlier Studio builds ignored this
                # entity type entirely, so those flames could never appear.
                ref = str(p.get("Pfx") or "").strip()
                if ref:
                    sc=max(0.0,_f(p.get("Pfx Scale"),1.0))
                    M=v.world_matrix.copy()
                    if abs(sc-1.0)>1e-7:
                        M=M @ _scale_matrix((sc,sc,sc))
                    spec.emitters.append(EmitterSpec(
                        v.name, "persistent_pfx", ref, M,
                        source=v.path, enabled=True
                    ))
            elif typ == "VuVehicleBackFirePfxParams":
                spec.emitters.append(EmitterSpec(
                    v.name, "backfire", str(p.get("Pfx") or ""), v.world_matrix.copy(),
                    alt_effect_ref=str(p.get("Pfx Blue") or ""), source=v.path, enabled=False
                ))

        # Suspension is an authored visible model and also supplies exact wheel
        # centers through the named Wheel Bone values.
        suspension_model = None
        suspension_world = np.eye(4, dtype=np.float32)
        if suspension_view:
            ref = suspension_view.properties.get("Model")
            if isinstance(ref, str):
                rec = self.db.resolve_model(ref)
                suspension_model = self.db.model(rec) if rec else None
                suspension_world = suspension_view.world_matrix.copy()
                if suspension_model:
                    spec.models.append(ModelSpec("Suspension", "Suspension", ref, suspension_model, suspension_world.copy(), suspension_view.path))
                else:
                    spec.warnings.append(f"Suspension model not found: {ref}")

        bone_by_name = {}
        if suspension_model and suspension_model.bones:
            mats = bind_globals(suspension_model)
            for i, b in enumerate(suspension_model.bones):
                if i < len(mats):
                    bone_by_name[b.name] = mats[i]

        # Models rigidly mounted to suspension bones (gap fillers, guards, etc.)
        # are authored separately from the suspension mesh.
        for sv in suspension_attachment_views:
            ref = sv.properties.get("Model")
            bone = str(sv.properties.get("Bone") or "")
            if not isinstance(ref, str) or not ref:
                continue
            rec = self.db.resolve_model(ref)
            model = self.db.model(rec) if rec else None
            if model is None:
                spec.warnings.append(f"Suspension attachment model not found: {ref} ({sv.path})")
                continue
            if bone and bone in bone_by_name:
                M = suspension_world @ bone_by_name[bone] @ sv.local_transform.matrix()
                spec.models.append(ModelSpec(sv.name or "SuspensionAttachment", "Attachment", ref, model, M, sv.path))
            else:
                spec.warnings.append(f"Suspension attachment bone not found: {bone or '(blank)'} ({sv.path})")

        # Render the vehicle's authored default cosmetic attachments using the
        # attachment project's configuration folder that matches the mount type.
        for av, attachment_name in default_attachment_views:
            self._append_default_attachment(spec, av, attachment_name)

        motorcycle = vehicle_name.lower().startswith("motorcycle_")
        for v in wheel_views:
            p = v.properties
            axle, side = _mask_semantics(p, v.name)
            wheel_name = str(p.get("Wheel Name") or "")
            bone_name = str(p.get("Wheel Bone") or "")
            diameter = _f(p.get("Diameter"), 1.0)
            width = _f(p.get("Width"), 1.0)
            offset_x = _f(p.get("Offset X"), 0.0)
            pfx = str(p.get("Power Slide Pfx") or "")
            row, refs, wheel_model = self._resolve_wheel_models(wheel_name, axle, motorcycle)
            if not wheel_model:
                spec.warnings.append(f"Wheel model preset not resolved: {wheel_name} ({v.name})")
                continue

            mount = np.eye(4, dtype=np.float32)
            if bone_name and bone_name in bone_by_name:
                mount = suspension_world @ bone_by_name[bone_name]
            else:
                spec.warnings.append(f"Exact wheel bone not found: {bone_name or '(blank)'} ({v.name})")
                # Strict mode deliberately avoids geometric chassis-bound guesses.
                mount = v.world_matrix.copy()

            # Offset X is authored on every wheel entity and is along the wheel
            # axle in vehicle/suspension space.
            offset = np.eye(4, dtype=np.float32)
            offset[0,3] = offset_x
            mount = mount @ offset

            radius, halfw = _model_radius_and_halfwidth(wheel_model)
            radial = (diameter * 0.5 / radius) if diameter > 0 and radius > 1e-8 else 1.0
            width_scale = (width * 0.5 / halfw) if width > 0 and halfw > 1e-8 else radial
            scale = _scale_matrix((width_scale, radial, radial))
            orient = _rot_y_180() if side == "left" else np.eye(4, dtype=np.float32)
            world = mount @ orient @ scale

            ws = WheelSpec(v.name, axle, side, wheel_name, bone_name, diameter, width, offset_x, pfx,
                           world, wheel_model, refs, v.path)
            spec.wheels.append(ws)
            if pfx:
                # Powerslide PFX is authored per wheel.  Do not duplicate one
                # reference onto both rear wheels as v0.43 did.
                em = mount.copy()
                spec.emitters.append(EmitterSpec(v.name + " PowerSlide", "powerslide", pfx, em, source=v.path, enabled=False))

        # Spare wheels reuse the authored axle preset/dimensions and the exact
        # spare-wheel project transform.
        if spare_views and spec.wheels:
            for sv in spare_views:
                wanted = str(sv.properties.get("Wheel") or "rear").lower()
                candidates = [w for w in spec.wheels if w.axle == wanted] or spec.wheels
                base = candidates[0]
                # Remove the source wheel's mount and keep its exact physical
                # scaling/orientation under the spare entity transform.
                radius, halfw = _model_radius_and_halfwidth(base.model)
                radial = (base.diameter * 0.5 / radius) if base.diameter > 0 and radius > 1e-8 else 1.0
                width_scale = (base.width * 0.5 / halfw) if base.width > 0 and halfw > 1e-8 else radial
                M = sv.world_matrix @ _scale_matrix((width_scale, radial, radial))
                spec.models.append(ModelSpec(sv.name, "SpareWheel", base.wheel_name, base.model, M, sv.path))

        # Exact geometric vehicle dimensions from resolved wheel centers.
        centers = [(w.axle, w.side, w.matrix[:3,3].copy()) for w in spec.wheels]
        front = [p for a,s,p in centers if a == "front"]
        rear = [p for a,s,p in centers if a == "rear"]
        if front and rear:
            fy = float(np.mean([p[1] for p in front])); ry = float(np.mean([p[1] for p in rear]))
            spec.wheelbase = abs(fy - ry)
        lr = []
        for axle in ("front", "middle", "rear"):
            L = [p for a,s,p in centers if a == axle and s == "left"]
            R = [p for a,s,p in centers if a == axle and s == "right"]
            if L and R:
                lr.append(abs(float(np.mean([p[0] for p in R]) - np.mean([p[0] for p in L]))))
        if lr:
            spec.track_width = float(np.mean(lr))

        return spec
