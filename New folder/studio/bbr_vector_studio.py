from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QSettings
    from PySide6.QtGui import QAction, QFont, QSurfaceFormat, QIcon
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
        QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
        QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter, QTabWidget, QTextEdit, QInputDialog,
        QVBoxLayout, QWidget
    )
except Exception as exc:
    raise SystemExit("PySide6 is required. Run RUN_BBR_VECTOR_STUDIO.bat first.\n" + str(exc))

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:
    from PySide6.QtWidgets import QOpenGLWidget

try:
    from OpenGL.GL import *
    from OpenGL.GLU import gluLookAt, gluPerspective, gluUnProject
except Exception as exc:
    raise SystemExit("PyOpenGL is required. Run RUN_BBR_VECTOR_STUDIO.bat first.\n" + str(exc))

from bbr50_assets import AssetDatabase, AssetRecord
from bbr50_character import CharacterResolver50, CharacterSpec
from bbr50_drive import DrivePreview50
from bbr50_driving_anim import DrivingAnimationResolver50, DrivingAnimInfo
from bbr50_effects import EffectLibrary50, BallSpawn, AbilityPfxCue
from bbr50_materials import MaterialResolver50, MaterialBinding
from bbr50_pfx import PfxLibrary50, PfxSystem50, PfxPattern50, PfxEmitter50, PfxProcessor50
from bbr50_vehicle import VehicleResolver50, VehicleSpec, WheelSpec
from bbr60_character_pfx import CharacterPfxTimelineResolver60, CharacterPfxTimelineSpec
from bbr61_character_props import CharacterPropTimelineResolver61, CharacterPropTimelineSpec
from bbr67_island import IslandAdventureResolver, IslandSceneSpec
from vector_formats import (
    ModelData, TextureData, parse_texture_asset, animation_globals, animation_globals_additive, animation_globals_weighted,
    animation_bone_visibility, bind_globals, skin_mesh_and_normals
)
from vector_audio import (
    inspect_audio_payload, extract_audio_payload, FmodVorbisSetupLibrary,
    rebuild_fsb5_vorbis_ogg
)
from external_animation import load_external_motion, mapping_report, auto_map_bones
from fbx_motion import retarget_fbx, retarget_custom_fbx
from fbx_character import CustomCharacter, load_fbx_character, skin_custom_mesh
from custom_skin import load_skin_package, match_texture
from apf_writer import (build_existing_skin_replacement, write_apf_from_archive,\n                            build_texture_usage_index, skin_texture_paths, texture_usage_diagnostics)
from android_repack import (
    discover_apf_targets, repack_with_apf, signing_guidance,
    verify_repacked_apf
)

APP_VERSION = "1.05"
APP_TITLE = f"BBR Vector Studio v{APP_VERSION}"


def _resource_path(relative: str) -> Path:
    """Resolve packaged resources in source and PyInstaller builds."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / relative

APP_ICON_PATH = _resource_path("assets/BBR_Vector_Studio.ico")


def _matrix_gl(M):
    return np.asarray(M, dtype=np.float32).flatten(order="F")


def _translation(x, y, z=0.0):
    M=np.eye(4,dtype=np.float32); M[:3,3]=(x,y,z); return M


def _scale_xyz(x, y=None, z=None):
    if y is None: y=x
    if z is None: z=x
    M=np.eye(4,dtype=np.float32); M[0,0]=float(x); M[1,1]=float(y); M[2,2]=float(z); return M


def _rot_z(deg):
    a=math.radians(float(deg)); c=math.cos(a); s=math.sin(a)
    M=np.eye(4,dtype=np.float32); M[0,0]=c; M[0,1]=-s; M[1,0]=s; M[1,1]=c; return M


def _rot_x(deg):
    a=math.radians(float(deg)); c=math.cos(a); s=math.sin(a)
    M=np.eye(4,dtype=np.float32); M[1,1]=c; M[1,2]=-s; M[2,1]=s; M[2,2]=c; return M


def _rot_y(deg):
    a=math.radians(float(deg)); c=math.cos(a); s=math.sin(a)
    M=np.eye(4,dtype=np.float32); M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c; return M


def _rotation_xyz(v):
    try:
        x,y,z=[float(a) for a in v[:3]]
    except Exception:
        x=y=z=0.0
    return _rot_z(z) @ _rot_y(y) @ _rot_x(x)


def _safe_json(obj):
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x,"tolist") else str(x))
    except Exception:
        return repr(obj)


@dataclass
class PfxRuntime:
    kind: str
    base_matrix: np.ndarray
    system: PfxSystem50


@dataclass
class AbilityPfxRuntime:
    cue: AbilityPfxCue
    system: PfxSystem50
    base_matrix: np.ndarray
    elapsed: float = 0.0
    started: bool = False


@dataclass
class CharacterPfxRuntime:
    spec: CharacterPfxTimelineSpec
    system: PfxSystem50
    last_local_time: float = -1.0


@dataclass
class CharacterPropRuntime:
    spec: CharacterPropTimelineSpec


@dataclass
class EffectBallRuntime:
    spawn: BallSpawn
    model: Optional[ModelData]
    age: float = 0.0
    pfx: Optional[PfxSystem50] = None


class GLViewport(QOpenGLWidget):
    status_message = Signal(str)
    selection_message = Signal(str)
    animation_frame_changed = Signal(int, int)
    drive_state_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.db: Optional[AssetDatabase] = None
        self.materials: Optional[MaterialResolver50] = None
        self.pfxlib: Optional[PfxLibrary50] = None
        self.vehicle: Optional[VehicleSpec] = None
        self.character: Optional[CharacterSpec] = None
        self.environment: Optional[IslandSceneSpec] = None
        self.show_island_track_nodes = True
        self.show_island_environment = True
        self.show_island_animated = True
        self.show_island_gameplay = False
        self.show_island_water = True
        self.show_island_decals = True
        self.show_island_spawns = True
        self.show_island_collision = False
        self.show_island_oob = False
        self.show_island_powerups = False
        self.show_island_pfx = False
        self.show_island_cameras = False
        self.show_island_sectors = False
        self.show_scene_bounds = False
        self.selected_map_model = None
        self.hidden_map_model_ids = set()
        self.isolate_selected_map_model = False
        # Optional developer overlays for vehicle/character inspection.
        self.show_debug_wireframe = False
        self.show_debug_skeleton = False
        self.show_debug_bounds = False
        self.show_debug_mounts = False
        self.show_debug_normals = False
        # Shared developer/runtime controls. These are intentionally independent
        # from the authored game data and affect preview/debug rendering only.
        self.debug_paused = False
        self.debug_time_scale = 1.0
        self.show_debug_world_axes = False
        self.debug_xray_overlays = False
        self.debug_backface_culling = True
        self.character_animation = None
        self.character_animation_ref = ""
        self.external_root_follow = False
        self.external_follow_bone = 0
        self.external_follow_anchor = np.zeros(3,dtype=np.float32)
        self.character_frame = 0.0
        self.character_preview_fps = 30.0
        self._last_emitted_character_frame = -1
        # Free character preview honors the animation asset's trailing loopFlag.
        # The UI can explicitly force looping for inspection without changing
        # the decoded asset metadata.
        self.force_character_loop = False
        # Some BBR character clips are authored as synchronized multi-rig
        # animations. Cyber's lounge hologram/screens are a separate 15-bone
        # model + *_Screens track; keep them paired with the 24-bone body.
        self.character_companion_model = None
        self.character_companion_animation = None
        self.character_companion_ref = ""
        self.character_companion_clip = ""
        self.character_companion_variants = {}
        self.character_companion_kind = ""
        self.driver_anim_resolver: Optional[DrivingAnimationResolver50] = None
        self.character_pfx_resolver: Optional[CharacterPfxTimelineResolver60] = None
        self.character_prop_resolver: Optional[CharacterPropTimelineResolver61] = None
        self.driver_anim_info: Optional[DrivingAnimInfo] = None
        self.driver_idle_animation = None
        self.driver_steer_animation = None
        self.driver_turn_animation = None
        self.driver_lean_animation = None
        self.suspension_steer_animation = None
        self.driver_idle_frame = 0.0
        self.attach_character = False
        # Character-only driving-pose preview.  This is independent of a
        # vehicle so shared Driving/* clips can be inspected on the character
        # tab without forcing a car into the scene.
        self.character_drive_only = False
        self.character_drive_set = "Standard"
        self.character_drive_info: Optional[DrivingAnimInfo] = None
        self.character_drive_idle_animation = None
        self.character_drive_frame = 0.0
        self.drive: Optional[DrivePreview50] = None
        self.drive_enabled = False
        self.show_vehicle_attachments = True
        self.force_backfire_preview = False
        self.enhanced_vehicle_coronas = True
        self.drive_response_scale = 1.0
        self._backfire_phase = 0.0
        self.keys=set()
        self.pfx_instances: List[PfxRuntime] = []
        self.ability_pfx_instances: List[AbilityPfxRuntime] = []
        self.ability_preview_time = 0.0
        self.character_pfx_instances: List[CharacterPfxRuntime] = []
        self.character_prop_instances: List[CharacterPropRuntime] = []
        self._last_character_matrix=np.eye(4,dtype=np.float32)
        self._last_character_globals=[]
        # Last synchronized companion-rig pose (Cyber screens). Character PFX
        # timelines can attach directly to these prop bones.
        self._last_companion_model=None
        self._last_companion_matrix=np.eye(4,dtype=np.float32)
        self._last_companion_globals=[]
        # Frame-local matrices for timeline props. Character PFX can attach to
        # these actors as well as to the body/Cyber companion rig.
        self._last_character_prop_matrices={}
        self.effect_balls: List[EffectBallRuntime] = []
        self.texture_ids: Dict[Tuple, int] = {}
        self.custom_skin_textures = {}
        self.custom_skin_source = ""
        self.custom_skin_kind = ""
        self.custom_skin_model = None
        self.cube_texture_ids: Dict[Tuple, int] = {}
        self._map_dynamic_cache: Dict[int, bool] = {}
        self.environment_program = 0
        self.water_program = 0
        self._water_fallback_textures = {}
        self._water_preview_time = 0.0
        self._water_redraw_accum = 0.0
        # Island Adventure static-map render cache. One compiled list per unique
        # model avoids thousands of Python glVertex/glTexCoord calls every frame.
        self.map_display_lists: Dict[int, int] = {}
        self.reflection_geometry_lists: Dict[int, int] = {}
        self.map_static_parts_lists: Dict[int, int] = {}
        self._map_reflection_meshes: Dict[int, list] = {}
        self.water_geometry_lists: Dict[Tuple[int,int], int] = {}
        self._map_display_lists_supported = True
        self.carpaint_program = 0
        self.carpaint_uniforms = {}
        self.carpaint_shader_error = ""
        self.camera_yaw = -34.0
        self.camera_pitch = 18.0
        self.camera_distance = 8.0
        self.camera_target = np.asarray((0.0,0.0,0.8),dtype=np.float32)
        # Character-only preview environment is grid-only.
        self.preview_stage = "Grid"
        self.last_mouse=None
        self.mouse_press_pos=None
        self.drive_path=[]
        self._clock=time.monotonic()
        self.timer=QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(16)
        self.setMinimumSize(640,480)

    def _clear_map_display_lists(self):
        if (not getattr(self,"map_display_lists",None) and not getattr(self,"reflection_geometry_lists",None)
                and not getattr(self,"map_static_parts_lists",None) and not getattr(self,"water_geometry_lists",None)):
            return
        try:
            self.makeCurrent()
            for lid in (list(self.map_display_lists.values())+list(self.reflection_geometry_lists.values())+
                        list(self.map_static_parts_lists.values())+list(self.water_geometry_lists.values())):
                try:
                    glDeleteLists(int(lid), 1)
                except Exception:
                    pass
        finally:
            try: self.doneCurrent()
            except Exception: pass
        self.map_display_lists.clear(); self.reflection_geometry_lists.clear()
        self.map_static_parts_lists.clear(); self._map_reflection_meshes.clear()
        self.water_geometry_lists.clear()

    def _preload_map_model_textures(self, model: ModelData):
        """Upload map textures while no OpenGL display list is being compiled.

        glGenerateMipmap operates on the current texture immediately rather
        than reliably recording a later upload in a display list. Creating a
        texture inside GL_COMPILE can therefore leave the list with an
        incomplete texture, even though the material and ETC2 decode are fine.
        """
        if self.materials is None:
            return
        glActiveTexture(GL_TEXTURE0)
        for mesh in model.meshes:
            binding=self.materials.resolve_mesh(mesh.name,"environment")
            if binding.diffuse_texture is not None:
                self._texture_id(binding)
            if binding.diffuse_texture is not None and binding.detail_texture is not None:
                self._texture_id_data(
                    binding.detail_texture,
                    binding.detail_cache_key or ("detail",id(binding.detail_texture)),
                    clamp=False,
                )
            if binding.mask_texture is not None:
                self._texture_id_data(
                    binding.mask_texture,
                    binding.mask_cache_key or ("emissive-mask",id(binding.mask_texture)),
                    clamp=False,
                )
            if binding.diffuse_texture is None:
                cube=binding.env_cube or binding.additive_env_cube
                if cube is not None:
                    key=(binding.env_cube_cache_key or binding.additive_env_cube_cache_key
                         or ("cube-synthetic",id(cube)))
                    self._cube_texture_id_data(cube,key)

    def _draw_map_model_cached(self, model: ModelData, matrix):
        """Draw a static Island Adventure model using an OpenGL display list.

        The old map path submitted every vertex from Python on every 16 ms
        timer tick. A track can contain >1000 instances, so camera movement
        quickly saturated the UI thread. Display lists keep authored geometry
        and material bindings on the driver side and only submit one call per
        instance. If the current OpenGL profile lacks display lists, fall back
        safely to the original renderer.
        """
        if model is None:
            return
        key=id(model)
        if self.environment_program:
            dynamic=self._map_dynamic_cache.get(key)
            if dynamic is None:
                dynamic=any(self.materials.resolve_mesh(mesh.name,"environment").dynamic_environment
                            for mesh in model.meshes) if self.materials else False
                self._map_dynamic_cache[key]=dynamic
            if dynamic:
                # Keep opaque geometry driver cached; update only reflection
                # uniforms per frame. Animated/skinned models use normal path.
                if not model.bones and self._map_display_lists_supported:
                    self._draw_dynamic_map_model_cached(model,matrix)
                else:
                    self._draw_model(model,matrix,material_context="environment")
                return
        lid=self.map_display_lists.get(key)
        if lid is None and self._map_display_lists_supported:
            try:
                self._preload_map_model_textures(model)
                lid=int(glGenLists(1))
                if lid <= 0:
                    raise RuntimeError("glGenLists returned 0")
                glNewList(lid, GL_COMPILE)
                self._draw_model(model, np.eye(4,dtype=np.float32), material_context="environment")
                glEndList()
                self.map_display_lists[key]=lid
            except Exception:
                try: glEndList()
                except Exception: pass
                if lid:
                    try: glDeleteLists(int(lid),1)
                    except Exception: pass
                self._map_display_lists_supported=False
                lid=None
        if lid is not None:
            glPushMatrix(); glMultMatrixf(_matrix_gl(matrix)); glCallList(int(lid)); glPopMatrix()
        else:
            self._draw_model(model, matrix, material_context="environment")

    def _draw_dynamic_map_model_cached(self, model: ModelData, matrix):
        key=id(model)
        lid=self.map_static_parts_lists.get(key)
        if lid is None:
            reflected=[]; ordinary=[]
            for mesh in model.meshes:
                binding=self.materials.resolve_mesh(mesh.name,"environment")
                (reflected if binding.dynamic_environment else ordinary).append(mesh)
            self._map_reflection_meshes[key]=reflected
            try:
                self._preload_map_model_textures(model)
                lid=int(glGenLists(1))
                if not lid: raise RuntimeError("glGenLists returned 0")
                glNewList(lid,GL_COMPILE)
                for mesh in ordinary:
                    self._draw_mesh(mesh,mesh.positions,mesh.normals,material_context="environment")
                glEndList()
                self.map_static_parts_lists[key]=lid
            except Exception:
                try: glEndList()
                except Exception: pass
                if lid:
                    try: glDeleteLists(lid,1)
                    except Exception: pass
                self._draw_model(model,matrix,material_context="environment")
                return
        glPushMatrix(); glMultMatrixf(_matrix_gl(matrix))
        glCallList(lid)
        for mesh in self._map_reflection_meshes[key]:
            self._draw_mesh(mesh,mesh.positions,mesh.normals,material_context="environment")
        glPopMatrix()

    def _camera_follow_skybox_matrix(self, ms, eye, far_plane: float):
        """Build the game-style camera-follow transform for a skybox model.

        Track projects serialize the sky cube/dome with a normal world matrix,
        but VuSkyBoxEntity is rendered relative to the active camera at runtime.
        Re-centering it prevents the sky mesh from appearing as a giant rounded
        blue block in the middle of the course.
        """
        M=np.asarray(ms.matrix,dtype=np.float32).copy()
        M[:3,3]=np.asarray(eye,dtype=np.float32)[:3]

        # Keep the authored orientation/aspect, but guarantee that the camera
        # remains safely inside the shell while zooming out on large maps.
        try:
            mn,mx=ms.model.bounds()
            corners=np.asarray([
                (x,y,z) for x in (float(mn[0]),float(mx[0]))
                        for y in (float(mn[1]),float(mx[1]))
                        for z in (float(mn[2]),float(mx[2]))
            ],dtype=np.float32)
            q=corners@M[:3,:3].T
            radius=max(1.0,float(np.max(np.linalg.norm(q,axis=1))))
            desired=min(float(far_plane)*0.42,
                        max(450.0,float(self.camera_distance)*4.0))
            if radius < desired:
                M[:3,:3]*=float(desired/radius)
        except Exception:
            pass
        return M

    def _draw_island_skyboxes(self, eye, far_plane: float):
        """Render authored track/map skyboxes as a true background layer."""
        if self.environment is None or not self.show_island_environment:
            return
        skyboxes=[ms for ms in self.environment.models
                  if bool(getattr(ms,"skybox",False))
                  and id(ms) not in self.hidden_map_model_ids]
        if not skyboxes:
            return

        # Background pass: camera-centred, visible from inside, and never writes
        # depth. Real scenery is drawn afterwards and therefore cannot be
        # hidden by the sky shell.
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)
        try:
            for ms in skyboxes:
                M=self._camera_follow_skybox_matrix(ms,eye,far_plane)
                self._draw_map_model_cached(ms.model,M)
        finally:
            glUseProgram(0)
            glActiveTexture(GL_TEXTURE0)
            glDepthMask(GL_TRUE)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            if self.debug_backface_culling:
                glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
            else:
                glDisable(GL_CULL_FACE)

    def set_database(self, db: Optional[AssetDatabase]):
        self._clear_map_display_lists()
        self._map_dynamic_cache.clear()
        self._map_display_lists_supported=True
        self.makeCurrent()
        try:
            for tid in list(self.texture_ids.values()) + list(self.cube_texture_ids.values()):
                try: glDeleteTextures([tid])
                except Exception: pass
        finally:
            self.doneCurrent()
        self.texture_ids.clear(); self.cube_texture_ids.clear()
        self._water_fallback_textures.clear()
        self.db=db
        self.materials=MaterialResolver50(db) if db else None
        self.pfxlib=PfxLibrary50(db) if db else None
        self.driver_anim_resolver=DrivingAnimationResolver50(db) if db else None
        self.character_pfx_resolver=CharacterPfxTimelineResolver60(db) if db else None
        self.character_prop_resolver=CharacterPropTimelineResolver61(db) if db else None
        self.clear_scene()

    def clear_scene(self):
        self.vehicle=None; self.character=None; self.environment=None; self.selected_map_model=None; self.hidden_map_model_ids.clear(); self.isolate_selected_map_model=False; self.character_animation=None; self.character_animation_ref=""; self.character_preview_fps=30.0; self.drive=None
        self.character_companion_model=None; self.character_companion_animation=None; self.character_companion_ref=""; self.character_companion_clip=""; self.character_companion_variants={}; self.character_companion_kind=""
        self.driver_anim_info=None; self.driver_idle_animation=None; self.driver_steer_animation=None; self.driver_turn_animation=None; self.driver_lean_animation=None; self.suspension_steer_animation=None; self.driver_idle_frame=0.0
        self.attach_character=False
        self.character_drive_only=False; self.character_drive_info=None; self.character_drive_idle_animation=None; self.character_drive_frame=0.0
        self.drive_enabled=False; self.force_backfire_preview=False; self._backfire_phase=0.0; self.pfx_instances.clear(); self.ability_pfx_instances.clear(); self.ability_preview_time=0.0; self.character_pfx_instances.clear(); self.character_prop_instances.clear(); self.effect_balls.clear(); self.drive_path.clear()
        self._last_character_matrix=np.eye(4,dtype=np.float32); self._last_character_globals=[]
        self._last_companion_model=None; self._last_companion_matrix=np.eye(4,dtype=np.float32); self._last_companion_globals=[]
        self.camera_target=np.asarray((0,0,0.8),dtype=np.float32); self.camera_distance=8.0
        self.external_root_follow=False; self.external_follow_anchor=self.camera_target.copy()
        self.update()

    def set_vehicle(self, spec: VehicleSpec):
        self.environment=None
        self.vehicle=spec
        if self.materials:
            self.materials.set_cube_macros({})
            self.materials.set_vehicle(spec.name, getattr(spec,"skin_key",spec.name))
        front=[w.diameter for w in spec.wheels if w.axle=="front" and w.diameter>0]
        rear=[w.diameter for w in spec.wheels if w.axle=="rear" and w.diameter>0]
        fd=float(np.mean(front)) if front else .8; rd=float(np.mean(rear)) if rear else fd
        self.drive=DrivePreview50(spec.tuning,spec.engine,spec.suspension,spec.wheelbase,spec.track_width,fd,rd,
                                  is_motorcycle=spec.name.lower().startswith("motorcycle_"))
        self.drive.preview_response_scale=float(self.drive_response_scale)
        self.drive_enabled=False
        self.drive_path.clear()
        self._resolve_driver_animations()
        self._resolve_suspension_animation()
        self._build_vehicle_pfx()
        self.effect_balls.clear()
        self.fit_scene()
        self.update()

    def set_character(self, spec: Optional[CharacterSpec]):
        self.environment=None
        if self.materials:
            self.materials.set_cube_macros({})
        self.character_pfx_instances.clear(); self.character_prop_instances.clear()
        self.character=spec; self.character_animation=None; self.character_animation_ref=""; self.character_frame=0.0; self.character_preview_fps=30.0
        self.external_root_follow=False
        self.character_companion_model=None; self.character_companion_animation=None; self.character_companion_ref=""; self.character_companion_clip=""; self.character_companion_variants={}; self.character_companion_kind=""
        self.character_drive_only=False; self.character_drive_frame=0.0
        self._resolve_driver_animations()
        self._resolve_character_drive_preview(self.character_drive_set)
        self.fit_scene(); self.update()

    def set_environment(self, spec: Optional[IslandSceneSpec]):
        self._clear_map_display_lists()
        self._map_dynamic_cache.clear()
        self._water_fallback_textures.clear()
        self._water_preview_time=0.0; self._water_redraw_accum=0.0
        self._map_display_lists_supported=True
        # Island Adventure scene preview is a separate inspection mode.  Keep
        # the package/database loaded but clear vehicle/character runtime state.
        self.vehicle=None; self.character=None; self.drive=None; self.drive_enabled=False
        self.character_animation=None; self.character_animation_ref=""; self.character_frame=0.0
        self.character_pfx_instances.clear(); self.character_prop_instances.clear(); self.pfx_instances.clear(); self.ability_pfx_instances.clear(); self.ability_preview_time=0.0; self.effect_balls.clear()
        self.environment=spec
        self.selected_map_model=None
        self.hidden_map_model_ids.clear()
        self.isolate_selected_map_model=False
        if self.materials:
            self.materials.set_vehicle("")
            self.materials.set_cube_macros(spec.cube_macros if spec is not None else {})
        # v0.68 starts near the authored starting grid instead of fitting the
        # entire kilometre-scale track into a tiny distant overview.
        self.focus_environment_start()
        self.update()

    def set_show_island_track_nodes(self, on: bool):
        self.show_island_track_nodes=bool(on)
        self.update()

    def set_island_layer(self, key: str, on: bool):
        attr={
            "environment":"show_island_environment",
            "animated":"show_island_animated",
            "gameplay":"show_island_gameplay",
            "water":"show_island_water",
            "decals":"show_island_decals",
            "spawns":"show_island_spawns",
            "collision":"show_island_collision",
            "oob":"show_island_oob",
            "powerups":"show_island_powerups",
            "pfx":"show_island_pfx",
            "cameras":"show_island_cameras",
            "sectors":"show_island_sectors",
            "bounds":"show_scene_bounds",
        }.get(str(key))
        if attr:
            setattr(self,attr,bool(on)); self.update()

    def set_debug_wireframe(self, on: bool):
        self.show_debug_wireframe=bool(on); self.update()

    def set_debug_skeleton(self, on: bool):
        self.show_debug_skeleton=bool(on); self.update()

    def set_debug_bounds(self, on: bool):
        self.show_debug_bounds=bool(on); self.update()

    def set_debug_mounts(self, on: bool):
        self.show_debug_mounts=bool(on); self.update()

    def set_debug_normals(self, on: bool):
        self.show_debug_normals=bool(on); self.update()

    def set_debug_paused(self, on: bool):
        self.debug_paused=bool(on)
        # Reset the wall-clock anchor so resuming cannot accumulate a long dt.
        self._clock=time.monotonic()
        self.update()

    def set_debug_time_scale(self, scale: float):
        try: self.debug_time_scale=max(0.05,min(8.0,float(scale)))
        except Exception: self.debug_time_scale=1.0
        self._clock=time.monotonic(); self.update()

    def set_debug_world_axes(self, on: bool):
        self.show_debug_world_axes=bool(on); self.update()

    def set_debug_xray(self, on: bool):
        self.debug_xray_overlays=bool(on); self.update()

    def set_debug_backface_culling(self, on: bool):
        self.debug_backface_culling=bool(on); self.update()

    def step_character_frame(self, delta: int):
        if self.character_animation is None:
            self.status_message.emit("Load a character animation before stepping frames")
            return
        paired=(self.character_companion_animation is not None and
                not (self.attach_character and self.vehicle) and
                not self.character_drive_only)
        if paired:
            limit=max(int(self.character_animation.frame_count),
                      int(self.character_companion_animation.frame_count),1)-1
        else:
            limit=max(0,int(self.character_animation.frame_count)-1)
        self.character_frame=max(0.0,min(float(limit),float(self.character_frame)+float(delta)))
        cur=int(round(self.character_frame)); self._last_emitted_character_frame=cur
        self.animation_frame_changed.emit(cur,int(limit))
        self.status_message.emit(f"Animation frame {self.character_frame:.0f} / {limit}")
        self.update()

    def set_character_frame(self, frame: int):
        if self.character_animation is None:
            return
        paired=(self.character_companion_animation is not None and
                not (self.attach_character and self.vehicle) and
                not self.character_drive_only)
        if paired:
            limit=max(int(self.character_animation.frame_count),
                      int(self.character_companion_animation.frame_count),1)-1
        else:
            limit=max(0,int(self.character_animation.frame_count)-1)
        self.character_frame=max(0.0,min(float(limit),float(frame)))
        cur=int(round(self.character_frame)); self._last_emitted_character_frame=cur
        self.animation_frame_changed.emit(cur,int(limit))
        self.update()

    def set_camera_preset(self, name: str):
        key=str(name or '').strip().lower()
        presets={
            'perspective':(-34.0,18.0),
            'front':(180.0,0.0),
            'rear':(0.0,0.0),
            'left':(-90.0,0.0),
            'right':(90.0,0.0),
            'top':(0.0,80.0),
        }
        if key not in presets:
            return
        self.camera_yaw,self.camera_pitch=presets[key]
        self.update()

    @staticmethod
    def _world_bounds_for_model_spec(ms):
        mn,mx=ms.model.bounds()
        corners=[]
        for x in (float(mn[0]),float(mx[0])):
            for y in (float(mn[1]),float(mx[1])):
                for z in (float(mn[2]),float(mx[2])):
                    p=np.asarray(ms.matrix,dtype=np.float32)@np.asarray((x,y,z,1.0),dtype=np.float32)
                    corners.append(p[:3])
        a=np.asarray(corners,dtype=np.float32)
        return a.min(0),a.max(0)

    def focus_selected_map_object(self):
        ms=self.selected_map_model
        if ms is None:
            self.status_message.emit("Select a map object first (Ctrl + left-click)")
            return
        try:
            mn,mx=self._world_bounds_for_model_spec(ms)
            self.camera_target=(mn+mx)*0.5
            radius=max(1.0,float(np.linalg.norm(mx-mn)*0.5))
            self.camera_distance=max(3.0,radius*2.8)
            self.update()
        except Exception:
            self.status_message.emit("Could not focus the selected map object")

    def set_isolate_selected_map(self, on: bool):
        on=bool(on)
        if on and self.selected_map_model is None:
            self.isolate_selected_map_model=False
            self.status_message.emit("Select a map object before enabling isolate mode")
        else:
            self.isolate_selected_map_model=on
        self.update()

    def hide_selected_map_object(self):
        ms=self.selected_map_model
        if ms is None:
            self.status_message.emit("Select a map object first")
            return
        self.hidden_map_model_ids.add(id(ms))
        name=str(getattr(ms,'name','object'))
        self.selected_map_model=None
        self.isolate_selected_map_model=False
        self.selection_message.emit("No map object selected")
        self.status_message.emit(f"Hidden map object: {name}")
        self.update()

    def show_all_map_objects(self):
        count=len(self.hidden_map_model_ids)
        self.hidden_map_model_ids.clear(); self.isolate_selected_map_model=False
        self.status_message.emit(f"Restored {count} hidden map object(s)")
        self.update()

    def debug_scene_report(self) -> str:
        def counts(model):
            if model is None:return (0,0,0,0)
            meshes=list(getattr(model,'meshes',[]) or [])
            verts=sum(len(getattr(m,'positions',[])) for m in meshes)
            tris=sum(int(np.asarray(getattr(m,'indices',[])).size)//3 for m in meshes)
            bones=len(getattr(model,'bones',[]) or [])
            return len(meshes),verts,tris,bones
        lines=[
            "Developer scene statistics",
            f"Paused: {self.debug_paused}",
            f"Time scale: {self.debug_time_scale:g}x",
            f"Backface culling: {self.debug_backface_culling}",
            f"X-ray overlays: {self.debug_xray_overlays}",
            f"World axes: {self.show_debug_world_axes}",
        ]
        if self.vehicle is not None:
            total=[0,0,0,0]
            for ms in self.vehicle.models:
                c=counts(ms.model); total=[a+b for a,b in zip(total,c)]
            for w in self.vehicle.wheels:
                c=counts(w.model); total=[a+b for a,b in zip(total,c)]
            lines += ["",f"Vehicle: {self.vehicle.name}",
                      f"  Models: {len(self.vehicle.models)} + {len(self.vehicle.wheels)} wheel instances",
                      f"  Meshes: {total[0]}",f"  Vertices: {total[1]:,}",f"  Triangles: {total[2]:,}",
                      f"  Bones (summed): {total[3]}",f"  Emitters: {len(self.vehicle.emitters)}"]
            if self.drive is not None:
                st=self.drive.state
                lines += [
                    f"  Drive speed: {st.speed_mps*3.6:.1f} km/h",
                    f"  Steering raw/rack: {st.steering_input_raw:+.2f} / {st.steering_input:+.2f}",
                    f"  Wheel steering: {st.steering_deg:+.1f} deg",
                    f"  Driver turn input: {st.driver_turn_input:+.2f}",
                    f"  Slip: {st.slip_deg:+.1f} deg • powerslide={st.powersliding}",
                    f"  Preview response: {self.drive_response_scale:.2f}x",
                ]
        if self.character is not None:
            c=counts(self.character.model)
            lines += ["",f"Character: {self.character.name} / {self.character.skin}",
                      f"  Meshes: {c[0]}",f"  Vertices: {c[1]:,}",f"  Triangles: {c[2]:,}",f"  Bones: {c[3]}"]
            if self.character_animation is not None:
                lines += [f"  Animation: {self.character_animation_ref}",
                          f"  Frame: {self.character_frame:.1f}/{max(0,self.character_animation.frame_count-1)}"]
        active_systems=list(self.pfx_instances)+list(self.ability_pfx_instances)
        if active_systems or self.character_pfx_instances or self.effect_balls:
            systems=[x.system for x in active_systems] + [x.system for x in self.character_pfx_instances]
            lines += ["", "PFX / ability preview:",
                      f"  Vehicle systems: {len(self.pfx_instances)}",
                      f"  Ability systems: {len(self.ability_pfx_instances)}",
                      f"  Character timeline systems: {len(self.character_pfx_instances)}",
                      f"  Drop-ball actors: {len(self.effect_balls)}",
                      f"  Live particles: {sum(len(x.particles) for x in systems)}",
                      f"  Trail points: {sum(len(x.trail_points) for x in systems)}"]
        if self.environment is not None:
            instances=list(self.environment.models)
            unique={id(ms.model):ms.model for ms in instances if ms.model is not None}
            ui=[counts(m) for m in unique.values()]
            unique_meshes=sum(x[0] for x in ui); unique_verts=sum(x[1] for x in ui); unique_tris=sum(x[2] for x in ui)
            submitted_tris=sum(counts(ms.model)[2] for ms in instances if ms.model is not None)
            layers={}
            for ms in instances:layers[ms.layer]=layers.get(ms.layer,0)+1
            skybox_count=sum(1 for ms in instances if bool(getattr(ms,"skybox",False)))
            lines += ["",f"Map: {self.environment.name} ({self.environment.kind})",
                      f"  Model instances: {len(instances):,}",f"  Skybox instances: {skybox_count:,}",
                      f"  Hidden instances: {len(self.hidden_map_model_ids):,}",
                      f"  Isolate selected: {self.isolate_selected_map_model}",f"  Unique models: {len(unique):,}",
                      f"  Unique meshes: {unique_meshes:,}",f"  Unique vertices: {unique_verts:,}",
                      f"  Unique triangles: {unique_tris:,}",f"  Instance triangles: {submitted_tris:,}",
                      "  Layers: "+", ".join(f"{k}={v}" for k,v in sorted(layers.items()))]
        return "\n".join(lines)

    def clear_map_selection(self):
        self.selected_map_model=None
        self.isolate_selected_map_model=False
        self.selection_message.emit("No map object selected")
        self.update()

    def focus_environment_start(self):
        if self.environment is None:
            return
        if self.environment.start_focus is not None:
            self.camera_target=np.asarray(self.environment.start_focus,dtype=np.float32).copy()
            self.camera_target[2]+=4.0
            self.camera_distance=90.0 if self.environment.kind=="Track" else 220.0
            self.camera_yaw=-34.0; self.camera_pitch=28.0
        else:
            self.fit_scene()
        self.update()

    def fit_environment_full(self):
        if self.environment is None:
            return
        self.fit_scene(); self.update()

    def _resolve_character_drive_preview(self, set_name: str):
        self.character_drive_set=str(set_name or "Standard")
        self.character_drive_info=None; self.character_drive_idle_animation=None; self.character_drive_frame=0.0
        if not self.character or not self.driver_anim_resolver or not self.db:
            return
        info=self.driver_anim_resolver.resolve(self.character_drive_set)
        self.character_drive_info=info
        ref=info.idle_animation or info.steering_animation
        if ref:
            rec=self.db.resolve_animation(ref)
            if rec:
                try:
                    anim=self.db.animation(rec)
                    if anim and anim.bone_count==self.character.model.bone_count:
                        self.character_drive_idle_animation=anim
                except Exception:
                    pass
        # Some motorcycle sets keep the actual Turn/Lean controls embedded in
        # VuDrivingAnimationSetAsset.  If the external clip is unavailable,
        # use the embedded Turn control as the authored fallback.
        if self.character_drive_idle_animation is None and info.turn is not None:
            if info.turn.bone_count==self.character.model.bone_count:
                self.character_drive_idle_animation=info.turn
        if info.idle_pose_frame is not None:
            self.character_drive_frame=float(info.idle_pose_frame)

    def set_character_driving_preview(self, set_name: str, on: bool=True):
        self._resolve_character_drive_preview(set_name)
        self.character_drive_only=bool(on and self.character_drive_idle_animation is not None)
        if self.character_drive_only:
            self.character_animation=None; self.character_frame=0.0
            self.character_pfx_instances.clear(); self.character_prop_instances.clear()
        self.fit_scene(); self.update()
        return self.character_drive_only

    def clear_vehicle_keep_character(self):
        self.vehicle=None; self.drive=None; self.drive_enabled=False; self.keys.clear(); self.drive_path.clear()
        self.driver_anim_info=None; self.driver_idle_animation=None; self.driver_steer_animation=None; self.driver_turn_animation=None; self.driver_lean_animation=None; self.suspension_steer_animation=None; self.driver_idle_frame=0.0
        self.attach_character=False
        self.pfx_instances.clear(); self.effect_balls.clear()
        if self.materials:
            self.materials.set_vehicle("")
        self.fit_scene(); self.update()

    def _resolve_driver_animations(self):
        self.driver_anim_info=None; self.driver_idle_animation=None; self.driver_steer_animation=None; self.driver_turn_animation=None; self.driver_lean_animation=None; self.suspension_steer_animation=None; self.driver_idle_frame=0.0
        if not self.vehicle or not self.character or not self.driver_anim_resolver or not self.db:
            return
        info=self.driver_anim_resolver.resolve(self.vehicle.driving_anim_set)
        self.driver_anim_info=info
        self.driver_turn_animation=info.turn if info.turn and info.turn.bone_count==self.character.model.bone_count else None
        self.driver_lean_animation=info.lean if info.lean and info.lean.bone_count==self.character.model.bone_count else None
        def load(ref):
            if not ref: return None
            rec=self.db.resolve_animation(ref)
            if not rec: return None
            try:
                a=self.db.animation(rec)
                return a if a and a.bone_count==self.character.model.bone_count else None
            except Exception:
                return None
        self.driver_idle_animation=load(info.idle_animation)
        self.driver_steer_animation=load(info.steering_animation)
        if info.idle_pose_frame is not None:
            self.driver_idle_frame=float(info.idle_pose_frame)

    def _resolve_suspension_animation(self):
        self.suspension_steer_animation=None
        if not self.vehicle or not self.db:
            return
        ref=str(self.vehicle.suspension.get("Steering Animation") or "")
        if not ref:
            return
        rec=self.db.resolve_animation(ref)
        if not rec:
            return
        try:
            anim=self.db.animation(rec)
        except Exception:
            return
        susp=next((m for m in self.vehicle.models if m.role=="Suspension"),None)
        if susp and anim and anim.bone_count==susp.model.bone_count:
            self.suspension_steer_animation=anim

    def _resolve_character_companion(self, logical_ref: str):
        """Resolve Cyber's authored lounge screen rig(s) for a body clip.

        Cyber's hologram is not part of her 24-bone body model. The lounge
        timeline drives a separate 15-bone screen rig and swaps compatible
        Screen/Keyboard/Spinner models during the fidgets. v0.66 keeps those
        authored states and fixes the skinning/visibility corruption that made
        hidden or unweighted vertices stretch into oversized green rings.
        """
        self.character_companion_model=None
        self.character_companion_animation=None
        self.character_companion_ref=""
        self.character_companion_clip=""
        self.character_companion_variants={}
        self.character_companion_kind=""
        if not self.db or not self.character:
            return
        ref=str(logical_ref or "").strip().strip("/")

        # BeachBro's beach ball is a separate 2-bone animated companion actor.
        # It shares the exact frame count with Idle_A/B/C through *_Ball clips
        # and must be shown beside BeachBro rather than exposed as a skin.
        if self.character.name == "BeachBro":
            prefix="Character/Animations/Lounge_BeachBro/"
            if ref.startswith(prefix):
                clip=ref[len(prefix):]
                pair_ref=prefix+clip+"_Ball"
                arec=self.db.resolve_animation(pair_ref)
                skin=str(getattr(self.character,"skin","") or "")
                model_ref=("Character/BeachBro/BeachBall_SkinA" if skin.lower().endswith("_skina")
                           else "Character/BeachBro/BeachBall")
                mrec=self.db.resolve_model(model_ref)
                try:
                    anim=self.db.animation(arec) if arec else None
                    model=self.db.model(mrec) if mrec else None
                except Exception:
                    anim=model=None
                if anim is not None and model is not None and model.bone_count==anim.bone_count:
                    self.character_companion_model=model
                    self.character_companion_animation=anim
                    self.character_companion_ref=pair_ref
                    self.character_companion_clip=clip
                    self.character_companion_variants={"ball":model}
                    self.character_companion_kind="BeachBroBall"
            return

        if self.character.name != "Cyber":
            return
        prefix="Character/Animations/Lounge_Cyber/"
        if not ref.startswith(prefix):
            return
        clip=ref[len(prefix):]
        if clip not in ("Fidget_A","Fidget_B","Fidget_C","Fidget_D"):
            return

        screen_anim_ref=prefix+clip+"_Screens"
        arec=self.db.resolve_animation(screen_anim_ref)
        if not arec:
            return
        try:
            anim=self.db.animation(arec)
        except Exception:
            return
        if anim is None or anim.bone_count != 15:
            return

        skin=str(getattr(self.character,"skin","") or "")
        base_ref=("Character/Animations/Lounge_Cyber/Screen_SkinA"
                  if skin.lower().endswith("_skina") else
                  "Character/Animations/Lounge_Cyber/Screen")
        refs={
            "screen": base_ref,
            "keyboard": "Character/Animations/Lounge_Cyber/Screen_Keyboard",
            "spinner1": "Character/Animations/Lounge_Cyber/Screen_Spinner1",
            "spinner2": "Character/Animations/Lounge_Cyber/Screen_Spinner2",
            "spinnerboth": "Character/Animations/Lounge_Cyber/Screen_SpinnerBoth",
        }
        variants={}
        for key,mref in refs.items():
            mrec=self.db.resolve_model(mref)
            if not mrec:
                continue
            try:
                model=self.db.model(mrec)
            except Exception:
                model=None
            if model is not None and model.bone_count == anim.bone_count:
                variants[key]=model
        if not variants:
            return

        # Timeline-authored default actor for each fidget. Fidget_C changes
        # variants by frame in _cyber_companion_model_for_frame().
        default_key="keyboard" if clip=="Fidget_B" else "screen"
        self.character_companion_model=variants.get(default_key) or variants.get("screen") or next(iter(variants.values()))
        self.character_companion_animation=anim
        self.character_companion_ref=screen_anim_ref
        self.character_companion_clip=clip
        self.character_companion_variants=variants
        self.character_companion_kind="Cyber"

    def _cyber_companion_model_for_frame(self, frame: float):
        """Return Cyber's authored lounge companion for the current frame.

        v0.65 kept Fidget_C on the base Screen rig to avoid the green-ring
        corruption, but that also removed the real keyboard/spinner screen
        changes. v0.66 restores the authored model switches. The actual ring
        corruption is handled in the skinning/triangle-visibility path instead
        of deleting legitimate screen states.
        """
        if not self.character_companion_animation:
            return None
        if self.character_companion_kind == "BeachBroBall":
            return self.character_companion_model
        clip=self.character_companion_clip
        f=float(frame)
        v=self.character_companion_variants
        if clip=="Fidget_A":
            # Screen Show at local frame 0; Hide ~0.333s before body stop.
            if f >= 264.0:
                return None
            return v.get("screen") or self.character_companion_model
        if clip=="Fidget_B":
            # Keyboard actor Show 0.4s after animation start, Hide 0.333s early.
            if f < 12.0 or f >= 234.0:
                return None
            return v.get("keyboard") or self.character_companion_model
        if clip=="Fidget_C":
            # Authored slot/spinner sequence. These are separate 15-bone models
            # sharing Fidget_C_Screens. v0.66 can use them again because hidden
            # and unweighted vertices are no longer allowed to collapse to the
            # origin and form giant bridging triangles/rings.
            if f < 45.0:
                return v.get("screen") or self.character_companion_model
            if f < 76.0:
                return v.get("spinner1") or v.get("screen") or self.character_companion_model
            if f < 97.0:
                return v.get("spinnerboth") or v.get("spinner1") or v.get("screen") or self.character_companion_model
            if f < 127.0:
                return v.get("spinner2") or v.get("screen") or self.character_companion_model
            if f < 199.0:
                return v.get("screen") or self.character_companion_model
            return None
        # Fidget_D is not part of the main lounge cinematic in this package,
        # but it has a valid 15-bone Fidget_D_Screens track. Keep the standard
        # screen rig visible and synchronize it using the shared master clock.
        return v.get("screen") or self.character_companion_model

    def set_character_animation(self, anim, logical_ref: str="", preview_fps: Optional[float]=None):
        self.character_drive_only=False
        self.character_animation=anim; self.character_animation_ref=str(logical_ref or ""); self.character_frame=0.0
        self.character_preview_fps=float(preview_fps or (getattr(anim,"fps",30.0) if anim is not None else 30.0))
        self._resolve_character_companion(self.character_animation_ref)
        self._build_character_pfx()
        self._build_character_props()
        self._last_emitted_character_frame=-1
        limit=max(0,int(getattr(anim,"frame_count",1) or 1)-1) if anim is not None else 0
        self.animation_frame_changed.emit(0,limit)
        self.update()

    def _build_character_pfx(self):
        self.character_pfx_instances.clear()
        if not (self.character and self.character_animation and self.character_pfx_resolver and self.pfxlib):
            return
        try:
            specs=self.character_pfx_resolver.resolve(
                self.character.name, self.character.skin, self.character_animation_ref
            )
        except Exception as exc:
            self.status_message.emit(f"Character PFX timeline decode failed: {exc}")
            return
        for spec in specs:
            try:
                system=self.pfxlib.build(spec.effect_ref)
            except Exception:
                system=None
            if system is None:
                continue
            system.active=False; system.was_active=False
            self.pfxlib.reset(system, clear_particles=True)
            system.active=False; system.was_active=False
            self.character_pfx_instances.append(CharacterPfxRuntime(spec,system,-1.0))

    def _build_character_props(self):
        self.character_prop_instances.clear()
        if not (self.character and self.character_animation and self.character_prop_resolver):
            return
        try:
            specs=self.character_prop_resolver.resolve(
                self.character.name, self.character.skin, self.character_animation_ref
            )
        except Exception as exc:
            self.status_message.emit(f"Character prop timeline decode failed: {exc}")
            return
        self.character_prop_instances=[CharacterPropRuntime(x) for x in specs]

    def _character_prop_local_time(self, runtime: CharacterPropRuntime) -> float:
        if not self.character_animation:
            return 0.0
        fps=max(1e-6,float(self.character_preview_fps or self.character_animation.fps or 30.0))
        t=max(0.0,float(self.character_frame)/fps)
        dur=max(0.0,float(runtime.spec.duration))
        loops=bool(self.force_character_loop or self.character_animation.looping)
        if dur>1e-6:
            t=(t%dur) if loops else min(t,dur)
        return t

    def _draw_character_props(self, C: np.ndarray, char_globals):
        if not self.character_prop_instances or not self.character:
            return
        bone_map={b.name:i for i,b in enumerate(self.character.model.bones)}
        actor_leaf=lambda x: str(x or '').replace('\\','/').rsplit('/',1)[-1]
        for runtime in self.character_prop_instances:
            spec=runtime.spec
            t=self._character_prop_local_time(runtime)
            if not spec.visible_at(t):
                continue
            pos,rot,scl=spec.transform_at(t)
            M=C @ _translation(*pos) @ _rotation_xyz(rot) @ _scale_xyz(*scl)
            att=spec.attachment_at(t)
            resolved=True
            if att:
                parent,bone,rel_pos,rel_rot=att
                parent_leaf=actor_leaf(parent)
                rel=_translation(*rel_pos) @ _rotation_xyz(rel_rot) @ _scale_xyz(*scl)
                if parent_leaf==actor_leaf(spec.character_actor_name):
                    bi=bone_map.get(bone,-1)
                    if 0 <= bi < len(char_globals):
                        M=C @ char_globals[bi] @ rel
                    else:
                        resolved=False
                elif (parent_leaf.startswith("PropAnimated_Screen") or
                      parent_leaf.startswith("PropAnimated_Spinner")):
                    # Companion-bound props are valid only while that animated
                    # Cyber actor is actually visible this frame. Falling back
                    # to the prop's base transform is what created detached
                    # locks/panels around Cyber in earlier builds.
                    if self._last_companion_model is None:
                        resolved=False
                    else:
                        cb={b.name:i for i,b in enumerate(self._last_companion_model.bones)}
                        bi=cb.get(bone,-1)
                        if 0 <= bi < len(self._last_companion_globals):
                            M=self._last_companion_matrix @ self._last_companion_globals[bi] @ rel
                        else:
                            resolved=False
                elif parent_leaf in self._last_character_prop_matrices:
                    M=(self._last_character_prop_matrices[parent_leaf] @
                       _translation(*rel_pos) @ _rotation_xyz(rel_rot) @
                       _scale_xyz(*scl))
                else:
                    # An authored attachment whose parent is unavailable should
                    # be hidden, never rendered at the character origin.
                    resolved=False
            if not resolved:
                continue
            self._last_character_prop_matrices[actor_leaf(spec.entity_name)]=np.asarray(M,dtype=np.float32).copy()
            self._draw_model(spec.model,M,material_context="attachment")

    def _reset_character_pfx(self):
        if not self.pfxlib:
            return
        for runtime in self.character_pfx_instances:
            runtime.system.active=False; runtime.system.was_active=False
            self.pfxlib.reset(runtime.system, clear_particles=True)
            runtime.system.active=False; runtime.system.was_active=False
            runtime.last_local_time=-1.0

    def set_force_character_loop(self, on: bool):
        self.force_character_loop=bool(on)
        self.update()

    def set_external_root_follow(self, on: bool):
        if self.external_root_follow:
            anchor=self.external_follow_anchor.copy()
        else:
            anchor=self.camera_target.copy()
        self.external_root_follow=bool(on)
        self.external_follow_anchor=anchor
        if not on:
            self.camera_target[:2]=anchor[:2]
        self.update()

    def restart_character_animation(self):
        self.character_frame=0.0
        self._reset_character_pfx()
        self.update()

    def set_attach_character(self, on: bool):
        self.attach_character=bool(on); self.update()

    def set_show_vehicle_attachments(self, on: bool):
        self.show_vehicle_attachments=bool(on); self.update()

    def set_backfire_preview(self, on: bool):
        self.force_backfire_preview=bool(on); self.update()

    def set_enhanced_coronas(self, on: bool):
        self.enhanced_vehicle_coronas=bool(on); self.update()

    def set_drive_response_scale(self, scale: float):
        try:self.drive_response_scale=max(0.65,min(1.75,float(scale)))
        except Exception:self.drive_response_scale=1.0
        if self.drive is not None:
            self.drive.preview_response_scale=float(self.drive_response_scale)

    def stop_ability_preview(self):
        if self.pfxlib:
            for runtime in self.ability_pfx_instances:
                runtime.system.active=False
                self.pfxlib.step(runtime.system,0.0,world_matrix=self._ability_runtime_matrix(runtime),system_velocity=np.zeros(3,dtype=np.float32))
        self.ability_pfx_instances.clear(); self.ability_preview_time=0.0; self.effect_balls.clear(); self.update()

    def restart_ability_preview(self):
        if not self.pfxlib:return
        self.ability_preview_time=0.0
        for runtime in self.ability_pfx_instances:
            runtime.elapsed=0.0; runtime.started=False
            self.pfxlib.reset(runtime.system,clear_particles=True)
            runtime.system.active=False; runtime.system.was_active=False
        self.update()

    def _ability_runtime_matrix(self, runtime: AbilityPfxRuntime):
        cue=runtime.cue
        # Projectile-like PFX advances from the authored local origin. Velocity
        # is kept local so steering/vehicle orientation remains intuitive.
        move=np.asarray(cue.velocity,dtype=np.float32)*max(0.0,float(runtime.elapsed))
        local=runtime.base_matrix @ _translation(float(move[0]),float(move[1]),float(move[2]))
        if cue.bone and self.character is not None and self._last_character_globals:
            bone_map={b.name:i for i,b in enumerate(self.character.model.bones)}
            bi=bone_map.get(str(cue.bone),-1)
            if 0<=bi<len(self._last_character_globals):
                return self._last_character_matrix @ self._last_character_globals[bi] @ local
        if self.vehicle is not None:
            return self._vehicle_runtime_matrix() @ local
        if self.character is not None:
            return self._last_character_matrix @ local
        return local

    def trigger_ability_pfx(self, cues: List[AbilityPfxCue]):
        if not self.pfxlib:return 0
        self.ability_pfx_instances.clear(); self.ability_preview_time=0.0
        built=0
        for cue in cues or []:
            ref=str(cue.ref or '').strip(); base=_translation(*cue.position) @ _rotation_xyz(cue.rotation)
            # Some effects (notably AstroDog) identify an authored vehicle PFX
            # attachment by Mount instead of naming the system in VehicleEffectDB.
            if cue.mount:
                wanted=str(cue.mount).strip().lower()
                match=None
                if self.vehicle is not None:
                    match=next((e for e in self.vehicle.emitters if str(e.name or '').strip().lower()==wanted),None)
                    if match is None:
                        match=next((e for e in self.vehicle.emitters if wanted and wanted in str(e.name or '').lower()),None)
                if match is not None:
                    if not ref:ref=str(match.effect_ref or '')
                    base=np.asarray(match.matrix,dtype=np.float32) @ base
                elif not ref:
                    # Some VehicleEffectDB rows use Mount as the logical PFX
                    # system name instead of a vehicle socket (AstroDog uses
                    # "Astronaut"). This must also work in character-only
                    # preview where no vehicle is currently loaded.
                    candidate=str(cue.mount).strip()
                    if self.pfxlib.build(candidate) is not None:
                        ref=candidate
            if not ref:continue
            sys=self.pfxlib.build(ref)
            if sys is None:continue
            sys.active=False; sys.was_active=False
            self.ability_pfx_instances.append(AbilityPfxRuntime(cue,sys,base))
            built+=1
        self.update(); return built

    def _build_vehicle_pfx(self):
        self.pfx_instances.clear()
        if not self.vehicle or not self.pfxlib:
            return
        # PFX mounts and references are authored on the vehicle BIN.  Keep one
        # runtime per authored emitter; do not duplicate rear-wheel smoke or
        # synthesize backfire positions.
        for e in self.vehicle.emitters:
            if e.kind not in ("powerslide", "backfire", "persistent_pfx") or not e.effect_ref:
                continue
            sys=self.pfxlib.build(e.effect_ref)
            if sys:
                sys.active=bool(e.kind=="persistent_pfx")
                sys.was_active=False
                self.pfx_instances.append(PfxRuntime(e.kind, e.matrix.copy(), sys))
            if e.kind=="backfire":
                if getattr(e,"alt_effect_ref",""):
                    blue=self.pfxlib.build(e.alt_effect_ref)
                    if blue:
                        blue.active=False; blue.was_active=False
                        self.pfx_instances.append(PfxRuntime("backfire_blue",e.matrix.copy(),blue))

    @staticmethod
    def _transform_ball_spawn(spawn: BallSpawn, M: np.ndarray) -> BallSpawn:
        # DropBalls launch data is authored in vehicle-local coordinates.  The
        # native createBall path supplies the vehicle transform separately, so
        # the editor rotates/translates the launch plan here.  Vehicle linear
        # velocity is intentionally not added because that inheritance has not
        # been established from the native createBall body.
        p=(M@np.asarray((*spawn.position,1.0),dtype=np.float32))[:3]
        v=np.asarray(M[:3,:3]@np.asarray(spawn.velocity,dtype=np.float32),dtype=np.float32)
        return BallSpawn(
            spawn.model_ref, spawn.looping_pfx, spawn.radius, spawn.mass,
            spawn.life_time, np.asarray(p,dtype=np.float32), v,
            spawn.source_index, spawn.mode, spawn.delay
        )

    def trigger_effect(self, spawns: List[BallSpawn]):
        if not self.db:
            return
        self.effect_balls.clear()
        V=self._vehicle_runtime_matrix() if self.vehicle else np.eye(4,dtype=np.float32)
        for source_spawn in spawns:
            s=self._transform_ball_spawn(source_spawn,V)
            rec=self.db.resolve_model(s.model_ref) if s.model_ref else None
            model=self.db.model(rec) if rec else None
            pfx=self.pfxlib.build(s.looping_pfx) if self.pfxlib and s.looping_pfx else None
            if pfx:
                pfx.active=True
                pfx.was_active=False
            # Negative age is the authored/native release delay.  This keeps
            # Biker/BeachBro from dumping every ball into one frame.
            self.effect_balls.append(EffectBallRuntime(s,model,-float(s.delay),pfx))
        self.update()

    def toggle_drive(self, on: Optional[bool]=None):
        if not self.drive: return
        self.drive_enabled=(not self.drive_enabled) if on is None else bool(on)
        if not self.drive_enabled:
            self.keys.clear()
        self.update()

    def reset_drive(self):
        if self.drive: self.drive.reset()
        self._backfire_phase=0.0
        self.drive_path.clear()
        if self.pfxlib:
            for runtime in self.pfx_instances:
                self.pfxlib.reset(runtime.system)
        self.update()

    def initializeGL(self):
        glClearColor(0.055,0.065,0.085,1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0,GL_POSITION,(5.0,4.0,8.0,1.0))
        glLightfv(GL_LIGHT0,GL_DIFFUSE,(1.0,1.0,1.0,1.0))
        glLightfv(GL_LIGHT0,GL_AMBIENT,(0.30,0.30,0.34,1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        self._init_carpaint_shader()
        self._init_environment_shaders()

    def resizeGL(self,w,h):
        glViewport(0,0,max(1,w),max(1,h))

    def _vehicle_runtime_matrix(self):
        if not self.drive: return np.eye(4,dtype=np.float32)
        s=self.drive.state
        return _translation(s.x,s.y,0) @ _rot_z(s.yaw_deg)

    def _apply_driver_leg_clearance(self, globals_):
        """Small lower-leg clearance correction for Motorcycle_Anime.

        The package supplies the exact Motorcycle_Anime Turn/Lean clips and
        driver mount, and v0.62 continues to use them.  Unlike the other bikes,
        however, the anime fairing encloses the generic driver foot path and
        the project exposes no foot-peg/IK sockets.  Pull only the lower-leg
        bones slightly outward in model X so bulky character boots do not cut
        through the fairing; torso/hands/steering remain fully authored.
        """
        if not (self.vehicle and self.character and globals_ and self.vehicle.name=="Motorcycle_Anime"):
            return globals_
        out=[np.asarray(g,dtype=np.float32).copy() for g in globals_]
        bone_map={b.name:i for i,b in enumerate(self.character.model.bones)}
        # Gradual offsets avoid a hard kink at the knee while moving the boot
        # clear of the side panel. Right/left use mirrored model-space X.
        for side,prefix in ((1.0,"R_"),(-1.0,"L_")):
            for bone,amount in ((prefix+"Knee",0.035),(prefix+"Ankle",0.085),(prefix+"Foot",0.115)):
                bi=bone_map.get(bone,-1)
                if 0 <= bi < len(out):
                    out[bi]=_translation(side*amount,0.0,0.0) @ out[bi]
        return out

    def _draw_island_track_nodes(self):
        if not self.environment or not self.show_island_track_nodes or not self.environment.track_segments:
            return
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(2.0); glColor4f(1.0, 0.65, 0.08, 0.95)
        for seg in self.environment.track_segments:
            if len(seg) < 2:
                continue
            glBegin(GL_LINE_STRIP)
            for p in seg:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]) + 0.25)
            glEnd()
        glPointSize(4.0); glColor4f(1.0, 0.92, 0.25, 1.0)
        glBegin(GL_POINTS)
        for seg in self.environment.track_segments:
            for p in seg:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]) + 0.25)
        glEnd()
        glLineWidth(1.0); glPointSize(1.0); glEnable(GL_LIGHTING)

    @staticmethod
    def _draw_point_cloud(points, rgba, size=6.0):
        if not points:
            return
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glPointSize(float(size)); glColor4f(*rgba)
        glBegin(GL_POINTS)
        for item in points:
            p=item.position
            glVertex3f(float(p[0]),float(p[1]),float(p[2])+0.35)
        glEnd()
        glPointSize(1.0); glEnable(GL_LIGHTING)

    def _draw_island_collision(self):
        if not self.environment or not self.show_island_collision:
            return
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(2.0); glColor4f(1.0,0.12,0.12,0.95)
        for seg in self.environment.collision_segments:
            if len(seg)<2: continue
            glBegin(GL_LINE_STRIP)
            for p in seg:
                glVertex3f(float(p[0]),float(p[1]),float(p[2])+0.15)
            glEnd()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    def _draw_island_oob(self):
        if not self.environment or not self.show_island_oob:
            return
        edges=((0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7))
        corners=[(-.5,-.5,-.5),(.5,-.5,-.5),(-.5,.5,-.5),(.5,.5,-.5),(-.5,-.5,.5),(.5,-.5,.5),(-.5,.5,.5),(.5,.5,.5)]
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(1.5); glColor4f(1.0,0.15,0.85,0.85)
        for box in self.environment.out_of_bounds_boxes:
            glPushMatrix(); glMultMatrixf(_matrix_gl(box.matrix))
            glBegin(GL_LINES)
            for a,b in edges:
                glVertex3f(*corners[a]); glVertex3f(*corners[b])
            glEnd(); glPopMatrix()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    def _water_fallback_texture(self, water):
        """Keep authored shore coverage if the driver cannot compile GLSL."""
        source=water.water_map
        if source is None: return None
        key=(water.water_map_ref,water.decal_texture_ref)
        if key in self._water_fallback_textures: return self._water_fallback_textures[key]
        channels=source.rgba.astype(np.float32)/255.0
        shade=channels[...,0:1]
        rgb=np.asarray((0.07,0.23,0.32),np.float32)+shade*np.asarray((0.10,0.26,0.28),np.float32)
        decal=water.decal_texture
        if decal is not None:
            # Repeat the decal at roughly the same world scale as the shader.
            x=(np.arange(source.width)*max(1.0,water.size_x/24.0)*decal.width/source.width).astype(np.int32)%decal.width
            y=(np.arange(source.height)*max(1.0,water.size_y/24.0)*decal.height/source.height).astype(np.int32)%decal.height
            d=decal.rgba[y[:,None],x[None,:]].astype(np.float32)/255.0
            amount=channels[...,2:3]*d[...,3:4]
            rgb=rgb*(1.0-amount)+d[...,:3]*amount
        rgba=np.empty_like(source.rgba)
        rgba[...,:3]=np.clip(np.round(rgb*255.0),0,255).astype(np.uint8)
        rgba[...,3]=np.clip(np.round(channels[...,3]*235.0),0,255).astype(np.uint8)
        tex=TextureData("WaterFallback/"+water.water_map_ref,source.width,source.height,1,5,rgba,"Water map preview")
        self._water_fallback_textures[key]=tex
        return tex

    def _water_grid_list(self, water):
        """Tessellate the water once so vertex swells can move between frames."""
        if not self._map_display_lists_supported:
            return 0
        key=(float(water.size_x),float(water.size_y))
        cached=self.water_geometry_lists.get(key)
        if cached is not None: return cached
        count=max(16,min(72,int(math.ceil(max(key)/28.0))))
        lid=int(glGenLists(1))
        if not lid: return 0
        try:
            glNewList(lid,GL_COMPILE)
            glBegin(GL_QUADS)
            glNormal3f(0.0,0.0,1.0)
            for row in range(count):
                for col in range(count):
                    for u,v in ((col/count,row/count),((col+1)/count,row/count),
                                ((col+1)/count,(row+1)/count),(col/count,(row+1)/count)):
                        glTexCoord2f(u,v)
                        glVertex3f((u-.5)*water.size_x,(v-.5)*water.size_y,0.0)
            glEnd(); glEndList()
            self.water_geometry_lists[key]=lid
            return lid
        except Exception:
            try: glEndList()
            except Exception: pass
            try: glDeleteLists(lid,1)
            except Exception: pass
            return 0

    def _water_map_for_shader(self, water):
        """Return authored WaterMap or a 1x1 full-coverage procedural map.

        Some small/test surfaces omit WaterMap completely.  v0.82 dropped them
        to the non-animated fallback path, so they never received waves or
        reflections.  A synthetic coverage map keeps the same shader pipeline
        without inventing shoreline/decal masks.
        """
        if getattr(water,"water_map",None) is not None:
            return water.water_map
        key=("synthetic-water-map",)
        tex=self._water_fallback_textures.get(key)
        if tex is None:
            rgba=np.asarray([[[255,0,0,255]]],dtype=np.uint8)
            tex=TextureData("Water/SyntheticCoverage",1,1,1,5,rgba,"Synthetic water coverage")
            self._water_fallback_textures[key]=tex
        return tex

    def _water_cube_for_surface(self, water):
        if not (self.db and self.environment):
            return "",None
        ref=str(getattr(water,"reflection_cube_ref","") or "").strip()
        procedural=bool(getattr(water,"procedural_reflection",True))
        if not ref and procedural:
            ref=str(self.environment.cube_macros.get("Proxy_cube","") or "")
        if ref in self.environment.cube_macros:
            ref=str(self.environment.cube_macros.get(ref) or ref)
        if not ref:
            return "",None
        try:
            return ref,self.db.cube_texture(ref)
        except Exception:
            return ref,None

    def _draw_island_water(self):
        if not self.environment or not self.show_island_water or not self.environment.water_surfaces:
            return
        glDisable(GL_LIGHTING); glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)

        fresnel=None; foam_ramp=None
        if self.water_program and self.db:
            try: fresnel=self.db.texture("Water/Fresnel")
            except Exception: pass
            try: foam_ramp=self.db.texture("Water/FoamRamp")
            except Exception: pass
        fresnel_id=self._texture_id_data(fresnel,("water-fresnel",),clamp=True) if fresnel else 0
        foam_ramp_id=self._texture_id_data(foam_ramp,("water-foam-ramp",),clamp=False) if foam_ramp else 0
        try:
            for w in self.environment.water_surfaces:
                hx=float(w.size_x)*0.5; hy=float(w.size_y)*0.5
                water_map=self._water_map_for_shader(w)
                map_id=self._texture_id_data(water_map,("water-map",w.water_map_ref or "synthetic"),clamp=True) if self.water_program else 0
                decal_id=self._texture_id_data(w.decal_texture,("water-decal",w.decal_texture_ref)) if (map_id and w.decal_texture) else 0
                foam_id=self._texture_id_data(w.foam_texture,("water-foam",w.foam_texture_ref)) if (map_id and w.foam_texture) else 0
                cube_ref,cube_data=self._water_cube_for_surface(w)
                cube_id=self._cube_texture_id_data(cube_data,("water-cube",cube_ref)) if (map_id and cube_data) else 0
                use_shader=bool(self.water_program and map_id)
                if use_shader:
                    glUseProgram(self.water_program)
                    u=self.water_uniforms
                    for name,unit in (("uMap",0),("uDecal",1),("uFresnel",2),("uCube",3),("uFoam",4),("uFoamRamp",5)):
                        glUniform1i(u[name],unit)
                    for name,tid in (("uHasDecal",decal_id),("uHasFresnel",fresnel_id),
                                     ("uHasCube",cube_id),("uHasFoam",foam_id),("uHasFoamRamp",foam_ramp_id)):
                        glUniform1i(u[name],1 if tid else 0)
                    glUniform2f(u["uMapTexel"],1.0/max(1,water_map.width),1.0/max(1,water_map.height))
                    # Tiling is world-scale based rather than texture-resolution based.
                    glUniform2f(u["uTile"],max(1.0,w.size_x/32.0),max(1.0,w.size_y/32.0))
                    glUniform1f(u["uTime"],float(self._water_preview_time%4096.0))
                    glUniform1f(u["uFoamRampSpeed"],max(0.0,float(getattr(w,"foam_ramp_speed",0.08))))
                    glUniform1f(u["uFoamRampFrequency"],max(0.01,float(getattr(w,"foam_ramp_frequency",3.0))))

                    direction=math.radians(float(getattr(w,"wave_direction_deg",90.0)))
                    glUniform2f(u["uWaveDir"],math.cos(direction),math.sin(direction))
                    wind=max(0.0,float(getattr(w,"wave_wind_speed",8.0)))
                    complexity=max(1.0,float(getattr(w,"wave_complexity",6.0)))
                    suppression=max(0.0,float(getattr(w,"wave_suppression",0.0)))
                    # The game multiplier is a spectral-wave coefficient, not a
                    # direct vertex offset.  Scale it into preview world units
                    # while preserving Standard (.04) vs Small (.08) strength.
                    authored_height=max(0.0,float(getattr(w,"wave_height_multiplier",0.04)))
                    wave_height=min(0.24,max(0.0,authored_height*2.0))
                    glUniform1f(u["uWaveHeight"],wave_height)
                    glUniform1f(u["uWaveSpeed"],0.45+wind*0.035)
                    glUniform1f(u["uWaveScale"],0.047*(0.75+complexity/8.0))
                    glUniform1f(u["uWaveSuppression"],suppression)
                    self._reflection_matrix_uniform(u)
                    for unit,target,tid in ((0,GL_TEXTURE_2D,map_id),(1,GL_TEXTURE_2D,decal_id),
                                            (2,GL_TEXTURE_2D,fresnel_id),(3,GL_TEXTURE_CUBE_MAP,cube_id),
                                            (4,GL_TEXTURE_2D,foam_id),(5,GL_TEXTURE_2D,foam_ramp_id)):
                        if tid:
                            glActiveTexture(GL_TEXTURE0+unit); glEnable(target); glBindTexture(target,tid)
                    glActiveTexture(GL_TEXTURE0)
                else:
                    glUseProgram(0)
                    fallback=self._water_fallback_texture(w)
                    tid=self._texture_id_data(fallback,("water-fallback",w.water_map_ref,w.decal_texture_ref),clamp=True) if fallback else 0
                    if tid:
                        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); glColor4f(1,1,1,1)
                    else:
                        glDisable(GL_TEXTURE_2D); glColor4f(0.10,0.45,0.78,0.38)

                glPushMatrix(); glMultMatrixf(_matrix_gl(w.matrix))
                grid=self._water_grid_list(w) if use_shader else 0
                if grid:
                    glCallList(grid)
                else:
                    if use_shader: glUniform1f(u["uWaveHeight"],0.0)
                    glBegin(GL_QUADS)
                    glNormal3f(0.0,0.0,1.0)
                    for x,y,tu,tv in ((-hx,-hy,0,0),(hx,-hy,1,0),(hx,hy,1,1),(-hx,hy,0,1)):
                        glTexCoord2f(tu,tv if use_shader else 1-tv); glVertex3f(x,y,0.0)
                    glEnd()
                glPopMatrix()

                if use_shader:
                    glUseProgram(0)
                    for unit,target,tid in ((5,GL_TEXTURE_2D,foam_ramp_id),(4,GL_TEXTURE_2D,foam_id),
                                            (3,GL_TEXTURE_CUBE_MAP,cube_id),(2,GL_TEXTURE_2D,fresnel_id),
                                            (1,GL_TEXTURE_2D,decal_id),(0,GL_TEXTURE_2D,map_id)):
                        if tid: glActiveTexture(GL_TEXTURE0+unit); glDisable(target)
                    glActiveTexture(GL_TEXTURE0)
                else:
                    glDisable(GL_TEXTURE_2D)
        finally:
            glUseProgram(0); glActiveTexture(GL_TEXTURE0); glDisable(GL_TEXTURE_2D)
            glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING)
            if self.debug_backface_culling: glEnable(GL_CULL_FACE)

    def _draw_island_track_decals(self):
        if (not self.environment or not self.show_island_decals or
                not self.show_island_environment or not self.environment.track_decals):
            return
        glDisable(GL_LIGHTING); glDisable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1.0,-2.0)
        glColor4f(1.0,1.0,1.0,1.0)
        try:
            for decal in self.environment.track_decals:
                rec=decal.texture_record
                tid=self._texture_id_data(
                    decal.texture,("asset",rec.archive_index,rec.entry_index),clamp=True)
                if not tid:continue
                glBindTexture(GL_TEXTURE_2D,tid)
                glPushMatrix(); glMultMatrixf(_matrix_gl(decal.matrix))
                vertices=decal.projected_triangles
                if len(vertices):
                    glBegin(GL_TRIANGLES)
                    for x,y,z in vertices:
                        glTexCoord2f(float(x+.5),float(.5-y))
                        glVertex3f(float(x),float(y),float(z+.008))
                else:
                    # Some arrows have no sampled track mesh in this package;
                    # retain their authored orientation as a planar decal.
                    glBegin(GL_QUADS)
                    for u,v,x,y in ((0,1,-.5,-.5),(1,1,.5,-.5),
                                    (1,0,.5,.5),(0,0,-.5,.5)):
                        glTexCoord2f(u,v); glVertex3f(x,y,.008)
                glEnd(); glPopMatrix()
        finally:
            glDisable(GL_POLYGON_OFFSET_FILL); glDepthMask(GL_TRUE)
            glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING)
            if self.debug_backface_culling:glEnable(GL_CULL_FACE)

    def _draw_island_overlays(self):
        if not self.environment:
            return
        self._draw_island_water()
        self._draw_island_track_decals()
        self._draw_island_track_nodes()
        self._draw_island_collision()
        self._draw_island_oob()
        if self.show_island_spawns:
            self._draw_point_cloud(self.environment.spawn_points,(0.15,1.0,0.25,1.0),9.0)
        if self.show_island_powerups:
            self._draw_point_cloud(self.environment.powerup_points,(0.2,0.8,1.0,1.0),7.0)
        if self.show_island_pfx:
            self._draw_point_cloud(self.environment.pfx_points,(1.0,0.55,0.05,1.0),6.0)
        if self.show_island_cameras:
            self._draw_point_cloud(self.environment.camera_points,(0.75,0.35,1.0,1.0),8.0)
        if self.show_island_sectors:
            self._draw_point_cloud(self.environment.sector_points,(1.0,0.25,0.25,1.0),8.0)
        self._draw_selected_map_debug()

    def paintGL(self):
        # Developer option: some reverse-engineered/one-sided models are easier
        # to inspect with culling disabled. Restore the requested state every frame.
        if self.debug_backface_culling:
            glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)
        if self.environment is not None and getattr(self.environment,"clear_color",None) is not None:
            cc=np.asarray(self.environment.clear_color,dtype=np.float32)
            glClearColor(float(cc[0]),float(cc[1]),float(cc[2]),float(cc[3] if len(cc)>3 else 1.0))
        else:
            glClearColor(0.055,0.065,0.085,1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        h=max(1,self.height()); w=max(1,self.width())
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        far_plane=max(2000.0, float(self.camera_distance)*12.0) if self.environment is not None else 2000.0
        gluPerspective(45.0,float(w)/h,0.02,far_plane)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        target=self.camera_target.copy()
        if self.drive_enabled and self.drive:
            target[:2]=(self.drive.state.x,self.drive.state.y)
            target[2]=0.8
        pitch=math.radians(self.camera_pitch); cp=math.cos(pitch)
        if self.drive_enabled and self.drive:
            # Chase the game's authored +Y forward axis so W/S read visually as
            # forward/reverse instead of lateral motion from an oblique camera.
            a=math.radians(self.drive.state.yaw_deg)
            # _rot_z(yaw) rotates the vehicle's authored +Y forward axis to
            # (-sin(yaw), cos(yaw)). v0.72 used +sin here, so the chase camera
            # mirrored left/right relative to the car and made A/D feel inverted.
            forward=np.asarray((-math.sin(a),math.cos(a),0.0),dtype=np.float32)
            eye=target-forward*(cp*self.camera_distance)
            eye[2]+=math.sin(pitch)*self.camera_distance
        else:
            yaw=math.radians(self.camera_yaw)
            eye=target+np.asarray((
                math.sin(yaw)*cp*self.camera_distance,
                -math.cos(yaw)*cp*self.camera_distance,
                math.sin(pitch)*self.camera_distance
            ),dtype=np.float32)
        gluLookAt(float(eye[0]),float(eye[1]),float(eye[2]),float(target[0]),float(target[1]),float(target[2]),0,0,1)
        view=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order="F")
        self._eye_to_world=np.linalg.inv(view[:3,:3]).astype(np.float32)

        if self.environment is not None:
            amb=getattr(self.environment,"ambient_color",None)
            dif=getattr(self.environment,"directional_color",None)
            if amb is not None:
                a=np.asarray(amb,dtype=np.float32); glLightfv(GL_LIGHT0,GL_AMBIENT,(float(a[0]),float(a[1]),float(a[2]),float(a[3] if len(a)>3 else 1.0)))
            else:
                glLightfv(GL_LIGHT0,GL_AMBIENT,(0.30,0.30,0.34,1.0))
            if dif is not None:
                d=np.asarray(dif,dtype=np.float32); glLightfv(GL_LIGHT0,GL_DIFFUSE,(float(d[0]),float(d[1]),float(d[2]),float(d[3] if len(d)>3 else 1.0)))
            else:
                glLightfv(GL_LIGHT0,GL_DIFFUSE,(1.0,1.0,1.0,1.0))
        else:
            glLightfv(GL_LIGHT0,GL_AMBIENT,(0.30,0.30,0.34,1.0))
            glLightfv(GL_LIGHT0,GL_DIFFUSE,(1.0,1.0,1.0,1.0))

        if self.environment is not None:
            self._draw_island_skyboxes(eye,far_plane)
        elif self.drive_enabled and self.drive:
            self._draw_grid(target)
            self._draw_drive_course()
        else:
            self._draw_grid(target)
        self._draw_world_axes()
        V=self._vehicle_runtime_matrix()
        if self.environment is not None:
            for ms in self.environment.models:
                if bool(getattr(ms,"skybox",False)):
                    continue
                if id(ms) in self.hidden_map_model_ids:
                    continue
                if self.isolate_selected_map_model and self.selected_map_model is not None and ms is not self.selected_map_model:
                    continue
                if ms.layer=="Gameplay" and not self.show_island_gameplay:
                    continue
                if ms.layer=="Animated" and not self.show_island_animated:
                    continue
                if ms.layer=="Environment" and not self.show_island_environment:
                    continue
                self._draw_map_model_cached(ms.model, ms.matrix)
            self._draw_island_overlays()
        if self.vehicle:
            for ms in self.vehicle.models:
                if not ms.visible:
                    continue
                if ms.role in ("Attachment","LooseMount","AttachmentPreview") and not self.show_vehicle_attachments:
                    continue
                if ms.role=="Suspension" and self.suspension_steer_animation is not None:
                    steer_input=self.drive.state.steering_input if self.drive else 0.0
                    u=max(0.0,min(1.0,steer_input*0.5+0.5))
                    frame=u*float(max(0,self.suspension_steer_animation.frame_count-1))
                    globals_override=animation_globals_additive(ms.model,self.suspension_steer_animation,frame)
                    ctx="attachment" if ms.role in ("Attachment","LooseMount","AttachmentPreview") else "vehicle"
                    self._draw_model(ms.model,V@ms.matrix,globals_override=globals_override,material_context=ctx)
                else:
                    ctx="attachment" if ms.role in ("Attachment","LooseMount","AttachmentPreview") else "vehicle"
                    self._draw_model(ms.model,V@ms.matrix,material_context=ctx)
            for ws in self.vehicle.wheels:
                self._draw_wheel(ws,V)
            self._draw_coronas(V)
            self._draw_vehicle_mount_debug(V)
        # Companion pose is frame-local; do not let a hidden/detached Cyber
        # screen from the previous paint frame remain a valid PFX parent.
        self._last_companion_model=None
        self._last_companion_matrix=np.eye(4,dtype=np.float32)
        self._last_companion_globals=[]
        self._last_character_prop_matrices={}
        if self.character:
            C=np.eye(4,dtype=np.float32)
            anim=self.character_animation; frame=self.character_frame; globals_override=None
            if self.attach_character and self.vehicle:
                C=V@self.vehicle.driver_matrix
                if self.drive_enabled and self.drive and self.driver_turn_animation is not None:
                    st=self.drive.state
                    turn_u=max(0.0,min(1.0,st.driver_turn_input*0.5+0.5))
                    turn_frame=turn_u*float(max(0,self.driver_turn_animation.frame_count-1))
                    if self.driver_lean_animation is not None:
                        # Recovered runtime: motorcycle Turn/Lean weights are
                        # (1-speedRatio, speedRatio), with powerslide blending
                        # them toward Turn=1/Lean=0 at 4 units/second.
                        ratio=max(0.0,min(1.0,st.motorcycle_speed_ratio))
                        slide=max(0.0,min(1.0,st.powerslide_anim_blend))
                        lean_weight=ratio*(1.0-slide)
                        turn_weight=1.0-lean_weight
                        lean_u=max(0.0,min(1.0,st.motorcycle_lean_input*0.5+0.5))
                        lean_frame=lean_u*float(max(0,self.driver_lean_animation.frame_count-1))
                        globals_override=animation_globals_weighted(self.character.model,[
                            (self.driver_turn_animation,turn_frame,turn_weight),
                            (self.driver_lean_animation,lean_frame,lean_weight),
                        ])
                        anim=None
                    else:
                        # Native control time uses (steering*0.5+0.5), not the
                        # reversed mapping used by v0.50.
                        anim=self.driver_turn_animation; frame=turn_frame
                elif self.driver_idle_animation is not None:
                    # Being attached to the driver mount always needs the
                    # authored seated/driver idle.  Drive Preview only controls
                    # steering/lean simulation; it must not decide whether the
                    # character stands or sits.
                    anim=self.driver_idle_animation
                    if self.driver_anim_info and self.driver_anim_info.idle_pose_frame is not None:
                        frame=float(self.driver_anim_info.idle_pose_frame)
                    else:
                        frame=self.driver_idle_frame
            elif self.character_drive_only and self.character_drive_idle_animation is not None:
                anim=self.character_drive_idle_animation
                if self.character_drive_info and self.character_drive_info.idle_pose_frame is not None:
                    frame=float(self.character_drive_info.idle_pose_frame)
                else:
                    frame=self.character_drive_frame
            # Cyber lounge fidgets A-D are synchronized multi-rig animations.
            # The body and screen clips do not always have the same frame count
            # (notably Fidget_D), so use one master timeline and hold the shorter
            # component on its last authored frame instead of letting each loop
            # independently. This mirrors the working pre-v0.50 Studio behavior.
            # Force-loop preview follows the authored cinematic occurrence,
            # not merely the raw asset length. Alien Rapture_A, for example,
            # holds its last pose while its looping PFX finishes, then restarts.
            # v0.62 looped the body every 2.63s while the PFX looped every 3.67s.
            if (anim is self.character_animation and self.force_character_loop and
                    not (self.attach_character and self.vehicle) and not self.character_drive_only):
                cycle=self._character_authored_cycle_duration()
                pfps=max(1e-6,float(self.character_preview_fps or anim.fps or 30.0))
                if cycle > 1e-6:
                    ct=(float(self.character_frame)/pfps) % cycle
                    frame=min(ct*pfps,float(max(0,anim.frame_count-1)))

            paired_companion=(
                self.character_companion_model is not None and
                self.character_companion_animation is not None and
                anim is self.character_animation and
                not (self.attach_character and self.vehicle) and
                not self.character_drive_only
            )
            companion_frame=0.0
            preview_loop_override=None
            if anim is self.character_animation and not (self.attach_character and self.vehicle) and not self.character_drive_only:
                preview_loop_override=False if self.force_character_loop else None
            elif anim is self.driver_idle_animation and self.driver_anim_info is not None:
                # The DrivingAnimationSet timeline is a higher-level runtime
                # instruction and can explicitly loop an otherwise one-shot
                # VuAnimationAsset (e.g. Character/Animations/Driving/Idle).
                preview_loop_override=bool(self.driver_anim_info.idle_looping)
            elif self.character_drive_only and anim is self.character_drive_idle_animation and self.character_drive_info is not None:
                preview_loop_override=bool(self.character_drive_info.idle_looping)
            if paired_companion and self.character_animation is not None:
                master_count=max(int(self.character_animation.frame_count),
                                 int(self.character_companion_animation.frame_count),1)
                if self.character_companion_kind == "BeachBroBall":
                    # Ball/body clips are authored frame-for-frame. Use the same
                    # preview clock so throws/catches cannot drift apart.
                    master_frame=max(0.0,min(float(frame),float(master_count-1)))
                else:
                    pair_loops=bool(self.force_character_loop or
                                    (self.character_animation.looping and self.character_companion_animation.looping))
                    if pair_loops:
                        master_frame=float(self.character_frame)%float(master_count)
                    else:
                        master_frame=max(0.0,min(float(self.character_frame),float(master_count-1)))
                frame=min(master_frame,float(max(0,self.character_animation.frame_count-1)))
                companion_frame=min(master_frame,float(max(0,self.character_companion_animation.frame_count-1)))
            active_character_model = self.custom_skin_model or self.character.model
            char_globals = globals_override
            if char_globals is None:
                char_globals = animation_globals(active_character_model,anim,frame,loop_override=preview_loop_override) if active_character_model.bones else []
            if self.attach_character and self.vehicle:
                char_globals=self._apply_driver_leg_clearance(char_globals)
            self._last_character_matrix=np.asarray(C,dtype=np.float32).copy()
            self._last_character_globals=[np.asarray(g,dtype=np.float32).copy() for g in char_globals]
            context="custom" if isinstance(self.character,CustomCharacter) else "character"
            self._draw_model(active_character_model,C,anim,frame,globals_override=char_globals,material_context=context,loop_override=preview_loop_override)
            self._draw_character_mount_debug(C,char_globals)
            # Resolve/draw the synchronized companion first. Static cinematic
            # props and character PFX may attach to its bones in the same frame.
            if paired_companion:
                companion_model=self._cyber_companion_model_for_frame(companion_frame)
                if companion_model is not None:
                    companion_loop=(False if (self.force_character_loop or self.character_companion_kind=="BeachBroBall") else None)
                    companion_globals=(animation_globals(
                        companion_model,self.character_companion_animation,
                        companion_frame,loop_override=companion_loop
                    ) if companion_model.bones else [])
                    self._last_companion_model=companion_model
                    self._last_companion_matrix=np.asarray(C,dtype=np.float32).copy()
                    self._last_companion_globals=[np.asarray(g,dtype=np.float32).copy() for g in companion_globals]
                    self._draw_model(companion_model,C,self.character_companion_animation,
                                     companion_frame,globals_override=companion_globals,
                                     material_context="character",loop_override=companion_loop)
            if anim is self.character_animation and not (self.attach_character and self.vehicle) and not self.character_drive_only:
                self._draw_character_props(C,char_globals)
            if self.attach_character and self.vehicle and self.vehicle.driver_attachments:
                bone_map={b.name:i for i,b in enumerate(self.character.model.bones)}
                for att in self.vehicle.driver_attachments:
                    bi=bone_map.get(att.bone,-1)
                    if 0 <= bi < len(char_globals):
                        self._draw_model(att.model,C@char_globals[bi]@att.matrix,material_context="attachment")
        self._draw_pfx(V)
        self._draw_effect_balls()

    def _draw_world_axes(self):
        if not self.show_debug_world_axes:return
        length=35.0 if self.environment is not None else 3.5
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(3.0); glBegin(GL_LINES)
        glColor4f(1.0,0.18,0.18,1.0); glVertex3f(0,0,0.03); glVertex3f(length,0,0.03)
        glColor4f(0.18,1.0,0.25,1.0); glVertex3f(0,0,0.03); glVertex3f(0,length,0.03)
        glColor4f(0.20,0.55,1.0,1.0); glVertex3f(0,0,0.03); glVertex3f(0,0,length)
        glEnd(); glLineWidth(1.0); glEnable(GL_LIGHTING)

    def _draw_grid(self,target):
        glDisable(GL_LIGHTING); glDisable(GL_TEXTURE_2D); glColor4f(.22,.24,.29,1)
        glBegin(GL_LINES)
        base_x=round(float(target[0])/2)*2; base_y=round(float(target[1])/2)*2
        for i in range(-20,21):
            x=base_x+i*2; glVertex3f(x,base_y-40,0); glVertex3f(x,base_y+40,0)
            y=base_y+i*2; glVertex3f(base_x-40,y,0); glVertex3f(base_x+40,y,0)
        glEnd(); glEnable(GL_LIGHTING)

    def set_preview_stage(self, name):
        # Grid is the only supported character preview environment.
        self.preview_stage = "Grid"
        self.update()

    def _draw_drive_course(self):
        """Fixed world-space reference road and driven path.

        v0.50 followed the car with a repeating grid, which made correct motion
        look like skating because there were no unique landmarks.  The road,
        start line and breadcrumb path stay in world coordinates.
        """
        glDisable(GL_LIGHTING); glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)
        glLineWidth(2.0)
        # Lane edges and center line along +Y from the reset position.
        glColor4f(.42,.44,.50,1)
        glBegin(GL_LINES)
        for x in (-3.5,3.5):
            glVertex3f(x,-500,0.015); glVertex3f(x,500,0.015)
        # Start box / cross-line gives a non-repeating reference near origin.
        glVertex3f(-4.5,0,0.02); glVertex3f(4.5,0,0.02)
        glVertex3f(0,-4.5,0.02); glVertex3f(0,4.5,0.02)
        glEnd()
        glColor4f(.60,.62,.68,1)
        glBegin(GL_LINES)
        for y in range(-500,501,10):
            glVertex3f(0,y,0.018); glVertex3f(0,y+4,0.018)
            # Asymmetric roadside tick prevents the repeating-grid illusion.
            glVertex3f(3.5,y,0.018); glVertex3f(4.2,y,0.018)
        glEnd()
        # Actual chassis-center path.  If the car is truly sliding, this path
        # makes the lateral deviation visible instead of relying on the camera.
        if len(self.drive_path)>=2:
            glColor4f(.88,.88,.92,1); glLineWidth(2.5); glBegin(GL_LINE_STRIP)
            for x,y in self.drive_path:
                glVertex3f(float(x),float(y),0.035)
            glEnd()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    def _init_carpaint_shader(self):
        """Compile a compatibility-profile renderer for BBR's generic CarPaint slot.

        The target shader's paint/decal core is reproduced: two colour ramps are
        sampled by incident angle and VehicleDecalTexture.r supplies coverage.
        Lighting remains a lightweight viewport approximation; paint selection
        and decal semantics are authored game data rather than a baked guess.
        """
        self.carpaint_program=0
        self.carpaint_uniforms={}
        self.carpaint_shader_error=""
        vs=r'''#version 120
varying vec3 vNormalEye;
varying vec3 vViewEye;
varying vec2 vUV;
varying vec4 vVertexColor;
void main(){
    vec4 p=gl_ModelViewMatrix*gl_Vertex;
    vViewEye=p.xyz;
    vNormalEye=normalize(gl_NormalMatrix*gl_Normal);
    vUV=gl_MultiTexCoord0.xy;
    vVertexColor=gl_Color;
    gl_Position=gl_ProjectionMatrix*p;
}
'''
        fs=r'''#version 120
uniform sampler2D uPaint;
uniform sampler2D uDecalColor;
uniform sampler2D uDecalMask;
uniform int uMode;
uniform vec4 uDiffuse;
varying vec3 vNormalEye;
varying vec3 vViewEye;
varying vec2 vUV;
varying vec4 vVertexColor;
void main(){
    vec3 n=normalize(vNormalEye);
    vec3 viewDir=normalize(vViewEye);
    float ci=clamp(-dot(viewDir,n),0.0,1.0);
    vec3 paint=texture2D(uPaint,vec2(ci,0.5)).rgb;
    vec4 decalSample=texture2D(uDecalMask,vUV);
    vec3 base;
    if(uMode==1){
        // Art/CarDriver/CarPaint_CustomDecal: authored decal RGB + alpha.
        base=mix(paint,decalSample.rgb,decalSample.a)*uDiffuse.rgb*vVertexColor.rgb;
    }else{
        // Generic CarPaint and CarPaint_Custom: colour ramp + red mask.
        vec3 decal=texture2D(uDecalColor,vec2(ci,0.5)).rgb;
        base=mix(paint,decal,decalSample.r)*uDiffuse.rgb*vVertexColor.rgb;
    }
    vec3 L=normalize(vec3(0.35,0.45,1.0));
    float ndl=max(dot(n,L),0.0);
    float lightAmount=0.34+0.66*ndl;
    gl_FragColor=vec4(clamp(base*lightAmount,0.0,1.0),uDiffuse.a*vVertexColor.a);
}
'''
        try:
            v=glCreateShader(GL_VERTEX_SHADER); glShaderSource(v,vs); glCompileShader(v)
            if not glGetShaderiv(v,GL_COMPILE_STATUS):
                raise RuntimeError(str(glGetShaderInfoLog(v)))
            f=glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(f,fs); glCompileShader(f)
            if not glGetShaderiv(f,GL_COMPILE_STATUS):
                raise RuntimeError(str(glGetShaderInfoLog(f)))
            pr=glCreateProgram(); glAttachShader(pr,v); glAttachShader(pr,f); glLinkProgram(pr)
            if not glGetProgramiv(pr,GL_LINK_STATUS):
                raise RuntimeError(str(glGetProgramInfoLog(pr)))
            glDeleteShader(v); glDeleteShader(f)
            self.carpaint_program=int(pr)
            self.carpaint_uniforms={name:glGetUniformLocation(pr,name) for name in ("uPaint","uDecalColor","uDecalMask","uMode","uDiffuse")}
        except Exception as exc:
            self.carpaint_shader_error=str(exc)
            self.carpaint_program=0

    @staticmethod
    def _compile_preview_program(vertex_source, fragment_source):
        shaders=[]
        try:
            for kind,source in ((GL_VERTEX_SHADER,vertex_source),(GL_FRAGMENT_SHADER,fragment_source)):
                shader=glCreateShader(kind); shaders.append(shader)
                glShaderSource(shader,source); glCompileShader(shader)
                if not glGetShaderiv(shader,GL_COMPILE_STATUS):
                    raise RuntimeError(str(glGetShaderInfoLog(shader)))
            program=glCreateProgram()
            for shader in shaders: glAttachShader(program,shader)
            glLinkProgram(program)
            if not glGetProgramiv(program,GL_LINK_STATUS):
                message=str(glGetProgramInfoLog(program))
                glDeleteProgram(program)
                raise RuntimeError(message)
            return int(program)
        finally:
            for shader in shaders: glDeleteShader(shader)

    def _init_environment_shaders(self):
        """Preview the actual window/glass masks and view dependent cubemaps."""
        self.environment_program=0; self.water_program=0
        self.environment_shader_error=""; self.water_shader_error=""
        vertex=r'''#version 120
varying vec3 vEye;
varying vec3 vNormalEye;
varying vec2 vUV;
varying vec4 vColor;
void main(){
    vec4 eye=gl_ModelViewMatrix*gl_Vertex;
    vEye=eye.xyz;
    vNormalEye=normalize(gl_NormalMatrix*gl_Normal);
    vUV=gl_MultiTexCoord0.xy;
    vColor=gl_Color;
    gl_Position=gl_ProjectionMatrix*eye;
}'''
        environment=r'''#version 120
uniform sampler2D uBase;
uniform sampler2D uMask;
uniform sampler2D uFresnel;
uniform samplerCube uCube;
uniform samplerCube uAddCube;
uniform int uHasBase, uHasMask, uHasFresnel, uHasAddCube;
uniform int uDomeShell;
uniform float uEmissive;
uniform vec4 uDiffuse;
uniform mat3 uEyeToWorld;
varying vec3 vEye;
varying vec3 vNormalEye;
varying vec2 vUV;
varying vec4 vColor;
void main(){
    vec3 n=normalize(vNormalEye);
    if(uDomeShell!=0 && !gl_FrontFacing) n=-n;
    vec3 incident=normalize(vEye);
    float cosine=clamp(-dot(incident,n),0.0,1.0);
    vec4 base=(uHasBase!=0 ? texture2D(uBase,vUV) : vec4(1.0))*uDiffuse*vec4(vColor.rgb,1.0);
    vec4 mask=uHasMask!=0 ? texture2D(uMask,vUV) : vec4(1.0);
    vec3 color=base.rgb*(0.38+0.62*max(dot(n,normalize(vec3(0.35,0.45,0.82))),0.0));
    // The Alien dome is an opaque shell. Its authored white Fresnel ramp
    // otherwise replaces the entire surface with the darkest cube faces.
    if(uDomeShell!=0) color=base.rgb*(0.58+0.42*max(dot(n,normalize(vec3(0.35,0.45,0.82))),0.0));
    if(uEmissive>0.0) color=mix(color,base.rgb,clamp(uEmissive*mask.g,0.0,1.0));
    vec3 direction=normalize(uEyeToWorld*reflect(incident,n));
    float fresnel=uHasFresnel!=0 ? texture2D(uFresnel,vec2(cosine,0.5)).r
                                 : pow(1.0-cosine,3.0)*0.75+0.12;
    float amount=clamp(fresnel*(uHasMask!=0 ? mask.r : 1.0),0.0,1.0);
    if(uDomeShell!=0) amount=min(amount,0.58);
    color=mix(color,textureCube(uCube,direction).rgb,amount);
    if(uHasAddCube!=0) color+=textureCube(uAddCube,direction).rgb*(uHasMask!=0 ? mask.r : 1.0);
    gl_FragColor=vec4(clamp(color,0.0,1.0),base.a);
}'''
        water_vertex=r'''#version 120
uniform float uTime, uWaveHeight, uWaveSpeed, uWaveScale, uWaveSuppression;
uniform vec2 uWaveDir;
varying vec3 vEye;
varying vec3 vNormalEye;
varying vec2 vUV;
void main(){
    vec4 local=gl_Vertex;
    vec2 d1=normalize(uWaveDir);
    vec2 d2=normalize(vec2(-d1.y*0.76+d1.x*0.28,d1.x*0.76+d1.y*0.28));
    float k1=max(0.004,uWaveScale);
    float k2=k1*1.71;
    float phase1=dot(local.xy,d1)*k1+uTime*uWaveSpeed;
    float phase2=dot(local.xy,d2)*k2-uTime*uWaveSpeed*1.31;
    float secondary=clamp(0.46-uWaveSuppression*0.22,0.12,0.46);
    float w1=sin(phase1);
    float w2=sin(phase2);
    local.z+=(w1+secondary*w2)*uWaveHeight;
    float dx=uWaveHeight*(cos(phase1)*k1*d1.x+secondary*cos(phase2)*k2*d2.x);
    float dy=uWaveHeight*(cos(phase1)*k1*d1.y+secondary*cos(phase2)*k2*d2.y);
    vec3 localNormal=normalize(vec3(-dx,-dy,1.0));
    vec4 eye=gl_ModelViewMatrix*local;
    vEye=eye.xyz;
    vNormalEye=normalize(gl_NormalMatrix*localNormal);
    vUV=gl_MultiTexCoord0.xy;
    gl_Position=gl_ProjectionMatrix*eye;
}'''
        water=r'''#version 120
uniform sampler2D uMap;
uniform sampler2D uDecal;
uniform sampler2D uFresnel;
uniform samplerCube uCube;
uniform sampler2D uFoam;
uniform sampler2D uFoamRamp;
uniform int uHasDecal, uHasFresnel, uHasCube, uHasFoam, uHasFoamRamp;
uniform vec2 uMapTexel, uTile;
uniform float uTime, uFoamRampSpeed, uFoamRampFrequency;
uniform mat3 uEyeToWorld;
varying vec3 vEye;
varying vec3 vNormalEye;
varying vec2 vUV;
void main(){
    vec2 mapUV=vec2(vUV.x,1.0-vUV.y);
    vec4 channels=texture2D(uMap,mapUV);
    if(channels.a<0.004) discard;
    vec2 tileUV=vUV*uTile;
    vec2 flowA=vec2(uTime*0.026,-uTime*0.019);
    vec2 flowB=vec2(-uTime*0.017,uTime*0.023);
    vec3 foamA=uHasFoam!=0 ? texture2D(uFoam,tileUV*0.34+flowA).rgb : vec3(0.5);
    vec3 foamB=uHasFoam!=0 ? texture2D(uFoam,tileUV*0.21+flowB).rgb : vec3(0.5);
    vec2 slope=vec2((foamA.r-0.5)*0.24+(foamB.b-0.5)*0.14,
                    (foamA.g-0.5)*0.24+(foamB.r-0.5)*0.14);
    vec3 n=normalize(vNormalEye+vec3(slope*0.34,0.0));
    vec3 incident=normalize(vEye);
    float cosine=clamp(-dot(incident,n),0.0,1.0);
    float lightMask=channels.r;
    vec3 deep=vec3(0.035,0.145,0.205);
    vec3 shallow=vec3(0.125,0.405,0.505);
    vec3 color=mix(deep,shallow,clamp(0.18+0.82*lightMask,0.0,1.0));
    if(uHasCube!=0){
        vec3 direction=normalize(uEyeToWorld*reflect(incident,n));
        float amount=uHasFresnel!=0 ? texture2D(uFresnel,vec2(cosine,0.5)).r
                                    : 0.12+0.75*pow(1.0-cosine,3.0);
        color=mix(color,textureCube(uCube,direction).rgb,clamp(0.08+0.82*amount,0.0,0.88));
    }
    if(uHasDecal!=0){
        vec4 decal=texture2D(uDecal,tileUV+flowA*0.35);
        color=mix(color,decal.rgb,clamp(channels.b*decal.a,0.0,1.0));
    }
    float glint=(foamA.b-0.5)*0.055+(foamB.g-0.5)*0.035;
    color+=vec3(glint);
    if(uHasFoam!=0){
        float nearest=min(min(texture2D(uMap,mapUV+vec2(uMapTexel.x,0.0)).a,
                              texture2D(uMap,mapUV-vec2(uMapTexel.x,0.0)).a),
                          min(texture2D(uMap,mapUV+vec2(0.0,uMapTexel.y)).a,
                              texture2D(uMap,mapUV-vec2(0.0,uMapTexel.y)).a));
        float shore=clamp((channels.a-nearest)*3.2,0.0,1.0);
        float ramp=1.0;
        if(uHasFoamRamp!=0){
            vec2 rampUV=tileUV*max(0.25,uFoamRampFrequency*0.22)+vec2(uTime*uFoamRampSpeed,0.0);
            ramp=texture2D(uFoamRamp,rampUV).r;
        }
        float noise=clamp(foamA.r*0.72+foamB.b*0.42,0.0,1.0);
        float authoredFoam=clamp(channels.g,0.0,1.0);
        color=mix(color,vec3(0.80,0.91,0.93),shore*authoredFoam*noise*ramp*0.72);
    }
    gl_FragColor=vec4(clamp(color,0.0,1.0),clamp(channels.a*0.965,0.0,0.965));
}'''
        try:
            self.environment_program=self._compile_preview_program(vertex,environment)
            self.environment_uniforms={name:glGetUniformLocation(self.environment_program,name) for name in (
                "uBase","uMask","uFresnel","uCube","uAddCube","uHasBase","uHasMask",
                "uHasFresnel","uHasAddCube","uDomeShell","uEmissive","uDiffuse","uEyeToWorld")}
        except Exception as exc:
            self.environment_shader_error=str(exc)
        try:
            self.water_program=self._compile_preview_program(water_vertex,water)
            self.water_uniforms={name:glGetUniformLocation(self.water_program,name) for name in (
                "uMap","uDecal","uFresnel","uCube","uFoam","uFoamRamp",
                "uHasDecal","uHasFresnel","uHasCube","uHasFoam","uHasFoamRamp",
                "uMapTexel","uTile","uTime","uWaveHeight","uWaveDir","uWaveSpeed",
                "uWaveScale","uWaveSuppression","uFoamRampSpeed","uFoamRampFrequency","uEyeToWorld")}
        except Exception as exc:
            self.water_shader_error=str(exc)

    def _reflection_matrix_uniform(self, uniforms):
        glUniformMatrix3fv(uniforms.get("uEyeToWorld",-1),1,GL_TRUE,
                           np.ascontiguousarray(getattr(self,"_eye_to_world",np.eye(3)),dtype=np.float32))

    def _texture_id_data(self, tex, key, clamp=False):
        if tex is None or getattr(tex,"rgba",None) is None: return 0
        key=tuple(key) if isinstance(key,(tuple,list)) else (str(key),)
        if key in self.texture_ids: return self.texture_ids[key]
        rgba=np.ascontiguousarray(tex.rgba,dtype=np.uint8)
        tid=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        wrap=GL_CLAMP_TO_EDGE if clamp else GL_REPEAT
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,wrap); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,wrap)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,int(tex.width),int(tex.height),0,GL_RGBA,GL_UNSIGNED_BYTE,rgba)
        try: glGenerateMipmap(GL_TEXTURE_2D)
        except Exception: glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        try:
            max_aniso=glGetFloatv(0x84FF)
            if max_aniso: glTexParameterf(GL_TEXTURE_2D,0x84FE,min(8.0,float(max_aniso)))
        except Exception: pass
        self.texture_ids[key]=tid
        return tid

    def _cube_texture_id_data(self, cube, key):
        if cube is None or not getattr(cube, "faces", None): return 0
        key=tuple(key) if isinstance(key,(tuple,list)) else (str(key),)
        if key in self.cube_texture_ids: return self.cube_texture_ids[key]
        tid=glGenTextures(1); glBindTexture(GL_TEXTURE_CUBE_MAP,tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        targets=(GL_TEXTURE_CUBE_MAP_POSITIVE_X,GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                 GL_TEXTURE_CUBE_MAP_POSITIVE_Y,GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                 GL_TEXTURE_CUBE_MAP_POSITIVE_Z,GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
        for target,face in zip(targets,cube.faces[:6]):
            rgba=np.ascontiguousarray(face.rgba,dtype=np.uint8)
            glTexImage2D(target,0,GL_RGBA8,int(face.width),int(face.height),0,GL_RGBA,GL_UNSIGNED_BYTE,rgba)
        glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)
        try: glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE)
        except Exception: pass
        glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR)
        try: glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        except Exception: glTexParameteri(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        self.cube_texture_ids[key]=tid
        return tid

    def set_custom_skin_textures(self, textures, source="", kind="", model=None):
        self.custom_skin_textures = dict(textures or {})
        self.custom_skin_source = str(source or "")
        self.custom_skin_kind = str(kind or "")
        self.custom_skin_model = model
        self.texture_ids.clear()
        self.update()

    def clear_custom_skin(self):
        self.custom_skin_textures = {}
        self.custom_skin_source = ""
        self.custom_skin_kind = ""
        self.custom_skin_model = None
        self.texture_ids.clear()
        self.update()

    def _texture_id(self, binding: MaterialBinding):
        if binding is None or binding.diffuse_texture is None: return 0
        # A material preview can be derived from a real diffuse asset. Its
        # uploaded pixels are then different from that asset and from other
        # materials using the same source image (e.g. track ice/metal/foliage).
        if getattr(binding,"texture_cache_key",None) is not None:
            key=tuple(binding.texture_cache_key)
        elif binding.diffuse_texture_record is not None:
            rec=binding.diffuse_texture_record; key=("asset",rec.archive_index,rec.entry_index)
        else:
            key=("synthetic",id(binding.diffuse_texture))
        return self._texture_id_data(binding.diffuse_texture,key,clamp=False)

    def _apply_mesh_texture_wrap(self, mesh):
        """Restore the v0.43/v0.44 per-mesh wrap rule.

        Character/vehicle atlas UVs that stay inside 0..1 are clamped, while
        authored tiled paint/static UVs outside that range repeat.  v0.50+
        forced every ordinary texture to repeat, which can pull the opposite
        edge of an atlas into vehicle panels.
        """
        uvs = getattr(mesh, "uvs", None)
        if uvs is None or len(uvs) == 0:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            return
        uv = np.asarray(uvs, dtype=np.float32)
        # Static Vector models share one vertex buffer across several material
        # ranges.  Use only vertices referenced by this mesh; looking at the full
        # shared buffer can choose REPEAT because of another material's tiled UVs
        # and makes atlas textures appear to use the wrong edge/slot.
        try:
            used=np.unique(np.asarray(mesh.indices,dtype=np.int64).reshape(-1))
            used=used[(used>=0)&(used<len(uv))]
            if len(used): uv=uv[used]
        except Exception:
            pass
        finite = uv[np.isfinite(uv).all(axis=1)]
        if len(finite) == 0:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            return
        mn = finite.min(axis=0); mx = finite.max(axis=0)
        repeat_s = float(mn[0]) < -0.001 or float(mx[0]) > 1.001
        repeat_t = float(mn[1]) < -0.001 or float(mx[1]) > 1.001
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT if repeat_s else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT if repeat_t else GL_CLAMP_TO_EDGE)

    def _draw_carpaint_mesh(self,mesh,pos,nrm,binding,c):
        if not self.carpaint_program or not binding.carpaint:
            return False
        if binding.paint_texture is None or binding.decal_color_texture is None or binding.decal_mask_texture is None:
            return False
        paint_id=self._texture_id_data(binding.paint_texture,binding.paint_cache_key or ("paint",id(binding.paint_texture)),clamp=True)
        dcol_id=self._texture_id_data(binding.decal_color_texture,binding.decal_color_cache_key or ("dcol",id(binding.decal_color_texture)),clamp=True)
        mask_id=self._texture_id_data(binding.decal_mask_texture,binding.decal_mask_cache_key or ("dmask",id(binding.decal_mask_texture)),clamp=False)
        if not (paint_id and dcol_id and mask_id):
            return False
        glDisable(GL_BLEND); glDepthMask(GL_TRUE)
        glUseProgram(self.carpaint_program)
        try:
            glUniform1i(self.carpaint_uniforms.get("uPaint",-1),0)
            glUniform1i(self.carpaint_uniforms.get("uDecalColor",-1),1)
            glUniform1i(self.carpaint_uniforms.get("uDecalMask",-1),2)
            glUniform1i(self.carpaint_uniforms.get("uMode",-1),1 if getattr(binding,"carpaint_mode","")=="custom_decal" else 0)
            glUniform4f(self.carpaint_uniforms.get("uDiffuse",-1),float(c[0]),float(c[1]),float(c[2]),float(c[3]))
            for unit,tid in ((0,paint_id),(1,dcol_id),(2,mask_id)):
                glActiveTexture(GL_TEXTURE0+unit); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid)
            glActiveTexture(GL_TEXTURE0)
            idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
            glBegin(GL_TRIANGLES)
            for tri in idx:
                for vi in tri:
                    i=int(vi)
                    if i<0 or i>=len(pos): continue
                    if i<len(nrm): glNormal3f(float(nrm[i,0]),float(nrm[i,1]),float(nrm[i,2]))
                    if (getattr(binding,"uses_vertex_color",False) and
                            getattr(mesh,"colors",None) is not None and i<len(mesh.colors)):
                        vc=mesh.colors[i]; glColor4f(float(vc[0]),float(vc[1]),float(vc[2]),float(vc[3]))
                    else:
                        glColor4f(1.0,1.0,1.0,1.0)
                    if i<len(mesh.uvs): glTexCoord2f(float(mesh.uvs[i,0]),float(mesh.uvs[i,1]))
                    glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
            glEnd()
        finally:
            glUseProgram(0)
            for unit in (2,1,0):
                glActiveTexture(GL_TEXTURE0+unit); glDisable(GL_TEXTURE_2D)
            glActiveTexture(GL_TEXTURE0)
        return True

    @staticmethod
    def _mesh_for_animation_visibility(mesh, bone_visible):
        """Cull only triangles that are actually controlled by hidden bones.

        v0.65 hid an entire mesh when hidden-bone weight dominated that mesh.
        Cyber's screen assets pack several independently shown/hidden pieces
        into shared meshes, so that rule removed valid screen parts. Conversely,
        submitting triangles that bridge visible and zero-scale/NaN bones makes
        long spikes, panels and the characteristic giant green rings.

        Keep the mesh intact where possible and remove only triangles containing
        vertices whose authored skin influences are entirely hidden. Unweighted
        vertices stay visible because they are legitimate rigid geometry.
        """
        if bone_visible is None or not bone_visible:
            return mesh
        weights=getattr(mesh,"weights",None)
        joints=getattr(mesh,"joints",None)
        if weights is None or joints is None:
            return mesh
        try:
            w=np.asarray(weights,dtype=np.float32)
            j=np.asarray(joints,dtype=np.int32)
            if w.ndim!=2 or j.shape!=w.shape or not w.size:
                return mesh
            total=np.sum(np.where(w>1.0e-7,w,0.0),axis=1)
            visible=np.zeros(len(w),dtype=np.float32)
            for k in range(w.shape[1]):
                idx=j[:,k]; wk=w[:,k]
                good=(wk>1.0e-7) & (idx>=0) & (idx<len(bone_visible))
                if not np.any(good):
                    continue
                rows=np.nonzero(good)[0]
                mask=np.asarray([bool(bone_visible[int(idx[n])]) for n in rows],dtype=bool)
                if np.any(mask):
                    rr=rows[mask]
                    visible[rr]+=wk[rr]
            # Vertices with no authored weights are rigid/bind geometry. A
            # weighted vertex is drawable when at least one real influence is
            # currently visible. This is deliberately less destructive than
            # v0.65's mesh-wide 15% threshold.
            vertex_ok=(total<=1.0e-7) | (visible>1.0e-7)
            tri=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
            valid=(tri < len(vertex_ok)).all(axis=1)
            keep=valid.copy()
            if np.any(valid):
                tv=tri[valid]
                keep[valid]=vertex_ok[tv].all(axis=1)
            filtered=tri[keep]
            if len(filtered)==len(tri):
                return mesh
            if not len(filtered):
                return None
            return replace(mesh, indices=filtered.reshape(-1).astype(np.asarray(mesh.indices).dtype,copy=False))
        except Exception:
            return mesh

    @staticmethod
    def _draw_local_aabb(mn, mx, rgba=(0.15, 0.95, 1.0, 1.0), width=1.5):
        mn=np.asarray(mn,dtype=np.float32); mx=np.asarray(mx,dtype=np.float32)
        c=[(mn[0],mn[1],mn[2]),(mx[0],mn[1],mn[2]),(mn[0],mx[1],mn[2]),(mx[0],mx[1],mn[2]),
           (mn[0],mn[1],mx[2]),(mx[0],mn[1],mx[2]),(mn[0],mx[1],mx[2]),(mx[0],mx[1],mx[2])]
        e=((0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7))
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(float(width)); glColor4f(*rgba); glBegin(GL_LINES)
        for a,b in e:
            glVertex3f(*[float(x) for x in c[a]]); glVertex3f(*[float(x) for x in c[b]])
        glEnd(); glLineWidth(1.0); glEnable(GL_LIGHTING)

    @staticmethod
    def _draw_wireframe_mesh(mesh, pos):
        tri=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(1.0); glColor4f(0.05,0.95,1.0,0.95); glBegin(GL_LINES)
        for a,b,c in tri:
            ia,ib,ic=int(a),int(b),int(c)
            if max(ia,ib,ic)>=len(pos): continue
            for u,v in ((ia,ib),(ib,ic),(ic,ia)):
                glVertex3f(float(pos[u,0]),float(pos[u,1]),float(pos[u,2]))
                glVertex3f(float(pos[v,0]),float(pos[v,1]),float(pos[v,2]))
        glEnd(); glLineWidth(1.0); glEnable(GL_LIGHTING)

    @staticmethod
    def _draw_normals_mesh(pos, nrm):
        if len(pos)==0 or len(nrm)==0: return
        step=max(1,int(len(pos)//450))
        # Scale from sampled geometry span so normals remain readable on both
        # characters and large vehicles.
        try:
            span=np.ptp(np.asarray(pos,dtype=np.float32),axis=0)
            scale=max(0.02,min(0.35,float(np.linalg.norm(span))*0.025))
        except Exception:
            scale=0.08
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glColor4f(1.0,0.35,0.15,0.9); glLineWidth(1.0); glBegin(GL_LINES)
        for i in range(0,min(len(pos),len(nrm)),step):
            p=np.asarray(pos[i],dtype=np.float32); n=np.asarray(nrm[i],dtype=np.float32)
            mag=float(np.linalg.norm(n))
            if mag<1e-7: continue
            q=p+(n/mag)*scale
            glVertex3f(float(p[0]),float(p[1]),float(p[2])); glVertex3f(float(q[0]),float(q[1]),float(q[2]))
        glEnd(); glEnable(GL_LIGHTING)

    @staticmethod
    def _draw_skeleton_debug(model, globals_):
        if not model.bones or not globals_: return
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glLineWidth(2.0); glColor4f(1.0,0.9,0.15,1.0); glBegin(GL_LINES)
        for i,b in enumerate(model.bones):
            if i>=len(globals_): continue
            parent=int(b.parent)
            if 0<=parent<len(globals_):
                a=np.asarray(globals_[parent],dtype=np.float32)[:3,3]
                q=np.asarray(globals_[i],dtype=np.float32)[:3,3]
                glVertex3f(float(a[0]),float(a[1]),float(a[2])); glVertex3f(float(q[0]),float(q[1]),float(q[2]))
        glEnd(); glPointSize(5.0); glColor4f(1.0,0.45,0.1,1.0); glBegin(GL_POINTS)
        for i in range(min(len(model.bones),len(globals_))):
            p=np.asarray(globals_[i],dtype=np.float32)[:3,3]; glVertex3f(float(p[0]),float(p[1]),float(p[2]))
        glEnd(); glPointSize(1.0); glLineWidth(1.0); glEnable(GL_LIGHTING)

    @staticmethod
    def _draw_world_markers(matrices, rgba=(0.15,1.0,0.35,1.0), size=8.0):
        mats=[np.asarray(M,dtype=np.float32) for M in matrices if M is not None]
        if not mats:return
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glDisable(GL_BLEND)
        glPointSize(float(size)); glColor4f(*rgba); glBegin(GL_POINTS)
        for M in mats:
            p=M[:3,3]; glVertex3f(float(p[0]),float(p[1]),float(p[2]))
        glEnd(); glPointSize(1.0); glEnable(GL_LIGHTING)

    def _draw_vehicle_mount_debug(self, V):
        if not self.vehicle or not self.show_debug_mounts:return
        mats=[V@self.vehicle.driver_matrix]
        mats.extend(V@w.matrix for w in self.vehicle.wheels)
        for ms in self.vehicle.models:
            if ms.role in ("Attachment","LooseMount","AttachmentPreview"):
                mats.append(V@ms.matrix)
        if self.debug_xray_overlays:glDisable(GL_DEPTH_TEST)
        self._draw_world_markers(mats,(0.15,1.0,0.35,1.0),9.0)
        if self.debug_xray_overlays:glEnable(GL_DEPTH_TEST)

    def _draw_character_mount_debug(self, C, globals_):
        if not self.character or not self.show_debug_mounts or not globals_:return
        terms=("mount","attach","socket","hand","head","weapon","prop","driver")
        mats=[]
        for i,b in enumerate(self.character.model.bones):
            if i>=len(globals_):continue
            if any(t in str(b.name).lower() for t in terms):
                mats.append(C@np.asarray(globals_[i],dtype=np.float32))
        if self.debug_xray_overlays:glDisable(GL_DEPTH_TEST)
        self._draw_world_markers(mats,(0.2,1.0,0.45,1.0),8.0)
        if self.debug_xray_overlays:glEnable(GL_DEPTH_TEST)

    @staticmethod
    def _ray_aabb(origin, direction, mn, mx):
        tmin=-1.0e30; tmax=1.0e30
        for i in range(3):
            o=float(origin[i]); d=float(direction[i]); lo=float(mn[i]); hi=float(mx[i])
            if abs(d)<1.0e-9:
                if o<lo or o>hi:return None
                continue
            a=(lo-o)/d; b=(hi-o)/d
            if a>b:a,b=b,a
            tmin=max(tmin,a); tmax=min(tmax,b)
            if tmin>tmax:return None
        if tmax<0:return None
        return tmin if tmin>=0 else tmax

    def _pick_map_object(self, x, y):
        if self.environment is None:return
        try:
            self.makeCurrent()
            viewport=glGetIntegerv(GL_VIEWPORT)
            model=glGetDoublev(GL_MODELVIEW_MATRIX); proj=glGetDoublev(GL_PROJECTION_MATRIX)
            yy=float(viewport[3])-float(y)
            p0=np.asarray(gluUnProject(float(x),yy,0.0,model,proj,viewport),dtype=np.float64)
            p1=np.asarray(gluUnProject(float(x),yy,1.0,model,proj,viewport),dtype=np.float64)
            d=p1-p0; dm=float(np.linalg.norm(d))
            if dm<1e-10:return
            d/=dm
            best=None; best_t=1.0e30
            for ms in self.environment.models:
                if bool(getattr(ms,"skybox",False)):continue
                if id(ms) in self.hidden_map_model_ids:continue
                if self.isolate_selected_map_model and self.selected_map_model is not None and ms is not self.selected_map_model:continue
                if ms.layer=="Gameplay" and not self.show_island_gameplay:continue
                if ms.layer=="Animated" and not self.show_island_animated:continue
                if ms.layer=="Environment" and not self.show_island_environment:continue
                try:
                    inv=np.linalg.inv(np.asarray(ms.matrix,dtype=np.float64))
                    lo=(inv@np.asarray((p0[0],p0[1],p0[2],1.0)))[:3]
                    ld=(inv@np.asarray((d[0],d[1],d[2],0.0)))[:3]
                    mn,mx=ms.model.bounds(); t=self._ray_aabb(lo,ld,mn,mx)
                    if t is not None and t<best_t:
                        best_t=t; best=ms
                except Exception:
                    continue
            self.selected_map_model=best
            if best is None:
                self.selection_message.emit("No map object selected")
            else:
                pos=np.asarray(best.matrix,dtype=np.float32)[:3,3]
                self.selection_message.emit(
                    f"Selected map object: {best.name}\n"
                    f"Layer: {best.layer}\nModel: {best.model_ref}\nSource entity: {best.source}\n"
                    f"Position: {[round(float(v),3) for v in pos]}\n"
                    f"Meshes: {len(best.model.meshes)} • Bones: {len(best.model.bones)}"
                )
            self.update()
        finally:
            try:self.doneCurrent()
            except Exception:pass

    def _draw_selected_map_debug(self):
        if self.environment is None:return
        if self.show_scene_bounds:
            self._draw_local_aabb(self.environment.bounds_min,self.environment.bounds_max,(0.2,0.75,1.0,0.9),2.0)
        ms=self.selected_map_model
        if ms is not None:
            glPushMatrix(); glMultMatrixf(_matrix_gl(ms.matrix))
            mn,mx=ms.model.bounds(); self._draw_local_aabb(mn,mx,(1.0,0.2,0.85,1.0),3.0)
            glPopMatrix()

    def _draw_model(self,model:ModelData,M,anim=None,frame=0.0,override_color=None,globals_override=None,material_context="generic",wheel_name="",loop_override=None):
        if model is None: return
        glPushMatrix(); glMultMatrixf(_matrix_gl(M))
        globals_=globals_override if globals_override is not None else (animation_globals(model,anim,frame,loop_override=loop_override) if model.bones else [])
        bone_visible=(animation_bone_visibility(model,anim,frame,loop_override)
                      if (anim is not None and model.bones) else None)
        if material_context=="custom" and bone_visible is not None and all(bone_visible):
            bone_visible=None

        # Vector Unit models can serialize additive eye/specular and transparent
        # lens meshes before the opaque face they are meant to sit on. Drawing
        # strictly in file order caused Hula_SkinA's EyeSpec pass to be erased
        # by the face and Hula_SkinHW's lens to be composited before her eyes.
        # Keep opaque geometry first and defer only authored overlay/lens roles.
        normal_meshes=[]; overlay_meshes=[]
        for source_mesh in model.meshes:
            mesh=self._mesh_for_animation_visibility(source_mesh,bone_visible) if bone_visible is not None else source_mesh
            if mesh is None:
                continue
            binding=self.materials.resolve_mesh(mesh.name,material_context,wheel_name) if self.materials and material_context!="custom" else None
            shader=str(binding.material.shader if binding else '').lower()
            nl=str(mesh.name or '').lower()
            is_overlay=('eyespec' in nl or 'lens' in nl or
                        (binding is not None and binding.transparent) or
                        shader.endswith('/envmapskinned'))
            (overlay_meshes if is_overlay else normal_meshes).append(mesh)
        for mesh in normal_meshes+overlay_meshes:
            if material_context=="custom" and model.bones:
                pos,nrm=skin_custom_mesh(mesh,model,globals_,bone_visible=bone_visible)
            else:
                pos,nrm=(skin_mesh_and_normals(mesh,model,globals_,bone_visible=bone_visible)
                         if model.bones else (mesh.positions,mesh.normals))
            self._draw_mesh(mesh,pos,nrm,override_color,material_context,wheel_name)
            if material_context in ("vehicle","wheel","attachment","character","custom"):
                xray=bool(self.debug_xray_overlays and (self.show_debug_wireframe or self.show_debug_normals))
                if xray:glDisable(GL_DEPTH_TEST)
                if self.show_debug_wireframe: self._draw_wireframe_mesh(mesh,pos)
                if self.show_debug_normals: self._draw_normals_mesh(pos,nrm)
                if xray:glEnable(GL_DEPTH_TEST)
        if material_context in ("vehicle","wheel","attachment","character","custom"):
            xray=bool(self.debug_xray_overlays and (self.show_debug_skeleton or self.show_debug_bounds))
            if xray:glDisable(GL_DEPTH_TEST)
            if self.show_debug_skeleton and model.bones: self._draw_skeleton_debug(model,globals_)
            if self.show_debug_bounds:
                mn,mx=model.bounds(); self._draw_local_aabb(mn,mx,(0.2,0.8,1.0,0.95),1.5)
            if xray:glEnable(GL_DEPTH_TEST)
        glPopMatrix()

    def _draw_envmap_lens_mesh(self,mesh,pos,nrm,binding):
        shader=str(binding.material.shader or '').strip().lower()
        if shader != 'art/cardriver/envmapskinned' or binding.diffuse_texture is not None:
            return False
        c=np.asarray(binding.material.diffuse_color,dtype=np.float32)
        # The target shader samples a cubemap plus Fresnel texture. Fixed
        # function OpenGL has no equivalent cube-map material path here, so use
        # the authored blue tint/alpha with a view-normal Fresnel approximation
        # instead of rendering the spectacle lens as a flat opaque polygon.
        mv=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order='F')
        try:
            inv=np.linalg.inv(mv); eye=(inv@np.asarray((0,0,0,1),dtype=np.float32))[:3]
        except Exception:
            eye=np.asarray((0,-5,2),dtype=np.float32)
        tint=np.maximum(c[:3],np.asarray((0.08,0.20,0.42),dtype=np.float32))
        base_alpha=max(0.10,min(0.36,float(c[3])*0.34))
        glDisable(GL_TEXTURE_2D); glDisable(GL_LIGHTING); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glBegin(GL_TRIANGLES)
        for tri in idx:
            for vi in tri:
                i=int(vi)
                if i<0 or i>=len(pos): continue
                nn=np.asarray(nrm[i] if i<len(nrm) else (0,1,0),dtype=np.float32)
                nl=float(np.linalg.norm(nn)); nn=nn/nl if nl>1e-7 else np.asarray((0,1,0),dtype=np.float32)
                vv=np.asarray(eye-pos[i],dtype=np.float32); vl=float(np.linalg.norm(vv)); vv=vv/vl if vl>1e-7 else np.asarray((0,-1,0),dtype=np.float32)
                fres=(1.0-max(0.0,min(1.0,abs(float(np.dot(nn,vv))))))**2.2
                rgb=np.clip(tint*(0.70+0.25*fres)+np.asarray((0.65,0.78,1.0),dtype=np.float32)*(0.38*fres),0,1)
                alpha=min(0.58,base_alpha+0.28*fres)
                glColor4f(float(rgb[0]),float(rgb[1]),float(rgb[2]),float(alpha))
                glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
        glEnd(); glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING)
        return True

    def _draw_additive_mask_pass(self,mesh,pos,nrm,binding):
        shader=str(binding.material.shader or '').strip().lower()
        if 'diffuseadditivemask' not in shader or binding.mask_texture is None:
            return
        # This is the second pass the target DiffuseAdditiveMask* shaders need.
        # Hula_SkinA_EyeSpec is the clearest case: the base atlas contains the
        # eye surface while the red mask gates its bright additive reflection.
        tid=self._texture_id_data(binding.mask_texture,binding.mask_cache_key or ('mask',id(binding.mask_texture)),clamp=False)
        if not tid: return
        glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_ONE,GL_ONE); glDepthMask(GL_FALSE)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); self._apply_mesh_texture_wrap(mesh)
        # Conservative white environment proxy; mask texture supplies coverage.
        glColor4f(0.58,0.58,0.58,1.0)
        idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glBegin(GL_TRIANGLES)
        for tri in idx:
            for vi in tri:
                i=int(vi)
                if i<0 or i>=len(pos): continue
                if i<len(mesh.uvs): glTexCoord2f(float(mesh.uvs[i,0]),float(mesh.uvs[i,1]))
                glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
        glEnd(); glDepthMask(GL_TRUE); glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING)

    def _draw_environment_cube_mesh(self, mesh, pos, nrm, binding, c):
        cube = getattr(binding, "env_cube", None) or getattr(binding, "additive_env_cube", None)
        if cube is None or getattr(binding, "diffuse_texture", None) is not None:
            return False
        key=(getattr(binding,"env_cube_cache_key",None) or
             getattr(binding,"additive_env_cube_cache_key",None) or
             ("cube-synthetic",id(cube)))
        tid=self._cube_texture_id_data(cube,key)
        if not tid:return False
        transparent=(bool(binding.transparent) if binding else False) or float(c[3])<.995
        if transparent:
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        else:
            glDisable(GL_BLEND); glDepthMask(GL_TRUE)
        glDisable(GL_TEXTURE_2D); glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP,tid)
        glEnable(GL_TEXTURE_GEN_S); glEnable(GL_TEXTURE_GEN_T); glEnable(GL_TEXTURE_GEN_R)
        glTexGeni(GL_S,GL_TEXTURE_GEN_MODE,GL_REFLECTION_MAP)
        glTexGeni(GL_T,GL_TEXTURE_GEN_MODE,GL_REFLECTION_MAP)
        glTexGeni(GL_R,GL_TEXTURE_GEN_MODE,GL_REFLECTION_MAP)
        idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glBegin(GL_TRIANGLES)
        for tri in idx:
            for vi in tri:
                i=int(vi)
                if i<0 or i>=len(pos):continue
                if i<len(nrm):glNormal3f(float(nrm[i,0]),float(nrm[i,1]),float(nrm[i,2]))
                if (getattr(binding,"uses_vertex_color",False) and getattr(mesh,"colors",None) is not None and i<len(mesh.colors)):
                    vc=mesh.colors[i]; glColor4f(float(c[0]*vc[0]),float(c[1]*vc[1]),float(c[2]*vc[2]),float(c[3]*vc[3]))
                else:glColor4f(*[float(x) for x in c[:4]])
                glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
        glEnd()
        glDisable(GL_TEXTURE_GEN_S); glDisable(GL_TEXTURE_GEN_T); glDisable(GL_TEXTURE_GEN_R); glDisable(GL_TEXTURE_CUBE_MAP)
        glDepthMask(GL_TRUE); glDisable(GL_BLEND)
        return True

    def _draw_environment_reflection_mesh(self,mesh,pos,nrm,binding,c):
        if not self.environment_program or not binding.dynamic_environment or binding.env_cube is None:
            return False
        cube=self._cube_texture_id_data(binding.env_cube,binding.env_cube_cache_key or ("cube",id(binding.env_cube)))
        if not cube: return False
        base=self._texture_id(binding)
        mask=self._texture_id_data(binding.mask_texture,binding.mask_cache_key or ("mask",id(binding.mask_texture)))
        fresnel=self._texture_id_data(binding.fresnel_texture,binding.fresnel_cache_key or ("fresnel",id(binding.fresnel_texture)),clamp=True)
        additive=self._cube_texture_id_data(binding.additive_env_cube,binding.additive_env_cube_cache_key or ("cube",id(binding.additive_env_cube)))
        dome_shell=str(mesh.name or "").startswith("Building/Alien/Alien_DomeExterior")
        if dome_shell: glDisable(GL_CULL_FACE)
        transparent=bool(binding.transparent) or float(c[3])<0.995
        if transparent:
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        else:
            glDisable(GL_BLEND); glDepthMask(GL_TRUE)
        glUseProgram(self.environment_program)
        u=self.environment_uniforms
        try:
            for name,unit in (("uBase",0),("uMask",1),("uFresnel",2),("uCube",3),("uAddCube",4)):
                glUniform1i(u[name],unit)
            for name,available in (("uHasBase",base),("uHasMask",mask),("uHasFresnel",fresnel),("uHasAddCube",additive)):
                glUniform1i(u[name],1 if available else 0)
            glUniform1i(u["uDomeShell"],1 if dome_shell else 0)
            emissive=(float(binding.material.emissive_amount or 0.0)
                      if "emissive" in str(binding.material.shader or "").lower() else 0.0)
            glUniform1f(u["uEmissive"],emissive)
            glUniform4f(u["uDiffuse"],*[float(x) for x in c[:4]])
            self._reflection_matrix_uniform(u)
            for unit,target,tid in ((0,GL_TEXTURE_2D,base),(1,GL_TEXTURE_2D,mask),
                                    (2,GL_TEXTURE_2D,fresnel),(3,GL_TEXTURE_CUBE_MAP,cube),
                                    (4,GL_TEXTURE_CUBE_MAP,additive)):
                if not tid: continue
                glActiveTexture(GL_TEXTURE0+unit); glEnable(target); glBindTexture(target,tid)
                if unit==0: self._apply_mesh_texture_wrap(mesh)
            glActiveTexture(GL_TEXTURE0)
            cacheable=(pos is mesh.positions and nrm is mesh.normals and self._map_display_lists_supported)
            list_id=self.reflection_geometry_lists.get(id(mesh)) if cacheable else None
            if cacheable and list_id is None:
                list_id=int(glGenLists(1))
                if list_id:
                    glNewList(list_id,GL_COMPILE)
            if list_id is None or (cacheable and id(mesh) not in self.reflection_geometry_lists):
                idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
                glBegin(GL_TRIANGLES)
                for tri in idx:
                    for vi in tri:
                        i=int(vi)
                        if i<0 or i>=len(pos): continue
                        if i<len(nrm): glNormal3f(float(nrm[i,0]),float(nrm[i,1]),float(nrm[i,2]))
                        if binding.uses_vertex_color and getattr(mesh,"colors",None) is not None and i<len(mesh.colors):
                            vc=mesh.colors[i]; glColor4f(*[float(x) for x in vc[:4]])
                        else: glColor4f(1.0,1.0,1.0,1.0)
                        if i<len(mesh.uvs): glTexCoord2f(float(mesh.uvs[i,0]),float(mesh.uvs[i,1]))
                        glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
                glEnd()
                if list_id:
                    glEndList(); self.reflection_geometry_lists[id(mesh)]=list_id
            if list_id: glCallList(list_id)
        finally:
            glUseProgram(0)
            for unit,target,tid in ((4,GL_TEXTURE_CUBE_MAP,additive),(3,GL_TEXTURE_CUBE_MAP,cube),
                                    (2,GL_TEXTURE_2D,fresnel),(1,GL_TEXTURE_2D,mask),(0,GL_TEXTURE_2D,base)):
                if tid:
                    glActiveTexture(GL_TEXTURE0+unit); glDisable(target)
            glActiveTexture(GL_TEXTURE0); glDepthMask(GL_TRUE); glDisable(GL_BLEND)
            if dome_shell and self.debug_backface_culling: glEnable(GL_CULL_FACE)
        return True

    def _draw_environment_emissive_pass(self,mesh,pos,binding):
        shader=str(binding.material.shader or '').strip().lower()
        if shader == 'art/world/emissiveenvmapnodetail': return
        if 'emissive' not in shader or binding.mask_texture is None:return
        amount=max(0.0,min(2.0,float(getattr(binding.material,'emissive_amount',0.0) or 0.0)))
        if amount<=0.001:return
        tid=self._texture_id_data(binding.mask_texture,binding.mask_cache_key or ('emissive-mask',id(binding.mask_texture)),clamp=False)
        if not tid:return
        glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_ONE,GL_ONE); glDepthMask(GL_FALSE)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); self._apply_mesh_texture_wrap(mesh)
        gain=min(0.72,0.22+0.24*amount); glColor4f(gain,gain,gain,1.0)
        idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glBegin(GL_TRIANGLES)
        for tri in idx:
            for vi in tri:
                i=int(vi)
                if i<0 or i>=len(pos):continue
                if i<len(mesh.uvs):glTexCoord2f(float(mesh.uvs[i,0]),float(mesh.uvs[i,1]))
                glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
        glEnd(); glDepthMask(GL_TRUE); glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING)

    def _draw_mesh(self,mesh,pos,nrm,override_color=None,material_context="generic",wheel_name=""):
        if not len(pos) or not len(mesh.indices): return
        binding=self.materials.resolve_mesh(mesh.name,material_context,wheel_name) if self.materials and material_context!="custom" else None
        if binding:
            c=np.asarray(binding.material.diffuse_color,dtype=np.float32).copy()
        else: c=np.asarray((.82,.82,.84,1),dtype=np.float32)
        # Custom skin overrides replace only the diffuse image sampled by the
        # existing mesh UVs. Geometry, UVs, bones, weights and animations are
        # untouched. A package can contain several named textures; match them
        # to the authored material texture name or mesh name.
        custom_tex = None
        if self.custom_skin_textures and material_context in ("character", "vehicle"):
            material_tex_name = ""
            try:
                material_tex_name = str(getattr(binding.material, "diffuse_texture", "") or "") if binding else ""
            except Exception:
                material_tex_name = ""
            custom_tex = match_texture(self.custom_skin_textures, getattr(mesh, "name", ""), material_tex_name)
        if material_context=="custom" and getattr(mesh,"custom_color",None) is not None:
            c=np.asarray(mesh.custom_color,dtype=np.float32)
        if override_color is not None: c=np.asarray(override_color,dtype=np.float32)
        if override_color is None and binding and self._draw_envmap_lens_mesh(mesh,pos,nrm,binding):
            return
        if override_color is None and binding and binding.carpaint:
            if self._draw_carpaint_mesh(mesh,pos,nrm,binding,c):
                return
        if override_color is None and binding and self._draw_environment_reflection_mesh(mesh,pos,nrm,binding,c):
            return
        if override_color is None and binding and self._draw_environment_cube_mesh(mesh,pos,nrm,binding,c):
            return
        # Fixed-function fallback is deliberately conservative: unknown shaders
        # keep authored diffuse color/texture; no border alpha keys or guessed
        # decal blend are fabricated.
        alpha_test=bool(binding.alpha_test) if binding else False
        transparent=(bool(binding.transparent) if binding else False) or float(c[3])<.995
        if alpha_test:
            # Vector Unit *1bit* materials are hard cutouts, not blended glass.
            # This is required by Tribal_SkinA's DiffuseEnvMap1bit shader.
            glDisable(GL_BLEND); glDepthMask(GL_TRUE)
            glEnable(GL_ALPHA_TEST); glAlphaFunc(GL_GREATER,0.5)
        elif transparent:
            glDisable(GL_ALPHA_TEST)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        else:
            glDisable(GL_ALPHA_TEST); glDisable(GL_BLEND); glDepthMask(GL_TRUE)
        tid=self._texture_id(binding) if binding else 0
        if custom_tex is not None:
            tid=self._texture_id_data(custom_tex,("custom-skin",self.custom_skin_source,custom_tex.name),clamp=False)
        elif material_context=="custom" and getattr(mesh,"custom_texture",None) is not None:
            tex=mesh.custom_texture
            tid=self._texture_id_data(tex,("fbx-image",id(tex)),clamp=False)
        detail_tid=0; detail_scale=1.0
        if binding and binding.detail_texture is not None and tid:
            detail_tid=self._texture_id_data(binding.detail_texture,binding.detail_cache_key or ('detail',id(binding.detail_texture)),clamp=False)
            detail_scale=max(0.01,min(64.0,float(getattr(binding,'detail_uv_scale',1.0) or 1.0)))
        glActiveTexture(GL_TEXTURE0)
        if tid:
            glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); self._apply_mesh_texture_wrap(mesh)
        else:glDisable(GL_TEXTURE_2D)
        if detail_tid:
            glActiveTexture(GL_TEXTURE1); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,detail_tid)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT)
            # Art/World/Basic_Normal* adds (detail.rgb - 0.5) to the base.
            # Keep the detail texture separate so scaled UVs tile correctly
            # even when the base texture is shared with another material.
            glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_COMBINE)
            glTexEnvi(GL_TEXTURE_ENV,GL_COMBINE_RGB,GL_ADD_SIGNED)
            glTexEnvi(GL_TEXTURE_ENV,GL_SOURCE0_RGB,GL_PREVIOUS)
            glTexEnvi(GL_TEXTURE_ENV,GL_SOURCE1_RGB,GL_TEXTURE)
            glTexEnvi(GL_TEXTURE_ENV,GL_COMBINE_ALPHA,GL_REPLACE)
            glTexEnvi(GL_TEXTURE_ENV,GL_SOURCE0_ALPHA,GL_PREVIOUS)
            glTexEnvf(GL_TEXTURE_ENV,GL_RGB_SCALE,1.0)
            glActiveTexture(GL_TEXTURE0)
        glColor4f(*[float(x) for x in c[:4]])
        idx=np.asarray(mesh.indices,dtype=np.uint32).reshape((-1,3))
        glBegin(GL_TRIANGLES)
        for tri in idx:
            for vi in tri:
                i=int(vi)
                if i<0 or i>=len(pos):continue
                if i<len(nrm):glNormal3f(float(nrm[i,0]),float(nrm[i,1]),float(nrm[i,2]))
                if (binding and getattr(binding,"uses_vertex_color",False) and getattr(mesh,"colors",None) is not None and i<len(mesh.colors)):
                    vc=mesh.colors[i]; glColor4f(float(c[0]*vc[0]),float(c[1]*vc[1]),float(c[2]*vc[2]),float(c[3]*vc[3]))
                else:glColor4f(*[float(x) for x in c[:4]])
                if i<len(mesh.uvs):
                    u=float(mesh.uvs[i,0]); v=float(mesh.uvs[i,1]); glTexCoord2f(u,v)
                    if detail_tid:
                        try:glMultiTexCoord2f(GL_TEXTURE1,u*detail_scale,v*detail_scale)
                        except Exception:pass
                glVertex3f(float(pos[i,0]),float(pos[i,1]),float(pos[i,2]))
        glEnd()
        if detail_tid:
            glActiveTexture(GL_TEXTURE1)
            glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE); glDisable(GL_TEXTURE_2D); glActiveTexture(GL_TEXTURE0)
        glDepthMask(GL_TRUE); glDisable(GL_ALPHA_TEST); glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
        if override_color is None and binding:
            if 'eyespec' in str(mesh.name or '').lower():self._draw_additive_mask_pass(mesh,pos,nrm,binding)
            elif material_context in ('generic','environment'):self._draw_environment_emissive_pass(mesh,pos,binding)

    def _draw_wheel(self,w:WheelSpec,V):
        M=w.matrix.copy()
        if self.drive:
            s=self.drive.state; c=M[:3,3].copy()
            steer=0.0
            if w.axle=="front": steer=s.front_left_steer_deg if w.side=="left" else s.front_right_steer_deg
            spin=s.wheel_spin_front_deg if w.axle=="front" else s.wheel_spin_rear_deg
            T=_translation(*c); Ti=_translation(*(-c))
            M=T@_rot_z(steer)@_rot_x(spin)@Ti@M
        self._draw_model(w.model,V@M,material_context="wheel",wheel_name=w.wheel_name)

    def _draw_coronas(self,V):
        if not self.vehicle:return
        # v0.73 corona pass: the old preview drew one hard sprite per lamp,
        # which made headlights/brakelights look like stickers. Keep the exact
        # authored texture/color/mount, but layer a soft halo + core and apply a
        # modest distance compensation so the glow reads like a light source.
        mv=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order='F')
        for e in self.vehicle.emitters:
            if not e.kind.endswith("_corona") or not e.texture_name:continue
            color=tuple(float(x) for x in e.color)
            if e.kind=="brakelight_corona":
                amount=float(self.drive.state.brake_intensity) if self.drive else 0.0
                if amount<=0.015:continue
                color=(color[0],color[1],color[2],color[3]*amount)
            rec=self.db.resolve_texture(e.texture_name) if self.db else None
            tex=self.db.texture(rec) if rec else None
            if tex is None:continue
            key=(rec.archive_index,rec.entry_index)
            if key not in self.texture_ids:
                class B:pass
                b=B(); b.diffuse_texture_record=rec; b.diffuse_texture=tex
                tid=self._texture_id(b)
            else:tid=self.texture_ids[key]
            P=V@e.matrix; p=np.asarray(P[:3,3],dtype=np.float32)
            if not self.enhanced_vehicle_coronas:
                self._draw_billboard(p,e.texture_size,color,tid,blend_mode="Additive")
                continue
            ep=mv@np.asarray((p[0],p[1],p[2],1.0),dtype=np.float32)
            dist=max(0.0,float(np.linalg.norm(ep[:3])))
            screen_comp=max(1.0,min(2.25,1.0+dist*0.018))
            base=max(0.02,float(e.texture_size))*screen_comp
            a=max(0.0,min(1.0,float(color[3])))
            rgb=color[:3]
            # Outer halo is deliberately faint; the same authored corona
            # texture provides the falloff, so no synthetic radial bitmap is used.
            self._draw_billboard(p,base*2.10,(rgb[0],rgb[1],rgb[2],a*0.18),tid,blend_mode="Additive")
            self._draw_billboard(p,base*1.25,(rgb[0],rgb[1],rgb[2],a*0.42),tid,blend_mode="Additive")
            self._draw_billboard(p,base*0.62,(rgb[0],rgb[1],rgb[2],min(1.0,a*1.15)),tid,blend_mode="Additive")

    def _draw_billboard(self,p,size,color,tid,blend_mode="Additive",rotation=0.0,velocity=None,directional_stretch=0.0,world_scale_z=1.0,uv_offset_v=0.0):
        # Camera-facing quad from current model-view right/up vectors.  Blend
        # mode comes from the decoded PFX pattern; v0.43/v0.44's one-size-fits-
        # all additive treatment was a major source of square-looking cards.
        mv=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order='F')
        right=mv[0,:3]; up=mv[1,:3]
        nr=np.linalg.norm(right); nu=np.linalg.norm(up)
        if nr>1e-6:right=right/nr
        if nu>1e-6:up=up/nu
        a=math.radians(rotation); r2=right*math.cos(a)+up*math.sin(a); u2=-right*math.sin(a)+up*math.cos(a)
        half=max(.001,float(size)*.5)
        # Directional-stretch is authored on spark/smoke emitters.  Older
        # previews ignored it, turning fast sparks into square cards.
        stretch=max(0.0,float(directional_stretch or 0.0))
        if stretch>0.0 and velocity is not None:
            vel=np.asarray(velocity,dtype=np.float32)
            vr=float(np.dot(vel,right)); vu=float(np.dot(vel,up))
            mag2=math.hypot(vr,vu)
            if mag2>1e-5:
                dr=vr/mag2; du=vu/mag2
                axis=right*dr+up*du
                perp=-right*du+up*dr
                length=half*(1.0+min(6.0,float(np.linalg.norm(vel))*stretch))
                r2=axis*length; u2=perp*half
            else:
                r2*=half; u2*=half
        else:
            r2*=half; u2*=half
        # VuPfxTickWorldScaleZ grows smoke in the vertical billboard axis.
        # This is especially visible in AstroDog_Liftoff and powerslide dust.
        u2*=max(0.01,float(world_scale_z or 1.0))
        glDisable(GL_LIGHTING); glEnable(GL_BLEND)
        mode=str(blend_mode or "Additive").strip().lower()
        if mode in ("add", "additive"):
            glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        else:
            # The target database's explicit non-default pattern mode is
            # Modulate.  For the fixed-function preview this is represented as
            # authored-alpha compositing rather than inventing a DST_COLOR
            # equation that the BIN does not specify.
            glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        if tid: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid)
        else: glDisable(GL_TEXTURE_2D)
        glColor4f(*[float(x) for x in color])
        glBegin(GL_QUADS)
        vo=float(uv_offset_v or 0.0)
        for uv,v in [((0,0+vo),p-r2-u2),((1,0+vo),p+r2-u2),((1,1+vo),p+r2+u2),((0,1+vo),p-r2+u2)]:
            glTexCoord2f(*uv); glVertex3f(float(v[0]),float(v[1]),float(v[2]))
        glEnd(); glDepthMask(GL_TRUE); glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING)

    def _radial_glow_texture_id(self):
        key=("procedural","radial_glow_v073")
        if key in self.texture_ids:return self.texture_ids[key]
        n=64
        y,x=np.mgrid[0:n,0:n].astype(np.float32)
        cx=(n-1)*0.5; cy=(n-1)*0.5
        r=np.sqrt(((x-cx)/cx)**2+((y-cy)/cy)**2)
        a=np.clip(1.0-r,0.0,1.0)**2.2
        rgba=np.empty((n,n,4),dtype=np.uint8); rgba[:,:,:3]=255; rgba[:,:,3]=np.clip(a*255.0,0,255).astype(np.uint8)
        class T:pass
        tex=T(); tex.rgba=np.ascontiguousarray(rgba); tex.width=n; tex.height=n
        return self._texture_id_data(tex,key,clamp=True)

    def _pfx_texture_id(self,pat):
        if not pat:
            return 0

        # Vector's flame quad shader uses Texture Asset as the coverage/flare
        # mask and Tile Texture Asset as the scrolling flame colour. v0.61
        # displayed the tile by itself. Fire_Flame is a full tile, so that
        # produced the obvious square cards around Tribal/AstroDog flames.
        # Build one fixed-function approximation that keeps the authored flame
        # RGB but multiplies it by the authored coverage mask alpha.
        base_rec=getattr(pat,"texture_record",None)
        base_tex=getattr(pat,"texture",None)
        tile_rec=getattr(pat,"tile_texture_record",None)
        tile_tex=getattr(pat,"tile_texture",None)
        if base_rec is not None and base_tex is not None and tile_rec is not None and tile_tex is not None:
            key=("pfx-masktile",base_rec.archive_index,base_rec.entry_index,
                 tile_rec.archive_index,tile_rec.entry_index)
            if key in self.texture_ids:
                return self.texture_ids[key]
            try:
                base=np.asarray(base_tex.rgba,dtype=np.uint8)
                tile=np.asarray(tile_tex.rgba,dtype=np.uint8)
                th,tw=tile.shape[:2]; bh,bw=base.shape[:2]
                yi=np.minimum(bh-1,(np.arange(th,dtype=np.int64)*bh)//max(1,th))
                xi=np.minimum(bw-1,(np.arange(tw,dtype=np.int64)*bw)//max(1,tw))
                mask=base[yi[:,None],xi[None,:],:]
                out=tile.copy()
                ma=mask[:,:,3].astype(np.float32)
                # Some mask textures are serialized with opaque alpha and put
                # the coverage in RGB. Fall back to luminance in that case.
                if float(ma.max()-ma.min()) < 1.0:
                    ma=np.max(mask[:,:,:3].astype(np.float32),axis=2)
                ta=out[:,:,3].astype(np.float32)
                out[:,:,3]=np.clip((ma*ta)/255.0,0.0,255.0).astype(np.uint8)
                class T: pass
                tex=T(); tex.rgba=np.ascontiguousarray(out); tex.width=int(tw); tex.height=int(th)
                return self._texture_id_data(tex,key,clamp=True)
            except Exception:
                pass

        # Ordinary quad patterns use their authored Texture Asset directly.
        # In particular this preserves Cloud/Cloud_cartoon alpha and prevents
        # the tile-texture regression from reintroducing square fire cards.
        rec=base_rec or tile_rec
        tex=base_tex or tile_tex
        if not rec or tex is None:
            return 0
        key=("pfx",rec.archive_index,rec.entry_index)
        return self._texture_id_data(tex,key,clamp=True)

    def _draw_pfx_trails(self, sys:PfxSystem50, pat_by, color_mul, scale_mul):
        if not getattr(sys,"trail_points",None):
            return
        mv=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order='F')
        right=mv[0,:3]
        nr=float(np.linalg.norm(right))
        if nr>1e-6:
            right=right/nr
        for name,pat in pat_by.items():
            if getattr(pat,"type","")!="VuPfxTrailPattern":
                continue
            pts=[x for x in sys.trail_points if x.pattern_name==name]
            if len(pts)<2:
                continue
            props=pat.properties or {}
            width=max(0.001,float(props.get("Width") or 0.1))*max(0.0,float(scale_mul))
            fade_out=max(0.05,float(props.get("Fade Out Time") or 0.9))
            base=np.asarray(props.get("Color") or (1,1,1,1),dtype=np.float32)
            if base.size<4: base=np.asarray((1,1,1,1),dtype=np.float32)
            base=np.clip(base*np.asarray(color_mul,dtype=np.float32)[:4],0.0,1.0)
            tid=self._pfx_texture_id(pat)
            glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE); glDepthMask(GL_FALSE)
            if tid: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid)
            else: glDisable(GL_TEXTURE_2D)
            glBegin(GL_QUAD_STRIP)
            n=max(1,len(pts)-1)
            for i,tp in enumerate(pts):
                alpha=max(0.0,min(1.0,1.0-float(tp.age)/fade_out))
                c=base.copy(); c[3]*=alpha
                glColor4f(float(c[0]),float(c[1]),float(c[2]),float(c[3]))
                p=np.asarray(tp.position,dtype=np.float32)
                off=right*(width*0.5)
                v=float(i)/float(n)
                glTexCoord2f(0.0,v); glVertex3f(float(p[0]-off[0]),float(p[1]-off[1]),float(p[2]-off[2]))
                glTexCoord2f(1.0,v); glVertex3f(float(p[0]+off[0]),float(p[1]+off[1]),float(p[2]+off[2]))
            glEnd()
            glDepthMask(GL_TRUE); glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING)

    def _draw_pfx_system(self,sys:PfxSystem50,base_matrix:np.ndarray,color_mul=(1.0,1.0,1.0,1.0),scale_mul:float=1.0,draw_object_space:bool=True):
        if not sys:
            return
        cm=np.asarray(color_mul,dtype=np.float32)
        if cm.size < 4:
            cm=np.asarray((1.0,1.0,1.0,1.0),dtype=np.float32)
        sm=max(0.0,float(scale_mul))
        pat_by={p.name:p for p in sys.patterns}
        self._draw_pfx_trails(sys,pat_by,cm,sm)
        # Alpha-blended smoke/cloud particles need back-to-front submission.
        # v0.72 rendered spawn order, causing dark intersections and square-ish
        # layers when multiple quads overlapped. Sort by current eye-space Z;
        # additive particles are unaffected by ordering.
        mv_sort=np.asarray(glGetFloatv(GL_MODELVIEW_MATRIX),dtype=np.float32).reshape((4,4),order='F')
        def particle_depth(x):
            if x.world_space:
                w=np.asarray(x.position,dtype=np.float32)
            else:
                w=(base_matrix@np.asarray((*x.position,1.0),dtype=np.float32))[:3]
            e=mv_sort@np.asarray((w[0],w[1],w[2],1.0),dtype=np.float32)
            return float(e[2])
        particles=sorted(sys.particles,key=particle_depth)
        for x in particles:
            pat=pat_by.get(x.pattern_name)
            if (not x.world_space) and not draw_object_space:
                continue
            if x.world_space:
                world=np.asarray(x.position,dtype=np.float32)
                particle_matrix=_translation(*world) @ _rotation_xyz(x.rotation)
            else:
                world=(base_matrix@np.asarray((*x.position,1.0),dtype=np.float32))[:3]
                particle_matrix=base_matrix @ _translation(*x.position) @ _rotation_xyz(x.rotation)
            if not np.isfinite(world).all():
                continue
            if x.kind=="quad":
                if x.world_space:
                    world_vel=np.asarray(x.velocity,dtype=np.float32)
                else:
                    world_vel=np.asarray(base_matrix[:3,:3]@np.asarray(x.velocity,dtype=np.float32),dtype=np.float32)
                draw_color=tuple(np.clip(np.asarray(x.color,dtype=np.float32)*cm[:4],0.0,1.0).tolist())
                draw_scale=float(x.scale)*(sm if (pat is None or pat.respect_scale) else 1.0)
                if draw_scale<=1.0e-5 or draw_color[3]<=1.0e-4:
                    continue
                tid=self._pfx_texture_id(pat)
                # If an authored particle references a texture but that texture
                # failed to resolve/decode, do not replace it with a solid quad.
                # That fallback is what produced the obvious square/rectangle
                # cards on several character fire/smoke effects.
                if pat is not None and (pat.texture_ref or pat.tile_texture_ref) and not tid:
                    continue
                self._draw_billboard(
                    world,draw_scale,draw_color,tid,
                    blend_mode=(pat.blend_mode if pat else "Additive"),
                    rotation=float(x.rotation[2]), velocity=world_vel,
                    directional_stretch=float(getattr(x,"directional_stretch",0.0)),
                    world_scale_z=float(getattr(x,"world_scale_z",1.0)),
                    uv_offset_v=(float(getattr(pat,"tile_scroll_speed_v",0.0))*float(sys.elapsed) if (pat and pat.tile_texture_ref and not pat.texture_ref) else 0.0)
                )
            elif x.kind=="light":
                draw_color=tuple(np.clip(np.asarray(x.color,dtype=np.float32)*cm[:4],0.0,1.0).tolist())
                if draw_color[3]<=1.0e-4:continue
                props=(pat.properties if pat else {}) or {}
                falloff=max(float(props.get("Falloff Range Min") or 0.0),float(props.get("Falloff Range Max") or 0.0))
                size=max(0.15,float(x.scale)*(sm if (pat is None or pat.respect_scale) else 1.0),falloff*0.28)
                tid=self._radial_glow_texture_id()
                # PFX light patterns are actual runtime lights in-game. The
                # fixed-function Studio cannot inject them into every material,
                # so render a soft source glow instead of dropping the pattern.
                self._draw_billboard(world,size*1.65,(draw_color[0],draw_color[1],draw_color[2],draw_color[3]*0.22),tid,blend_mode="Additive")
                self._draw_billboard(world,size*0.72,(draw_color[0],draw_color[1],draw_color[2],draw_color[3]*0.70),tid,blend_mode="Additive")
            elif x.kind=="geom" and x.model:
                # Powerslide/character geom particles carry alpha. Object-space
                # geometry already inherits the system scale through base_matrix,
                # so applying scale_mul again would double-scale it (the source
                # of several oversized ring/mesh effects in v0.64).
                glEnable(GL_BLEND)
                mode=str(pat.blend_mode if pat else "Additive").strip().lower()
                glBlendFunc(GL_SRC_ALPHA,GL_ONE if mode in ("add","additive") else GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(GL_FALSE)
                draw_color=tuple(np.clip(np.asarray(x.color,dtype=np.float32)*cm[:4],0.0,1.0).tolist())
                inherit_scale=(pat is None or pat.respect_scale)
                extra_system_scale=(sm if (inherit_scale and x.world_space) else 1.0)
                draw_scale=float(x.scale)*extra_system_scale
                if draw_scale<=1.0e-5 or draw_color[3]<=1.0e-4:
                    glDepthMask(GL_TRUE); glDisable(GL_BLEND)
                    continue
                S=np.eye(4,dtype=np.float32); S[0,0]=S[1,1]=S[2,2]=draw_scale
                self._draw_model(x.model,particle_matrix@S,override_color=draw_color)
                glDepthMask(GL_TRUE); glDisable(GL_BLEND)

    def _character_pfx_local_time(self, runtime: CharacterPfxRuntime) -> float:
        if not self.character_animation:
            return 0.0
        fps=max(1e-6,float(self.character_preview_fps or self.character_animation.fps or 30.0))
        t=max(0.0,float(self.character_frame)/fps)
        dur=max(0.0,float(runtime.spec.duration))
        loops=bool(self.force_character_loop or self.character_animation.looping)
        if dur>1e-6:
            if loops:
                t=t%dur
            else:
                t=min(t,dur)
        return t

    def _character_pfx_matrix(self, runtime: CharacterPfxRuntime, local_time: float):
        spec=runtime.spec
        pos,rot,scl=spec.transform_at(local_time)
        local=_translation(*pos) @ _rotation_xyz(rot) @ _scale_xyz(*scl)
        actor_leaf=lambda x: str(x or '').replace('\\','/').rsplit('/',1)[-1]

        # Timeline attachments can target the character body, a synchronized
        # companion rig, or another cinematic prop.  If an authored parent is
        # hidden/unavailable this frame, return None instead of falling back to
        # the character origin. That fallback was the source of several detached
        # Cyber/Tribal effect meshes in v0.64.
        binding=getattr(spec,"attachment_binding_at",lambda _t:None)(local_time)
        if binding:
            parent,bone,rel_pos,rel_rot=binding
            parent_leaf=actor_leaf(parent)
            rel=_translation(*rel_pos) @ _rotation_xyz(rel_rot)
            if self.character and parent_leaf == actor_leaf(spec.character_actor_name):
                bone_map={b.name:i for i,b in enumerate(self.character.model.bones)}
                bi=bone_map.get(bone,-1)
                if 0 <= bi < len(self._last_character_globals):
                    return self._last_character_matrix @ self._last_character_globals[bi] @ rel @ _scale_xyz(*scl)
                return None

            if (parent_leaf.startswith("PropAnimated_Screen") or
                    parent_leaf.startswith("PropAnimated_Spinner")):
                if self._last_companion_model is None:
                    return None
                bone_map={b.name:i for i,b in enumerate(self._last_companion_model.bones)}
                bi=bone_map.get(bone,-1)
                if 0 <= bi < len(self._last_companion_globals):
                    return self._last_companion_matrix @ self._last_companion_globals[bi] @ rel @ _scale_xyz(*scl)
                return None

            prop_matrix=self._last_character_prop_matrices.get(parent_leaf)
            if prop_matrix is not None:
                return prop_matrix @ rel @ _scale_xyz(*scl)

            return None

        return self._last_character_matrix @ local

    def _character_authored_cycle_duration(self) -> float:
        if not self.character_animation:
            return 0.0
        fps=max(1e-6,float(self.character_preview_fps or self.character_animation.fps or 30.0))
        asset=max(0.0,float(max(0,self.character_animation.frame_count-1))/fps)
        spans=[asset]
        spans += [max(0.0,float(x.spec.duration)) for x in self.character_pfx_instances]
        spans += [max(0.0,float(x.spec.duration)) for x in self.character_prop_instances]
        return max(spans) if spans else asset

    def _draw_character_pfx(self):
        for runtime in self.character_pfx_instances:
            local_t=self._character_pfx_local_time(runtime)
            _pos,_rot,scl=runtime.spec.transform_at(local_t)
            scale_mul=max(0.0,max(abs(float(x)) for x in scl)) if scl else 1.0
            color_mul=(runtime.spec.color_at(local_t)
                       if hasattr(runtime.spec,"color_at") else runtime.spec.pfx_color)
            M=self._character_pfx_matrix(runtime,local_t)
            if M is None:
                # Parent actor is hidden/detached. Existing world-space particles
                # may finish naturally, but object-space particles must not jump
                # to the character origin.
                self._draw_pfx_system(
                    runtime.system,np.eye(4,dtype=np.float32),
                    color_mul=color_mul,scale_mul=scale_mul,
                    draw_object_space=False
                )
                continue
            self._draw_pfx_system(
                runtime.system,M,
                color_mul=color_mul,scale_mul=scale_mul
            )

    def _step_character_pfx(self, dt: float):
        if not (self.pfxlib and self.character_pfx_instances and self.character and self.character_animation):
            return
        # Timeline PFX belongs to free lounge animation playback, not to the
        # seated driving replacement pose.
        previewing=not (self.attach_character and self.vehicle) and not self.character_drive_only
        for runtime in self.character_pfx_instances:
            sys=runtime.system; spec=runtime.spec
            local_t=self._character_pfx_local_time(runtime)
            wrapped=runtime.last_local_time >= 0.0 and local_t + 1e-5 < runtime.last_local_time
            if wrapped:
                sys.active=False; sys.was_active=False
                self.pfxlib.reset(sys, clear_particles=True)
                sys.active=False; sys.was_active=False
                runtime.last_local_time=-1.0

            M=self._character_pfx_matrix(runtime,local_t)
            if not previewing:
                sys.active=False
                self.pfxlib.step(sys,dt,world_matrix=M,system_velocity=np.zeros(3,dtype=np.float32))
                runtime.last_local_time=local_t
                continue

            prev_abs=(spec.clip_start-1e-5 if runtime.last_local_time < 0.0
                      else spec.clip_start+runtime.last_local_time)
            curr_abs=spec.clip_start+local_t

            # Authored activity is derived from the timeline every frame rather
            # than from sys.active. This matters when an attachment parent is
            # temporarily hidden: emission pauses while the parent is absent and
            # automatically resumes if the same Start..Stop interval is still
            # active when the parent returns.
            authored_active=bool(spec.state_before(curr_abs))
            for et, action in spec.note_events:
                if not (prev_abs < et <= curr_abs + 1e-6):
                    continue
                a=str(action or '').strip().lower()
                if a=='start':
                    # Actor Start is a retrigger even if a finite emitter from
                    # an earlier Start still reports active. Preserve already
                    # emitted world particles but restart spawn counters/time.
                    if sys.active:
                        sys.active=False; sys.was_active=False
                        self.pfxlib.reset(sys, clear_particles=False)
                    sys.was_active=False
                elif a=='kill':
                    sys.particles.clear()
                    sys.trail_points.clear()

            sys.active=bool(authored_active and M is not None)
            self.pfxlib.step(
                sys,dt,world_matrix=M,
                system_velocity=np.zeros(3,dtype=np.float32)
            )
            runtime.last_local_time=local_t

    def _step_ability_pfx(self, dt: float):
        if not (self.pfxlib and self.ability_pfx_instances):
            return
        self.ability_preview_time+=max(0.0,float(dt))
        t=float(self.ability_preview_time)
        kept=[]
        for runtime in self.ability_pfx_instances:
            cue=runtime.cue
            start=max(0.0,float(cue.start_time)); duration=max(0.05,float(cue.duration))
            end=start+duration
            runtime.elapsed=max(0.0,t-start)
            should_run=bool(start<=t<end)
            if should_run and not runtime.started:
                self.pfxlib.reset(runtime.system,clear_particles=True)
                runtime.system.active=True; runtime.system.was_active=False; runtime.started=True
            else:
                runtime.system.active=should_run
            M=self._ability_runtime_matrix(runtime)
            vel=np.asarray(M[:3,:3]@np.asarray(cue.velocity,dtype=np.float32),dtype=np.float32)
            self.pfxlib.step(runtime.system,dt,world_matrix=M,system_velocity=vel)
            # Keep a short soft-kill tail, but always hard-expire editor
            # preview systems so an unusual long-lived particle cannot leave an
            # ability permanently resident after its authored preview window.
            if t <= end+4.0 and (t <= end+2.0 or runtime.system.particles or runtime.system.trail_points):
                kept.append(runtime)
        self.ability_pfx_instances=kept

    def _draw_ability_pfx(self):
        for runtime in self.ability_pfx_instances:
            self._draw_pfx_system(runtime.system,self._ability_runtime_matrix(runtime))

    def _draw_pfx(self,V):
        for runtime in self.pfx_instances:
            self._draw_pfx_system(runtime.system,V@runtime.base_matrix)
        self._draw_character_pfx()
        self._draw_ability_pfx()

    @staticmethod
    def _ball_position(b:EffectBallRuntime):
        t=max(0.0,float(b.age))
        p=b.spawn.position+b.spawn.velocity*t+np.asarray((0,0,-4.905*t*t),dtype=np.float32)
        if p[2] < b.spawn.radius:
            p=p.copy(); p[2]=b.spawn.radius
        return p

    def _draw_effect_balls(self):
        # Gravity is used only for editor trajectory playback.  Native collision
        # response/bounce remains PhysX behavior and is not fabricated here.
        for b in self.effect_balls:
            if b.age < 0.0 or b.age>b.spawn.life_time:
                continue
            p=self._ball_position(b)
            M=_translation(*p)
            if b.model:
                self._draw_model(b.model,M)
            if b.pfx:
                self._draw_pfx_system(b.pfx,M)

    def fit_scene(self):
        pts=[]
        if self.environment is not None and self.environment.models:
            pts.extend([self.environment.bounds_min, self.environment.bounds_max])
        if self.vehicle:
            for m in self.vehicle.models:
                mn,mx=m.model.bounds()
                for c in (mn,mx): pts.append((m.matrix@np.asarray((*c,1),dtype=np.float32))[:3])
            for w in self.vehicle.wheels:
                mn,mx=w.model.bounds()
                for c in (mn,mx): pts.append((w.matrix@np.asarray((*c,1),dtype=np.float32))[:3])
        if self.character and not (self.attach_character and self.vehicle):
            mn,mx=self.character.model.bounds(); pts.extend([mn,mx])
        if pts:
            a=np.asarray(pts); mn=a.min(0); mx=a.max(0); self.camera_target=(mn+mx)*.5
            radius=float(np.linalg.norm(mx-mn)*.5); self.camera_distance=max(2.5,radius*2.6)
        self.update()

    def _tick(self):
        now=time.monotonic(); dt=min(.1,max(0,now-self._clock)); self._clock=now
        sim_dt=0.0 if self.debug_paused else dt*float(self.debug_time_scale)
        animate_water=bool(self.environment and self.show_island_water and self.water_program
                           and self.environment.water_surfaces and not self.debug_paused)
        if animate_water:
            self._water_preview_time+=sim_dt
            self._water_redraw_accum+=dt
        if self.character_animation:
            self.character_frame += sim_dt*float(self.character_preview_fps or self.character_animation.fps)
            # Selected clips use the asset's exact trailing loopFlag.  For
            # synchronized Cyber body/screen clips, allow the longer companion
            # to finish while holding the shorter body on its last frame.
            paired=(self.character_companion_animation is not None and
                    not (self.attach_character and self.vehicle) and
                    not self.character_drive_only)
            if paired:
                limit=max(int(self.character_animation.frame_count),
                          int(self.character_companion_animation.frame_count),1)-1
                loops=bool(self.force_character_loop or
                           (self.character_animation.looping and self.character_companion_animation.looping))
            else:
                limit=max(0,int(self.character_animation.frame_count)-1)
                loops=bool(self.force_character_loop or self.character_animation.looping)
            if not loops:
                self.character_frame=min(self.character_frame,float(limit))
            cur=int(max(0,min(int(limit),round(self.character_frame))))
            if cur != self._last_emitted_character_frame:
                self._last_emitted_character_frame=cur
                self.animation_frame_changed.emit(cur,int(limit))
            if self.external_root_follow and len(self.character_animation.translations):
                fc=max(1,int(self.character_animation.frame_count))
                ff=(float(self.character_frame)%max(1,fc-1) if self.force_character_loop
                    else min(float(self.character_frame),float(fc-1)))
                fi=max(0,min(fc-1,int(ff)))
                follow_bone=max(0,min(self.character_animation.bone_count-1,self.external_follow_bone))
                origin=self.character_animation.translations[0,follow_bone,:2]
                self.camera_target[:2]=(self.external_follow_anchor[:2] +
                                         self.character_animation.translations[fi,follow_bone,:2]-origin)
        if self.driver_idle_animation and (not self.driver_anim_info or self.driver_anim_info.idle_pose_frame is None):
            self.driver_idle_frame += sim_dt*float(self.driver_idle_animation.fps)
        if (self.character_drive_only and self.character_drive_idle_animation and
                (not self.character_drive_info or self.character_drive_info.idle_pose_frame is None)):
            self.character_drive_frame += sim_dt*float(self.character_drive_idle_animation.fps)

        self._step_character_pfx(sim_dt)
        self._step_ability_pfx(sim_dt)

        # v0.68 map navigation: W/S move along the current camera heading,
        # A/D strafe, Q/E move vertically. Shift accelerates movement.
        map_moved=False
        if self.environment is not None and not self.drive_enabled:
            yaw=math.radians(self.camera_yaw)
            forward=np.asarray((-math.sin(yaw), math.cos(yaw), 0.0),dtype=np.float32)
            right=np.asarray((math.cos(yaw), math.sin(yaw), 0.0),dtype=np.float32)
            move=np.zeros(3,dtype=np.float32)
            if Qt.Key_W in self.keys: move += forward
            if Qt.Key_S in self.keys: move -= forward
            if Qt.Key_D in self.keys: move += right
            if Qt.Key_A in self.keys: move -= right
            if Qt.Key_E in self.keys: move[2] += 1.0
            if Qt.Key_Q in self.keys: move[2] -= 1.0
            mag=float(np.linalg.norm(move))
            if mag>1e-6:
                move/=mag
                speed=max(12.0,min(260.0,float(self.camera_distance)*0.65))
                if Qt.Key_Shift in self.keys: speed*=3.0
                self.camera_target += move*float(dt)*speed
                map_moved=True

        throttle=reverse=steer=0.0; slide=boost=False; state=None
        if self.drive_enabled and self.drive:
            throttle=1.0 if Qt.Key_W in self.keys else 0.0
            reverse=1.0 if Qt.Key_S in self.keys else 0.0
            steer=(1.0 if Qt.Key_A in self.keys else 0.0)+(-1.0 if Qt.Key_D in self.keys else 0.0)
            slide=Qt.Key_Space in self.keys; boost=Qt.Key_Shift in self.keys
            state=self.drive.step(sim_dt,throttle,reverse,steer,slide,boost)
            pt=(float(state.x),float(state.y))
            if not self.drive_path or (pt[0]-self.drive_path[-1][0])**2+(pt[1]-self.drive_path[-1][1])**2>=0.0625:
                self.drive_path.append(pt)
                if len(self.drive_path)>5000:
                    self.drive_path=self.drive_path[-5000:]
            extra=f" • slip {state.slip_deg:+.1f}°" if state.powersliding or abs(state.slip_deg)>0.05 else ""
            self.status_message.emit(
                f"{state.speed_mps*3.6:6.1f} km/h • steer {state.steering_deg:+.1f}° • "
                f"{'POWERSLIDE' if state.powersliding else 'grip'}{extra}"
            )
            self.drive_state_message.emit(
                f"raw {state.steering_input_raw:+.2f} • rack {state.steering_input:+.2f} • "
                f"wheels {state.steering_deg:+.1f}° • driver {state.driver_turn_input:+.2f} • slip {state.slip_deg:+.1f}°"
            )

        # Vehicle PFX uses the authored mount matrix and each pattern's decoded
        # World/Object space semantics.  World-space particles stay where they
        # were emitted instead of being dragged along with the moving chassis.
        if self.pfxlib and self.pfx_instances:
            self._backfire_phase += sim_dt
            V=self._vehicle_runtime_matrix()
            system_velocity=np.zeros(3,dtype=np.float32)
            if state is not None:
                system_velocity=np.asarray((state.world_vx,state.world_vy,0.0),dtype=np.float32)
            for runtime in self.pfx_instances:
                if runtime.kind=="persistent_pfx":
                    # Authored always-on vehicle effects (Pumpkin candles/body
                    # flame, etc.) run continuously at their project mounts.
                    runtime.system.active=True
                elif runtime.kind=="powerslide":
                    runtime.system.active=bool(state is not None and (state.powersliding or state.powerslide_anim_blend>0.03))
                elif runtime.kind in ("backfire","backfire_blue"):
                    # Native backfire is a short burst, not a permanently-on
                    # exhaust torch.  Re-trigger the authored system in visible event-sized
                    # pulses so its 50-80ms EngineFlame and fast alpha chunks
                    # remain visible at the real exhaust mounts. Boost uses the
                    # authored Pfx Blue variant when the vehicle provides one.
                    accelerating=bool(state is not None and throttle>0.0)
                    requested=bool(self.force_backfire_preview or accelerating or boost)
                    # Backfire/EngineFlame patterns commonly have ~0.10 s start delay.
                    # A 0.16 s pulse left too little emission time (14 Hz flame could
                    # produce zero particles), which is why v0.62 looked nearly blank.
                    pulse=(self._backfire_phase % 0.55) < 0.30
                    if runtime.kind=="backfire_blue":
                        runtime.system.active=bool(requested and boost and pulse)
                    else:
                        runtime.system.active=bool(requested and (not boost) and pulse)
                else:
                    runtime.system.active=False
                self.pfxlib.step(
                    runtime.system,sim_dt,world_matrix=V@runtime.base_matrix,
                    system_velocity=system_velocity
                )

        kept=[]
        for b in self.effect_balls:
            previous=b.age; b.age+=sim_dt
            if b.pfx and self.pfxlib and b.age>=0.0:
                b.pfx.active=bool(b.age<=b.spawn.life_time)
                p=self._ball_position(b)
                vel=b.spawn.velocity+np.asarray((0,0,-9.81*max(0.0,b.age)),dtype=np.float32)
                self.pfxlib.step(b.pfx,sim_dt,world_matrix=_translation(*p),system_velocity=vel)
            # Retain a short soft-kill tail after the model's authored lifetime.
            if b.age <= b.spawn.life_time+2.0 or (b.pfx and b.pfx.particles):
                kept.append(b)
        self.effect_balls=kept
        # Static Island Adventure maps do not need a 60-FPS redraw while idle.
        # Mouse/toggle/wheel events already call update(); keyboard navigation
        # repaints only while the camera actually moves. Vehicle/character/PFX
        # preview keeps the original continuous animation path.
        if self.environment is not None:
            if map_moved or (animate_water and self._water_redraw_accum>=1.0/30.0):
                self._water_redraw_accum=0.0
                self.update()
        else:
            continuous=(not self.debug_paused and (
                self.character_animation is not None or
                self.driver_idle_animation is not None or
                self.character_drive_idle_animation is not None or
                (self.drive_enabled and self.drive is not None) or
                bool(self.pfx_instances) or bool(self.character_pfx_instances) or bool(self.effect_balls)
            ))
            if continuous:self.update()

    def keyPressEvent(self,e):
        if e.key()==Qt.Key_R and not e.isAutoRepeat(): self.reset_drive(); return
        self.keys.add(e.key()); super().keyPressEvent(e)
    def keyReleaseEvent(self,e):
        self.keys.discard(e.key()); super().keyReleaseEvent(e)
    def mousePressEvent(self,e):
        self.last_mouse=e.position(); self.mouse_press_pos=e.position()
    def mouseReleaseEvent(self,e):
        try:
            if (self.environment is not None and e.button()==Qt.LeftButton and
                    (e.modifiers() & Qt.ControlModifier) and self.mouse_press_pos is not None):
                p=e.position(); dx=float(p.x()-self.mouse_press_pos.x()); dy=float(p.y()-self.mouse_press_pos.y())
                if dx*dx+dy*dy <= 36.0:
                    self._pick_map_object(p.x(),p.y())
        finally:
            self.last_mouse=None; self.mouse_press_pos=None
    def mouseMoveEvent(self,e):
        if self.last_mouse is None:return
        p=e.position(); dx=p.x()-self.last_mouse.x(); dy=p.y()-self.last_mouse.y(); self.last_mouse=p
        if e.buttons()&Qt.LeftButton:
            self.camera_yaw += dx*.45; self.camera_pitch=max(-10,min(80,self.camera_pitch+dy*.35)); self.update()
        elif self.environment is not None and e.buttons()&Qt.RightButton:
            yaw=math.radians(self.camera_yaw)
            right=np.asarray((math.cos(yaw),math.sin(yaw),0.0),dtype=np.float32)
            up=np.asarray((0.0,0.0,1.0),dtype=np.float32)
            scale=max(0.02,float(self.camera_distance)*0.0025)
            self.camera_target += (-right*float(dx) + up*float(dy))*scale
            self.update()
    def wheelEvent(self,e):
        self.camera_distance*=math.exp(-e.angleDelta().y()/1200.0)
        max_distance=20000.0 if self.environment is not None else 300.0
        self.camera_distance=max(.5,min(max_distance,self.camera_distance)); self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_TITLE); self.resize(1500,900)
        try:
            if APP_ICON_PATH.exists():
                self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        except Exception:
            pass
        self.db=None; self.vresolver=None; self.cresolver=None; self.effects=None; self.pfxlib=None; self.iaresolver=None
        self.current_vehicle=None; self.current_character=None; self.current_island_scene=None
        self.external_motion=None; self.external_retarget_cache={}; self.custom_character=None
        self._active_external_animation=None; self._external_item_title=""
        self.loaded_package_path=""; self.extra_apf_paths=[]
        self._fmod_setup_lib=None; self._current_audio_data=None; self._current_audio_info=None
        self._inspector_restore_width=390
        self._build_ui(); self._build_menu()
        self.statusBar().showMessage("Open an APF or game package to begin.")
        QTimer.singleShot(250, self._maybe_show_tutorial)

    def _build_menu(self):
        m=self.menuBar().addMenu("&File")
        a=QAction("Open APF / APK / XAPK…",self); a.setShortcut("Ctrl+O"); a.triggered.connect(self.open_dialog); m.addAction(a)
        self.add_pack_action=QAction("Add HW / DLC APF pack…",self)
        self.add_pack_action.setEnabled(False); self.add_pack_action.triggered.connect(self.add_apf_dialog)
        m.addAction(self.add_pack_action)
        self.repack_action=QAction("Repack Android package (replace APF)…",self)
        self.repack_action.triggered.connect(self.repack_android_package)
        m.addAction(self.repack_action)
        m.addSeparator(); q=QAction("Exit",self); q.triggered.connect(self.close); m.addAction(q)
        h=self.menuBar().addMenu("&View")
        fit=QAction("Fit Scene",self); fit.setShortcut("F"); fit.triggered.connect(self.viewport.fit_scene); h.addAction(fit)
        clear=QAction("Clear Preview",self); clear.setShortcut("Ctrl+L"); clear.triggered.connect(self.clear_preview); h.addAction(clear)
        h.addSeparator()
        self.inspector_action=QAction("Show Inspector Panel",self); self.inspector_action.setCheckable(True); self.inspector_action.setChecked(True); self.inspector_action.setShortcut("Ctrl+I"); self.inspector_action.toggled.connect(self.set_inspector_visible); h.addAction(self.inspector_action)
        help_menu=self.menuBar().addMenu("&Help")
        tutorial=QAction("Tutorial / Getting Started",self); tutorial.triggered.connect(self.show_tutorial); help_menu.addAction(tutorial)
        about=QAction("About BBR Vector Studio",self); about.triggered.connect(self.show_about); help_menu.addAction(about)


    def _maybe_show_tutorial(self):
        settings=QSettings("Grafix","BBR Vector Studio")
        if not settings.value("tutorial/seen",False,type=bool):
            self.show_tutorial(first_run=True)

    def show_tutorial(self, first_run=False):
        steps=[
            ("Welcome",
             "BBR Vector Studio lets you inspect and preview BBR assets, maps, vehicles, characters, animations and effects."),
            ("1. Open your assets",
             "Use File → Open APF / APK / XAPK, or the Open Game Package / APF button. For direct APF work, open Assets.apf from the exact game build you are using."),
            ("2. Preview vehicles",
             "Open Vehicles, choose a vehicle and skin, then use the viewport to inspect the model, attachments, lights, bounds and optional developer overlays."),
            ("3. Preview characters",
             "Open Characters to select a character, skin and animation. You can preview driving positions, abilities, skeletons and attachment points."),
            ("4. Maps and tracks",
             "Open Map to load a track or map. Use the layer checkboxes, Fit Full Map and Inspector to examine geometry, water, routes, collisions and environment assets."),
            ("5. Extra APF packs",
             "Use Add HW / DLC APF pack when you need an external content pack. HW.apf is not bundled with the Studio."),
            ("6. Useful controls",
             "Use Fit Scene (F), Clear Preview (Ctrl+L) and Show Inspector (Ctrl+I). The viewport also supports wireframe, bounds, mounts and other inspection overlays."),
            ("Need help?",
             "Created by Grafix.<br><br>Discord: <a href='https://discord.com/users/1223806728806731897'>grafix_098</a>")
        ]
        dlg=QDialog(self); dlg.setWindowTitle("BBR Vector Studio Tutorial"); dlg.setMinimumSize(520,300)
        layout=QVBoxLayout(dlg)
        title=QLabel(); title.setStyleSheet("font-size: 19px; font-weight: 600;")
        body=QLabel(); body.setWordWrap(True); body.setTextFormat(Qt.RichText); body.setOpenExternalLinks(True); body.setTextInteractionFlags(Qt.TextBrowserInteraction)
        progress=QLabel()
        dont_show=QCheckBox("Don't show this again")
        dont_show.setChecked(True if first_run else False)
        row=QHBoxLayout()
        back=QPushButton("Back"); nxt=QPushButton("Next"); skip=QPushButton("Skip")
        row.addWidget(back); row.addStretch(1); row.addWidget(skip); row.addWidget(nxt)
        layout.addWidget(title); layout.addWidget(body,1); layout.addWidget(progress); layout.addWidget(dont_show); layout.addLayout(row)
        state={'i':0}
        def render():
            i=state['i']; title.setText(steps[i][0]); body.setText(steps[i][1]); progress.setText(f"Step {i+1} of {len(steps)}")
            back.setEnabled(i>0); nxt.setText("Finish" if i==len(steps)-1 else "Next")
        def finish():
            if dont_show.isChecked():
                QSettings("Grafix","BBR Vector Studio").setValue("tutorial/seen",True)
            dlg.accept()
        def go_back():
            state['i']=max(0,state['i']-1); render()
        def go_next():
            if state['i']>=len(steps)-1: finish()
            else:
                state['i']+=1; render()
        back.clicked.connect(go_back); nxt.clicked.connect(go_next); skip.clicked.connect(finish)
        render(); dlg.exec()

    def show_about(self):
        dlg=QDialog(self); dlg.setWindowTitle("About BBR Vector Studio"); dlg.setMinimumWidth(440)
        layout=QVBoxLayout(dlg)
        label=QLabel(
            f"<h2>BBR Vector Studio v{APP_VERSION}</h2>"
            "<p>Vector Unit asset inspection and preview studio.</p>"
            "<p>Includes BBR2 mobile APK/XAPK/APF and Island Adventure APF map support, "
            "driving diagnostics, source-driven ability/PFX preview, improved vehicle lights, "
            "authored animated water/reflections, Android APF-container repacking, animation tools, "
            "map diagnostics, audio inspection, external mocap mapping diagnostics, and custom Vehicle/Character skin import with Vector .bin decoding and original UV preservation.</p>"
            "<p><b>Created by Grafix</b><br>"
            "Discord: <a href='https://discord.com/users/1223806728806731897'>grafix_098</a></p>"
        )
        label.setWordWrap(True); label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction); label.setOpenExternalLinks(True)
        layout.addWidget(label)
        buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons); dlg.exec()

    @staticmethod
    def _scroll_panel(widget):
        """Wrap long control panels in a vertical scroll area when needed."""
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _sync_debug_toggle(self, key: str, on: bool):
        pairs={
            "wire":(self.vehicle_wire_check,self.char_wire_check,self.viewport.set_debug_wireframe),
            "bounds":(self.vehicle_bounds_check,self.char_bounds_check,self.viewport.set_debug_bounds),
            "skeleton":(self.vehicle_skeleton_check,self.char_skeleton_check,self.viewport.set_debug_skeleton),
            "mounts":(self.vehicle_mounts_check,self.char_mounts_check,self.viewport.set_debug_mounts),
            "normals":(self.vehicle_normals_check,self.char_normals_check,self.viewport.set_debug_normals),
        }
        pair=pairs.get(str(key))
        if not pair:return
        a,b,setter=pair
        for w in (a,b):
            if w.isChecked()!=bool(on):
                w.blockSignals(True); w.setChecked(bool(on)); w.blockSignals(False)
        setter(bool(on))

    def _step_debug_frame(self, delta: int):
        self.dev_pause_check.setChecked(True)
        self.viewport.step_character_frame(delta)

    def show_debug_stats(self):
        self.inspector.setPlainText(self.viewport.debug_scene_report())
        self.statusBar().showMessage("Developer scene statistics sent to Inspector")

    def _update_animation_timeline(self, current: int, limit: int):
        limit=max(0,int(limit)); current=max(0,min(limit,int(current)))
        self.anim_frame_slider.blockSignals(True)
        self.anim_frame_slider.setRange(0,max(0,limit)); self.anim_frame_slider.setValue(current)
        self.anim_frame_slider.blockSignals(False)
        self.anim_frame_slider.setEnabled(limit>0)
        self.anim_frame_label.setText(f"Frame {current} / {limit}")
        self.ext_frame_slider.blockSignals(True)
        self.ext_frame_slider.setRange(0,max(0,limit)); self.ext_frame_slider.setValue(current)
        self.ext_frame_slider.blockSignals(False)
        self.ext_frame_slider.setEnabled(limit>0 and self._external_is_playing())
        self.ext_frame_label.setText(f"Frame {current} / {limit}")

    def _scrub_animation_frame(self, value: int):
        if not self.anim_frame_slider.isEnabled():
            return
        self.dev_pause_check.setChecked(True)
        self.viewport.set_character_frame(int(value))

    def import_custom_character(self):
        path,_=QFileDialog.getOpenFileName(self,"Import custom character","","Binary FBX character (*.fbx);;All files (*.*)")
        if not path:return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            character=load_fbx_character(path)
            self.custom_character=character
            self.custom_show_btn.setEnabled(True)
            self.show_custom_character()
            self.custom_status.setText(
                f"Imported {Path(path).name} • {len(character.model.meshes)} meshes • "
                f"{len(character.model.bones)} bones. Choose a motion in External Animation."
            )
            # Many downloadable FBXs include their own motion. A character-only
            # file is also valid; in that case leave the preview in bind pose.
            try:
                embedded=load_external_motion(path)
            except ValueError as exc:
                if "no changing animation curves" not in str(exc):
                    self.ext_anim_status.setText(f"Character loaded; embedded animation unavailable: {exc}")
            else:
                self.external_motion=embedded
                self.external_retarget_cache.clear()
                self._fill_external_map([b.name for b in character.model.bones])
                self.ext_anim_report.setPlainText(mapping_report(embedded,[b.name for b in character.model.bones]))
                self.ext_anim_status.setText(f"Loaded animation from {Path(path).name} • {embedded.frame_count} frames @ {embedded.fps:.2f} FPS")
                self.ext_play_btn.setEnabled(True); self.ext_map_apply_btn.setEnabled(True)
                self.play_external_animation()
                if self._active_external_animation is not None:
                    self.custom_status.setText(
                        f"Imported {Path(path).name} • {len(character.model.meshes)} meshes • "
                        f"{len(character.model.bones)} bones • playing embedded animation."
                    )
            self.inspector.setPlainText(character.report())
            self.statusBar().showMessage(f"Imported custom character: {Path(path).name}")
        except Exception as exc:
            self.custom_status.setText(f"Character import failed: {exc}")
            self.inspector.setPlainText(traceback.format_exc())
            QMessageBox.critical(self,"Custom character import failed",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def show_custom_character(self):
        character=self.custom_character
        if character is None:return
        self.attach_check.blockSignals(True); self.attach_check.setChecked(False); self.attach_check.blockSignals(False)
        self.viewport.set_attach_character(False)
        if self.current_vehicle is not None:
            self.remove_vehicle_keep_character()
        self.current_island_scene=None; self.current_character=character
        self.char_combo.blockSignals(True); self.char_combo.setCurrentIndex(-1); self.char_combo.blockSignals(False)
        self.skin_combo.blockSignals(True); self.skin_combo.clear(); self.skin_combo.blockSignals(False)
        self.anim_combo.blockSignals(True); self.anim_combo.clear(); self.anim_combo.addItem("(bind pose)"); self.anim_combo.blockSignals(False)
        self._active_external_animation=None
        self.viewport.set_character(character)
        self.drive_pose_btn.setEnabled(False); self.attach_check.setEnabled(False)
        self.char_remove_vehicle_btn.setEnabled(False)
        self.char_ability_combo.clear(); self.char_ability_btn.setEnabled(False)
        self.anim_info_label.setText("Animation: —")
        self.anim_force_loop.setEnabled(False); self.anim_replay_btn.setEnabled(False)
        self.ext_replay_btn.setEnabled(False); self.ext_loop_check.setEnabled(False)
        self._update_animation_timeline(0,0)
        if self.external_motion is not None:
            target=[b.name for b in character.model.bones]
            self._fill_external_map(target)
            self.ext_anim_report.setPlainText(mapping_report(self.external_motion,target))
        ready=self.external_motion is not None and self.external_motion.fbx_clip is not None
        self.ext_play_btn.setEnabled(ready); self.ext_map_apply_btn.setEnabled(ready)
        self.inspector.setPlainText(character.report())
        self.statusBar().showMessage(f"Showing custom character: {character.name}")

    def import_external_animation(self):
        path,_=QFileDialog.getOpenFileName(self,"Import external animation","","Animation files (*.bvh *.fbx);;BVH motion (*.bvh);;FBX animation (*.fbx);;All files (*.*)")
        if not path:return
        try:
            motion=load_external_motion(path)
            self.external_motion=motion
            self.external_retarget_cache.clear()
            target=[b.name for b in self.current_character.model.bones] if self.current_character is not None else None
            self._fill_external_map(target)
            report=mapping_report(motion,target)
            self.ext_anim_report.setPlainText(report)
            self.ext_anim_status.setText(f"Loaded {Path(path).name} • {len(motion.joints)} joints • {motion.frame_count} frames @ {motion.fps:.2f} FPS")
            self.ext_play_btn.setEnabled(motion.fbx_clip is not None and self.current_character is not None)
            self.ext_map_apply_btn.setEnabled(motion.fbx_clip is not None and self.current_character is not None)
            self.inspector.setPlainText(report)
            if motion.fbx_clip is not None and self.current_character is not None:
                self.play_external_animation()
            else:
                self.statusBar().showMessage("Select or import a character and click Play FBX" if motion.fbx_clip else "BVH motion inspected; FBX playback is available")
        except Exception as exc:
            self.external_motion=None
            self.ext_play_btn.setEnabled(False)
            self.ext_map_apply_btn.setEnabled(False)
            self.ext_anim_report.setPlainText(traceback.format_exc())
            QMessageBox.critical(self,"External animation import failed",str(exc))

    def _external_is_playing(self):
        return bool(self.external_motion is not None and self.external_motion.fbx_clip is not None
                    and self.viewport.character_animation is not None
                    and self.viewport.character_animation is self._active_external_animation)

    def _fill_external_map(self, target_names=None):
        if self.external_motion is None:return
        from external_animation import BBR24
        targets=list(target_names or BBR24)
        mapping=auto_map_bones([j.name for j in self.external_motion.joints],targets)
        self.ext_map_edit.setPlainText("\n".join(f"{name} = {mapping.get(name,'-')}" for name in targets))

    def _external_mapping(self, target_names):
        assert self.external_motion is not None
        source_names={j.name for j in self.external_motion.joints}
        targets=set(target_names)
        mapping={}
        for raw in self.ext_map_edit.toPlainText().splitlines():
            line=raw.strip()
            if not line or line.startswith("#"):continue
            if "=" not in line:
                raise ValueError(f"Bone map line needs Target = Source: {line}")
            target,source=[x.strip() for x in line.split("=",1)]
            if target not in targets:
                raise ValueError(f"Unknown target bone: {target}")
            if source and source != "-":
                if source not in source_names:
                    raise ValueError(f"Source bone not found in FBX: {source}")
                mapping[target]=source
        return mapping

    def play_external_animation(self):
        motion=self.external_motion
        if motion is None or motion.fbx_clip is None or self.current_character is None:
            return
        try:
            model=self.current_character.model
            target=[b.name for b in model.bones]
            mapping=self._external_mapping(target)
            key=(motion.source,self.current_character.name,self.current_character.skin,
                 getattr(self.current_character,"source",""),
                 bool(self.ext_root_motion_check.isChecked()),tuple(sorted(mapping.items())))
            anim=self.external_retarget_cache.get(key)
            if anim is None:
                retarget=retarget_custom_fbx if isinstance(self.current_character,CustomCharacter) else retarget_fbx
                anim=retarget(motion.fbx_clip,model,mapping,
                              root_motion=self.ext_root_motion_check.isChecked())
                self.external_retarget_cache[key]=anim
            self._active_external_animation=anim
            self.viewport.set_character_animation(anim,"",preview_fps=motion.fps)
            self.viewport.external_follow_bone=(next((i for i,b in enumerate(model.bones)
                if b.name.split(":")[-1].lower() in ("hips","pelvis")),0)
                if isinstance(self.current_character,CustomCharacter) else 0)
            self.viewport.set_external_root_follow(bool(self.ext_root_motion_check.isChecked()))
            title=f"External FBX: {Path(motion.source).name}"
            self._external_item_title=title
            self.anim_combo.blockSignals(True)
            if self.anim_combo.findText(title)<0:self.anim_combo.addItem(title)
            self.anim_combo.setCurrentText(title)
            self.anim_combo.blockSignals(False)
            self.anim_force_loop.setEnabled(True); self.anim_replay_btn.setEnabled(True)
            self.ext_replay_btn.setEnabled(True); self.ext_loop_check.setEnabled(True)
            self.ext_frame_slider.setEnabled(True)
            self.anim_info_label.setText(f"External FBX • {anim.frame_count} frames • {anim.fps:g} FPS • {anim.duration:.2f}s")
            self.ext_anim_report.setPlainText(mapping_report(motion,target,mapping))
            self.ext_pause_check.setChecked(False)
            self.statusBar().showMessage(f"Playing {Path(motion.source).name} on {self.current_character.name} ({self.current_character.skin})")
            self.viewport.update()
        except Exception as exc:
            self._active_external_animation=None
            self.ext_anim_report.setPlainText(traceback.format_exc())
            QMessageBox.critical(self,"FBX playback failed",str(exc))

    def _scrub_external_frame(self, value: int):
        if not self._external_is_playing():return
        self.ext_pause_check.setChecked(True)
        self.viewport.set_character_frame(value)

    def _toggle_external_root_motion(self, _on: bool):
        if self._external_is_playing():
            old_frame=int(self.viewport.character_frame)
            paused=self.ext_pause_check.isChecked()
            self.play_external_animation()
            self.viewport.set_character_frame(old_frame)
            self.ext_pause_check.setChecked(paused)

    def save_viewport_screenshot(self):
        path,_=QFileDialog.getSaveFileName(self,"Save viewport screenshot","BBR_Vector_Studio.png","PNG image (*.png);;JPEG image (*.jpg *.jpeg)")
        if not path:return
        try:
            image=self.viewport.grabFramebuffer()
            if not image.save(path):
                raise RuntimeError("Qt could not save the image")
            self.statusBar().showMessage(f"Viewport screenshot saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self,"Screenshot failed",str(exc))

    def restore_all_map_objects(self):
        self.ia_isolate_selected_check.blockSignals(True)
        self.ia_isolate_selected_check.setChecked(False)
        self.ia_isolate_selected_check.blockSignals(False)
        self.viewport.show_all_map_objects()

    def export_selected_map_obj(self):
        ms=self.viewport.selected_map_model
        if ms is None:
            QMessageBox.information(self,"Export selected model","Select a map object first using Ctrl + left-click.")
            return
        default=(str(getattr(ms,'name','MapObject')) or 'MapObject').replace('/','_').replace('\\','_') + '.obj'
        path,_=QFileDialog.getSaveFileName(self,"Export selected map model as OBJ",default,"Wavefront OBJ (*.obj)")
        if not path:return
        try:
            M=np.asarray(ms.matrix,dtype=np.float64)
            N=np.linalg.inv(M[:3,:3]).T
            lines=[f"# BBR Vector Studio v{APP_VERSION}",f"# Source: {getattr(ms,'model_ref','')}",f"o {str(getattr(ms,'name','MapObject')).replace(' ','_')}"]
            voff=1; noff=1; toff=1
            for mi,mesh in enumerate(ms.model.meshes):
                name=str(mesh.name or f'mesh_{mi}').replace(' ','_')
                lines.append(f"g {name}")
                pos=np.asarray(mesh.positions,dtype=np.float64)
                for v in pos:
                    w=M@np.asarray((v[0],v[1],v[2],1.0),dtype=np.float64)
                    lines.append(f"v {w[0]:.9g} {w[1]:.9g} {w[2]:.9g}")
                norms=np.asarray(mesh.normals,dtype=np.float64) if getattr(mesh,'normals',None) is not None else np.empty((0,3))
                for n in norms:
                    q=N@n; ln=float(np.linalg.norm(q)); q=q/ln if ln>1e-12 else q
                    lines.append(f"vn {q[0]:.9g} {q[1]:.9g} {q[2]:.9g}")
                uvs=np.asarray(mesh.uvs,dtype=np.float64) if getattr(mesh,'uvs',None) is not None else np.empty((0,2))
                for uv in uvs:
                    lines.append(f"vt {uv[0]:.9g} {uv[1]:.9g}")
                idx=np.asarray(mesh.indices,dtype=np.int64).reshape(-1)
                tri_count=len(idx)//3
                have_uv=len(uvs)==len(pos); have_n=len(norms)==len(pos)
                for t in range(tri_count):
                    face=[]
                    for raw_i in idx[t*3:t*3+3]:
                        i=int(raw_i); vi=voff+i
                        if have_uv and have_n: face.append(f"{vi}/{toff+i}/{noff+i}")
                        elif have_uv: face.append(f"{vi}/{toff+i}")
                        elif have_n: face.append(f"{vi}//{noff+i}")
                        else: face.append(str(vi))
                    lines.append("f "+" ".join(face))
                voff+=len(pos); toff+=len(uvs); noff+=len(norms)
            Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
            self.statusBar().showMessage(f"Selected map model exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self,"OBJ export failed",str(exc))

    def reset_debug_controls(self):
        for cb in (self.vehicle_wire_check,self.vehicle_bounds_check,self.vehicle_skeleton_check,
                   self.vehicle_mounts_check,self.vehicle_normals_check,self.char_wire_check,
                   self.char_bounds_check,self.char_skeleton_check,self.char_mounts_check,
                   self.char_normals_check,self.dev_pause_check,self.dev_axes_check,self.dev_xray_check):
            cb.setChecked(False)
        self.dev_cull_check.setChecked(True)
        i=self.dev_speed_combo.findData(1.0)
        if i>=0:self.dev_speed_combo.setCurrentIndex(i)
        self.viewport.update()
        self.statusBar().showMessage("Developer overlays reset")

    def _build_ui(self):
        root=QSplitter(Qt.Horizontal); self.main_splitter=root; self.setCentralWidget(root)
        left=QWidget(); lv=QVBoxLayout(left); left.setMinimumWidth(260); left.setMaximumWidth(380)
        title=QLabel("BBR Vector Studio"); f=title.font(); f.setPointSize(16); f.setBold(True); title.setFont(f); lv.addWidget(title)
        lv.addWidget(QLabel("v1.01 • APF index + Vehicle/Character custom skin import + Assets.apf export"))
        self.open_btn=QPushButton("Open Game Package / APF"); self.open_btn.clicked.connect(self.open_dialog); lv.addWidget(self.open_btn)
        self.add_pack_btn=QPushButton("Add HW / DLC APF pack…"); self.add_pack_btn.setEnabled(False)
        self.add_pack_btn.clicked.connect(self.add_apf_dialog); lv.addWidget(self.add_pack_btn)
        self.repack_btn=QPushButton("Repack Android Package…")
        self.repack_btn.clicked.connect(self.repack_android_package); lv.addWidget(self.repack_btn)
        self.tabs_left=QTabWidget(); lv.addWidget(self.tabs_left,1)

        veh=QWidget(); vf=QFormLayout(veh)
        self.vehicle_combo=QComboBox(); self.vehicle_combo.setPlaceholderText("Select vehicle…"); self.vehicle_combo.currentTextChanged.connect(self.select_vehicle); vf.addRow("Vehicle",self.vehicle_combo)
        self.vehicle_skin_combo=QComboBox(); self.vehicle_skin_combo.setEnabled(False); self.vehicle_skin_combo.currentTextChanged.connect(self.select_vehicle_skin); vf.addRow("Vehicle skin",self.vehicle_skin_combo)
        self.vehicle_import_skin_btn=QPushButton("Import custom vehicle skin…"); self.vehicle_import_skin_btn.clicked.connect(self.import_custom_vehicle_skin); vf.addRow(self.vehicle_import_skin_btn)
        self.vehicle_export_skin_btn=QPushButton("Export custom skin to Assets.apf…"); self.vehicle_export_skin_btn.clicked.connect(lambda: self.export_custom_skin_apf("vehicle")); vf.addRow(self.vehicle_export_skin_btn)
        self.vehicle_clear_skin_btn=QPushButton("Clear custom vehicle skin"); self.vehicle_clear_skin_btn.clicked.connect(self.clear_custom_vehicle_skin); vf.addRow(self.vehicle_clear_skin_btn)
        self.drive_check=QCheckBox("Drive Preview"); self.drive_check.setEnabled(False)
        vf.addRow(self.drive_check)
        self.reset_btn=QPushButton("Reset drive (R)"); self.reset_btn.setEnabled(False); vf.addRow(self.reset_btn)
        self.remove_vehicle_btn=QPushButton("Remove vehicle (keep character)"); self.remove_vehicle_btn.setEnabled(False); vf.addRow(self.remove_vehicle_btn)
        self.attachments_check=QCheckBox("Show vehicle attachments"); self.attachments_check.setChecked(True); self.attachments_check.setEnabled(False); vf.addRow(self.attachments_check)
        self.attachments_status=QLabel("Attachments: —"); vf.addRow(self.attachments_status)
        self.attachment_combo=QComboBox(); self.attachment_combo.setEnabled(False); self.attachment_combo.addItem("(authored defaults)"); vf.addRow("Attachment preview",self.attachment_combo)
        self.attachment_mount_combo=QComboBox(); self.attachment_mount_combo.setEnabled(False); vf.addRow("Attachment mount",self.attachment_mount_combo)
        self.backfire_check=QCheckBox("Backfire / exhaust-flame preview"); self.backfire_check.setChecked(False); self.backfire_check.setEnabled(False); vf.addRow(self.backfire_check)
        self.corona_check=QCheckBox("Enhanced coronas / vehicle lights"); self.corona_check.setChecked(True); vf.addRow(self.corona_check)
        self.drive_response_combo=QComboBox()
        for label,value in (("Smooth",0.82),("Normal",1.0),("Responsive",1.25),("Fast test",1.50)):
            self.drive_response_combo.addItem(label,value)
        self.drive_response_combo.setCurrentIndex(self.drive_response_combo.findData(1.25)); vf.addRow("Driving response",self.drive_response_combo)
        self.drive_debug_label=QLabel("Drive debug: —"); self.drive_debug_label.setWordWrap(True); vf.addRow(self.drive_debug_label)
        self.vehicle_wire_check=QCheckBox("Wireframe overlay"); vf.addRow(self.vehicle_wire_check)
        self.vehicle_bounds_check=QCheckBox("Bounding boxes"); vf.addRow(self.vehicle_bounds_check)
        self.vehicle_skeleton_check=QCheckBox("Skeleton / bones"); vf.addRow(self.vehicle_skeleton_check)
        self.vehicle_mounts_check=QCheckBox("Mount / attachment points"); vf.addRow(self.vehicle_mounts_check)
        self.vehicle_normals_check=QCheckBox("Vertex normals (debug)"); vf.addRow(self.vehicle_normals_check)
        vf.addRow(QLabel("W/S accelerate/reverse\nA/D steer\nSpace powerslide\nShift boost preview"))
        self.tabs_left.addTab(self._scroll_panel(veh),"Vehicles")

        ch=QWidget(); cf=QFormLayout(ch)
        self.char_combo=QComboBox(); self.char_combo.setPlaceholderText("Select character…"); self.char_combo.currentTextChanged.connect(self.select_character); cf.addRow("Character",self.char_combo)
        self.skin_combo=QComboBox(); self.skin_combo.setPlaceholderText("Select skin…"); self.skin_combo.currentTextChanged.connect(self.select_skin); cf.addRow("Skin",self.skin_combo)
        self.character_import_skin_btn=QPushButton("Import custom character skin…"); self.character_import_skin_btn.clicked.connect(self.import_custom_character_skin); cf.addRow(self.character_import_skin_btn)
        self.character_clear_skin_btn=QPushButton("Clear custom character skin"); self.character_clear_skin_btn.clicked.connect(self.clear_custom_character_skin); cf.addRow(self.character_clear_skin_btn)
        self.character_export_skin_btn=QPushButton("Export custom skin to Assets.apf…"); self.character_export_skin_btn.clicked.connect(lambda: self.export_custom_skin_apf("character")); cf.addRow(self.character_export_skin_btn)
        self.skin_info_label=QLabel("Custom skin: none"); self.skin_info_label.setWordWrap(True); cf.addRow(self.skin_info_label)
        self.anim_combo=QComboBox(); self.anim_combo.setPlaceholderText("Select animation…"); self.anim_combo.currentTextChanged.connect(self.select_animation); cf.addRow("Animation",self.anim_combo)
        self.anim_info_label=QLabel("Animation: —"); self.anim_info_label.setWordWrap(True); cf.addRow("Codec",self.anim_info_label)
        self.anim_force_loop=QCheckBox("Force loop preview"); self.anim_force_loop.setChecked(False); self.anim_force_loop.setEnabled(False); cf.addRow(self.anim_force_loop)
        self.anim_replay_btn=QPushButton("Replay selected animation"); self.anim_replay_btn.setEnabled(False); cf.addRow(self.anim_replay_btn)
        self.anim_frame_slider=QSlider(Qt.Horizontal); self.anim_frame_slider.setRange(0,0); self.anim_frame_slider.setEnabled(False)
        self.anim_frame_label=QLabel("Frame 0 / 0")
        frame_widget=QWidget(); frame_layout=QVBoxLayout(frame_widget); frame_layout.setContentsMargins(0,0,0,0); frame_layout.addWidget(self.anim_frame_slider); frame_layout.addWidget(self.anim_frame_label)
        cf.addRow("Animation timeline",frame_widget)
        # v0.67.2: character preview is intentionally grid-only.
        self.stage_combo=QComboBox(); self.stage_combo.addItems(["Grid"]); self.stage_combo.setCurrentText("Grid"); self.stage_combo.setVisible(False)
        self.stage_fit_btn=QPushButton("Fit character"); cf.addRow(self.stage_fit_btn)
        self.drive_anim_combo=QComboBox(); self.drive_anim_combo.addItems(["Standard","Motorcycle_Dirt","Motorcycle_Sport","Motorcycle_Anime","Motorcycle_Future","Motorcycle_Chopper"]); cf.addRow("Driving set",self.drive_anim_combo)
        self.drive_pose_btn=QPushButton("Preview driving position / idle"); self.drive_pose_btn.setEnabled(False); cf.addRow(self.drive_pose_btn)
        self.attach_check=QCheckBox("Attach to vehicle driver mount"); self.attach_check.setChecked(False); self.attach_check.setEnabled(False); cf.addRow(self.attach_check)
        self.char_remove_vehicle_btn=QPushButton("Character only — remove vehicle"); self.char_remove_vehicle_btn.setEnabled(False); cf.addRow(self.char_remove_vehicle_btn)
        self.char_ability_combo=QComboBox(); self.char_ability_combo.setPlaceholderText("No character ability"); cf.addRow("Character ability",self.char_ability_combo)
        self.char_ability_btn=QPushButton("Preview character ability"); self.char_ability_btn.setEnabled(False); cf.addRow(self.char_ability_btn)
        self.char_wire_check=QCheckBox("Wireframe overlay"); cf.addRow(self.char_wire_check)
        self.char_skeleton_check=QCheckBox("Skeleton / bones"); cf.addRow(self.char_skeleton_check)
        self.char_bounds_check=QCheckBox("Bounding box"); cf.addRow(self.char_bounds_check)
        self.char_mounts_check=QCheckBox("Mount / socket points"); cf.addRow(self.char_mounts_check)
        self.char_normals_check=QCheckBox("Vertex normals (debug)"); cf.addRow(self.char_normals_check)
        self.tabs_left.addTab(self._scroll_panel(ch),"Characters")

        custom=QWidget(); custom_form=QFormLayout(custom)
        self.custom_status=QLabel("Import a rigged binary FBX character with its mesh and skin weights. A game APF is not required.")
        self.custom_status.setWordWrap(True); custom_form.addRow(self.custom_status)
        self.custom_import_btn=QPushButton("Import custom FBX character…"); custom_form.addRow(self.custom_import_btn)
        self.custom_show_btn=QPushButton("Show imported character"); self.custom_show_btn.setEnabled(False); custom_form.addRow(self.custom_show_btn)
        custom_note=QLabel("For motion, use the External Animation tab to import a separate FBX. The editable bone map matches its joints to this character. Animations in the character file play automatically when present.")
        custom_note.setWordWrap(True); custom_form.addRow(custom_note)
        self.tabs_left.addTab(self._scroll_panel(custom),"Custom Character")

        fx=QWidget(); ff=QFormLayout(fx)
        self.effect_combo=QComboBox(); ff.addRow("Ability/effect",self.effect_combo)
        self.effect_btn=QPushButton("Preview effect / ability PFX"); ff.addRow(self.effect_btn)
        self.pfx_combo=QComboBox(); self.pfx_combo.currentTextChanged.connect(self.inspect_pfx); ff.addRow("Inspect PFX",self.pfx_combo)
        self.pfx_preview_btn=QPushButton("Preview selected PFX"); ff.addRow(self.pfx_preview_btn)
        pfx_row=QWidget(); pfx_row_l=QHBoxLayout(pfx_row); pfx_row_l.setContentsMargins(0,0,0,0)
        self.pfx_restart_btn=QPushButton("Restart preview"); self.pfx_stop_btn=QPushButton("Stop preview")
        pfx_row_l.addWidget(self.pfx_restart_btn); pfx_row_l.addWidget(self.pfx_stop_btn); ff.addRow("Playback",pfx_row)
        self.pfx_status=QLabel("Source-driven PFX preview. Start/loop/end systems are read from VehicleEffectDB where available."); self.pfx_status.setWordWrap(True); ff.addRow(self.pfx_status)
        self.tabs_left.addTab(self._scroll_panel(fx),"Effects / PFX")

        audio=QWidget(); av=QVBoxLayout(audio)
        self.audio_list=QListWidget(); self.audio_list.currentTextChanged.connect(self.inspect_audio_asset); av.addWidget(self.audio_list,1)
        self.audio_sample_combo=QComboBox(); self.audio_sample_combo.setPlaceholderText("Select FSB5 sample…"); self.audio_sample_combo.setEnabled(False); av.addWidget(self.audio_sample_combo)
        self.audio_sample_export_btn=QPushButton("Export selected sample…"); self.audio_sample_export_btn.setEnabled(False); self.audio_sample_export_btn.clicked.connect(self.export_audio_sample); av.addWidget(self.audio_sample_export_btn)
        self.audio_export_btn=QPushButton("Export embedded bank / stream…"); self.audio_export_btn.setEnabled(False); self.audio_export_btn.clicked.connect(self.export_audio_asset); av.addWidget(self.audio_export_btn)
        self.tabs_left.addTab(audio,"Audio")

        ia=QWidget(); iaf=QFormLayout(ia)
        self.ia_status=QLabel("Load Island Adventure APF or BBR2 mobile APK/XAPK/APF to enable map preview."); self.ia_status.setWordWrap(True); iaf.addRow(self.ia_status)
        self.ia_mode_combo=QComboBox(); self.ia_mode_combo.addItems(["Track","Career Map"]); self.ia_mode_combo.setEnabled(False); iaf.addRow("Scene type",self.ia_mode_combo)
        self.ia_scene_combo=QComboBox(); self.ia_scene_combo.setPlaceholderText("Select map / track…"); self.ia_scene_combo.setEnabled(False); iaf.addRow("Scene",self.ia_scene_combo)
        self.ia_load_btn=QPushButton("Load Map / Track"); self.ia_load_btn.setEnabled(False); iaf.addRow(self.ia_load_btn)
        self.ia_focus_btn=QPushButton("Focus Starting Grid"); self.ia_focus_btn.setEnabled(False); iaf.addRow(self.ia_focus_btn)
        self.ia_fit_btn=QPushButton("Fit Full Map"); self.ia_fit_btn.setEnabled(False); iaf.addRow(self.ia_fit_btn)
        self.ia_environment_check=QCheckBox("Environment / track geometry"); self.ia_environment_check.setChecked(True); iaf.addRow(self.ia_environment_check)
        self.ia_animated_check=QCheckBox("Animated environment props"); self.ia_animated_check.setChecked(True); iaf.addRow(self.ia_animated_check)
        self.ia_gameplay_check=QCheckBox("Gameplay / alternate-mode models"); self.ia_gameplay_check.setChecked(False); iaf.addRow(self.ia_gameplay_check)
        self.ia_water_check=QCheckBox("Water surface"); self.ia_water_check.setChecked(True); iaf.addRow(self.ia_water_check)
        self.ia_decals_check=QCheckBox("HW track arrow decals"); self.ia_decals_check.setChecked(True); iaf.addRow(self.ia_decals_check)
        self.ia_nodes_check=QCheckBox("AI / authored track route"); self.ia_nodes_check.setChecked(True); self.ia_nodes_check.setEnabled(False); iaf.addRow(self.ia_nodes_check)
        self.ia_spawns_check=QCheckBox("Starting-grid spawn points"); self.ia_spawns_check.setChecked(True); iaf.addRow(self.ia_spawns_check)
        self.ia_collision_check=QCheckBox("Collision walls"); self.ia_collision_check.setChecked(False); iaf.addRow(self.ia_collision_check)
        self.ia_oob_check=QCheckBox("Out-of-bounds volumes"); self.ia_oob_check.setChecked(False); iaf.addRow(self.ia_oob_check)
        self.ia_powerups_check=QCheckBox("Power-up positions"); self.ia_powerups_check.setChecked(False); iaf.addRow(self.ia_powerups_check)
        self.ia_pfx_check=QCheckBox("Static PFX markers"); self.ia_pfx_check.setChecked(False); iaf.addRow(self.ia_pfx_check)
        self.ia_cameras_check=QCheckBox("Camera / cinematic markers"); self.ia_cameras_check.setChecked(False); iaf.addRow(self.ia_cameras_check)
        self.ia_sectors_check=QCheckBox("Track sector markers"); self.ia_sectors_check.setChecked(False); iaf.addRow(self.ia_sectors_check)
        self.ia_bounds_check=QCheckBox("Scene / selected-object bounds"); self.ia_bounds_check.setChecked(False); iaf.addRow(self.ia_bounds_check)
        self.ia_selected_label=QLabel("Selected object: —"); self.ia_selected_label.setWordWrap(True); iaf.addRow(self.ia_selected_label)
        self.ia_clear_selection_btn=QPushButton("Clear selected object"); iaf.addRow(self.ia_clear_selection_btn)
        select_row=QWidget(); select_layout=QHBoxLayout(select_row); select_layout.setContentsMargins(0,0,0,0)
        self.ia_focus_selected_btn=QPushButton("Focus selected"); self.ia_hide_selected_btn=QPushButton("Hide selected")
        select_layout.addWidget(self.ia_focus_selected_btn); select_layout.addWidget(self.ia_hide_selected_btn); iaf.addRow("Selected object",select_row)
        self.ia_isolate_selected_check=QCheckBox("Isolate selected object"); iaf.addRow(self.ia_isolate_selected_check)
        self.ia_show_all_btn=QPushButton("Show all hidden map objects"); iaf.addRow(self.ia_show_all_btn)
        self.ia_export_obj_btn=QPushButton("Export selected model as OBJ…"); iaf.addRow(self.ia_export_obj_btn)
        for w in (self.ia_focus_selected_btn,self.ia_hide_selected_btn,self.ia_isolate_selected_check,self.ia_export_obj_btn): w.setEnabled(False)
        ia_note=QLabel("Map controls: left-drag orbit • Ctrl+left-click select object • right-drag pan • wheel zoom • W/A/S/D move • Q/E height • Shift faster."); ia_note.setWordWrap(True); iaf.addRow(ia_note)
        self.tabs_left.insertTab(2,self._scroll_panel(ia),"Map")

        dev=QWidget(); df=QFormLayout(dev)
        self.dev_pause_check=QCheckBox("Pause simulation / animation"); df.addRow(self.dev_pause_check)
        self.dev_speed_combo=QComboBox()
        for label,value in (("0.25×",0.25),("0.5×",0.5),("1×",1.0),("2×",2.0),("4×",4.0)):
            self.dev_speed_combo.addItem(label,value)
        self.dev_speed_combo.setCurrentIndex(self.dev_speed_combo.findData(1.0)); df.addRow("Animation speed",self.dev_speed_combo)
        step_row=QWidget(); step_layout=QHBoxLayout(step_row); step_layout.setContentsMargins(0,0,0,0)
        self.dev_prev_frame_btn=QPushButton("◀ Previous frame"); self.dev_next_frame_btn=QPushButton("Next frame ▶")
        step_layout.addWidget(self.dev_prev_frame_btn); step_layout.addWidget(self.dev_next_frame_btn); df.addRow("Frame step",step_row)
        self.dev_axes_check=QCheckBox("World XYZ axes"); df.addRow(self.dev_axes_check)
        self.dev_xray_check=QCheckBox("X-ray debug overlays"); df.addRow(self.dev_xray_check)
        self.dev_cull_check=QCheckBox("Backface culling"); self.dev_cull_check.setChecked(True); df.addRow(self.dev_cull_check)
        cull_note=QLabel("Disable backface culling to inspect one-sided or apparently hollow meshes."); cull_note.setWordWrap(True); df.addRow(cull_note)
        cam1=QWidget(); cam1l=QHBoxLayout(cam1); cam1l.setContentsMargins(0,0,0,0)
        self.cam_front_btn=QPushButton("Front"); self.cam_rear_btn=QPushButton("Rear"); self.cam_left_btn=QPushButton("Left")
        for b in (self.cam_front_btn,self.cam_rear_btn,self.cam_left_btn): cam1l.addWidget(b)
        df.addRow("Camera",cam1)
        cam2=QWidget(); cam2l=QHBoxLayout(cam2); cam2l.setContentsMargins(0,0,0,0)
        self.cam_right_btn=QPushButton("Right"); self.cam_top_btn=QPushButton("Top"); self.cam_perspective_btn=QPushButton("Perspective")
        for b in (self.cam_right_btn,self.cam_top_btn,self.cam_perspective_btn): cam2l.addWidget(b)
        df.addRow("",cam2)
        self.dev_screenshot_btn=QPushButton("Save viewport screenshot…"); df.addRow(self.dev_screenshot_btn)
        self.dev_stats_btn=QPushButton("Show scene / render statistics"); df.addRow(self.dev_stats_btn)
        self.dev_reset_btn=QPushButton("Reset developer overlays"); df.addRow(self.dev_reset_btn)
        dev_note=QLabel("Character/vehicle Wireframe, Skeleton, Bounds, Mounts and Normals remain available in their own tabs and stay active while animations run."); dev_note.setWordWrap(True); df.addRow(dev_note)
        self.tabs_left.insertTab(3,self._scroll_panel(dev),"Developer")

        ext=QWidget(); exf=QFormLayout(ext)
        self.ext_anim_status=QLabel("Select a game character or import a custom FBX character, then load a binary FBX animation."); self.ext_anim_status.setWordWrap(True); exf.addRow(self.ext_anim_status)
        self.ext_anim_btn=QPushButton("Import FBX / BVH…"); exf.addRow(self.ext_anim_btn)
        self.ext_play_btn=QPushButton("Play FBX on selected character"); self.ext_play_btn.setEnabled(False); exf.addRow(self.ext_play_btn)
        self.ext_map_edit=QTextEdit(); self.ext_map_edit.setFont(QFont("Consolas",8)); self.ext_map_edit.setMinimumHeight(250)
        self.ext_map_edit.setPlaceholderText("Select a character and import an FBX to see editable Target = Source bone pairs.")
        exf.addRow("Editable bone map",self.ext_map_edit)
        self.ext_map_apply_btn=QPushButton("Apply bone map and replay"); self.ext_map_apply_btn.setEnabled(False); exf.addRow(self.ext_map_apply_btn)
        self.ext_root_motion_check=QCheckBox("Use FBX root movement"); exf.addRow(self.ext_root_motion_check)
        self.ext_loop_check=QCheckBox("Loop animation"); self.ext_loop_check.setEnabled(False); exf.addRow(self.ext_loop_check)
        self.ext_pause_check=QCheckBox("Pause animation"); exf.addRow(self.ext_pause_check)
        self.ext_replay_btn=QPushButton("Replay FBX"); self.ext_replay_btn.setEnabled(False); exf.addRow(self.ext_replay_btn)
        self.ext_frame_slider=QSlider(Qt.Horizontal); self.ext_frame_slider.setRange(0,0); self.ext_frame_slider.setEnabled(False)
        self.ext_frame_label=QLabel("Frame 0 / 0")
        ext_frames=QWidget(); ext_frames_layout=QVBoxLayout(ext_frames); ext_frames_layout.setContentsMargins(0,0,0,0)
        ext_frames_layout.addWidget(self.ext_frame_slider); ext_frames_layout.addWidget(self.ext_frame_label)
        exf.addRow("Scrub frames",ext_frames)
        self.ext_anim_report=QTextEdit(); self.ext_anim_report.setReadOnly(True); self.ext_anim_report.setFont(QFont("Consolas",8)); self.ext_anim_report.setMinimumHeight(220); exf.addRow("Retarget map",self.ext_anim_report)
        ext_note=QLabel("FBX preview uses the selected character's own mesh and skin. Wireframe and skeleton overlays remain available in the Characters and Developer tabs. BVH files can still be inspected here."); ext_note.setWordWrap(True); exf.addRow(ext_note)
        self.tabs_left.addTab(self._scroll_panel(ext),"External Animation")

        raw=QWidget(); rv=QVBoxLayout(raw); self.search=QLineEdit(); self.search.setPlaceholderText("Filter asset paths…"); rv.addWidget(self.search)
        self.asset_list=QListWidget(); rv.addWidget(self.asset_list,1); self.search.textChanged.connect(self.filter_assets); self.asset_list.currentTextChanged.connect(self.inspect_asset)
        self.tabs_left.addTab(raw,"Raw Assets")

        diag=QWidget(); dv=QVBoxLayout(diag)
        self.diag_status=QLabel("Quick package diagnostics: asset-type coverage, APF codecs and unresolved map references."); self.diag_status.setWordWrap(True); dv.addWidget(self.diag_status)
        self.diag_scan_btn=QPushButton("Run Asset Support Scanner"); dv.addWidget(self.diag_scan_btn)
        self.diag_report=QTextEdit(); self.diag_report.setReadOnly(True); self.diag_report.setFont(QFont("Consolas",8)); dv.addWidget(self.diag_report,1)
        self.tabs_left.addTab(diag,"Diagnostics")

        self.viewport=GLViewport(); self.viewport.status_message.connect(self.statusBar().showMessage)
        self.viewport.selection_message.connect(self.on_map_selection_message)
        self.viewport.animation_frame_changed.connect(self._update_animation_timeline)
        self.viewport.drive_state_message.connect(self.drive_debug_label.setText)
        # Connections needing viewport constructed.
        self.drive_check.toggled.connect(self.viewport.toggle_drive)
        self.reset_btn.clicked.connect(self.viewport.reset_drive)
        self.attachments_check.toggled.connect(self.viewport.set_show_vehicle_attachments)
        self.attachment_combo.currentTextChanged.connect(self.select_custom_attachment)
        self.attachment_mount_combo.currentIndexChanged.connect(self.apply_custom_attachment_preview)
        self.backfire_check.toggled.connect(self.viewport.set_backfire_preview)
        self.corona_check.toggled.connect(self.viewport.set_enhanced_coronas)
        self.drive_response_combo.currentIndexChanged.connect(lambda _i:self.viewport.set_drive_response_scale(self.drive_response_combo.currentData() or 1.0))
        self.viewport.set_drive_response_scale(self.drive_response_combo.currentData() or 1.0)
        self.vehicle_wire_check.toggled.connect(lambda on:self._sync_debug_toggle("wire",on))
        self.vehicle_bounds_check.toggled.connect(lambda on:self._sync_debug_toggle("bounds",on))
        self.vehicle_skeleton_check.toggled.connect(lambda on:self._sync_debug_toggle("skeleton",on))
        self.vehicle_mounts_check.toggled.connect(lambda on:self._sync_debug_toggle("mounts",on))
        self.vehicle_normals_check.toggled.connect(lambda on:self._sync_debug_toggle("normals",on))
        self.char_wire_check.toggled.connect(lambda on:self._sync_debug_toggle("wire",on))
        self.char_skeleton_check.toggled.connect(lambda on:self._sync_debug_toggle("skeleton",on))
        self.char_bounds_check.toggled.connect(lambda on:self._sync_debug_toggle("bounds",on))
        self.char_mounts_check.toggled.connect(lambda on:self._sync_debug_toggle("mounts",on))
        self.char_normals_check.toggled.connect(lambda on:self._sync_debug_toggle("normals",on))
        self.dev_pause_check.toggled.connect(self.viewport.set_debug_paused)
        self.dev_pause_check.toggled.connect(self.ext_pause_check.setChecked)
        self.ext_pause_check.toggled.connect(self.dev_pause_check.setChecked)
        self.dev_speed_combo.currentIndexChanged.connect(lambda _i:self.viewport.set_debug_time_scale(self.dev_speed_combo.currentData() or 1.0))
        self.dev_prev_frame_btn.clicked.connect(lambda:self._step_debug_frame(-1))
        self.dev_next_frame_btn.clicked.connect(lambda:self._step_debug_frame(1))
        self.dev_axes_check.toggled.connect(self.viewport.set_debug_world_axes)
        self.dev_xray_check.toggled.connect(self.viewport.set_debug_xray)
        self.dev_cull_check.toggled.connect(self.viewport.set_debug_backface_culling)
        self.cam_front_btn.clicked.connect(lambda:self.viewport.set_camera_preset("front"))
        self.cam_rear_btn.clicked.connect(lambda:self.viewport.set_camera_preset("rear"))
        self.cam_left_btn.clicked.connect(lambda:self.viewport.set_camera_preset("left"))
        self.cam_right_btn.clicked.connect(lambda:self.viewport.set_camera_preset("right"))
        self.cam_top_btn.clicked.connect(lambda:self.viewport.set_camera_preset("top"))
        self.cam_perspective_btn.clicked.connect(lambda:self.viewport.set_camera_preset("perspective"))
        self.dev_screenshot_btn.clicked.connect(self.save_viewport_screenshot)
        self.dev_stats_btn.clicked.connect(self.show_debug_stats)
        self.dev_reset_btn.clicked.connect(self.reset_debug_controls)
        self.remove_vehicle_btn.clicked.connect(self.remove_vehicle_keep_character)
        self.attach_check.toggled.connect(self.viewport.set_attach_character)
        self.anim_force_loop.toggled.connect(self.viewport.set_force_character_loop)
        self.anim_force_loop.toggled.connect(self.ext_loop_check.setChecked)
        self.ext_loop_check.toggled.connect(self.anim_force_loop.setChecked)
        self.anim_replay_btn.clicked.connect(self.viewport.restart_character_animation)
        self.anim_frame_slider.valueChanged.connect(self._scrub_animation_frame)
        self.stage_combo.currentTextChanged.connect(self.viewport.set_preview_stage)
        self.stage_fit_btn.clicked.connect(self.viewport.fit_scene)
        self.viewport.set_preview_stage(self.stage_combo.currentText())
        self.drive_pose_btn.clicked.connect(self.preview_character_driving_pose)
        self.char_remove_vehicle_btn.clicked.connect(self.remove_vehicle_keep_character)
        self.char_ability_btn.clicked.connect(self.preview_character_ability)
        self.effect_btn.clicked.connect(self.preview_effect)
        self.pfx_preview_btn.clicked.connect(self.preview_selected_pfx)
        self.pfx_restart_btn.clicked.connect(self.viewport.restart_ability_preview)
        self.pfx_stop_btn.clicked.connect(self.viewport.stop_ability_preview)
        self.ext_anim_btn.clicked.connect(self.import_external_animation)
        self.custom_import_btn.clicked.connect(self.import_custom_character)
        self.custom_show_btn.clicked.connect(self.show_custom_character)
        self.ext_play_btn.clicked.connect(self.play_external_animation)
        self.ext_map_apply_btn.clicked.connect(self.play_external_animation)
        self.ext_replay_btn.clicked.connect(self.viewport.restart_character_animation)
        self.ext_root_motion_check.toggled.connect(self._toggle_external_root_motion)
        self.ext_frame_slider.valueChanged.connect(self._scrub_external_frame)
        self.ia_mode_combo.currentTextChanged.connect(self.populate_island_scenes)
        self.ia_load_btn.clicked.connect(self.load_island_scene)
        self.ia_focus_btn.clicked.connect(self.viewport.focus_environment_start)
        self.ia_fit_btn.clicked.connect(self.viewport.fit_environment_full)
        self.ia_nodes_check.toggled.connect(self.viewport.set_show_island_track_nodes)
        self.ia_environment_check.toggled.connect(lambda on:self.viewport.set_island_layer("environment",on))
        self.ia_animated_check.toggled.connect(lambda on:self.viewport.set_island_layer("animated",on))
        self.ia_gameplay_check.toggled.connect(lambda on:self.viewport.set_island_layer("gameplay",on))
        self.ia_water_check.toggled.connect(lambda on:self.viewport.set_island_layer("water",on))
        self.ia_decals_check.toggled.connect(lambda on:self.viewport.set_island_layer("decals",on))
        self.ia_spawns_check.toggled.connect(lambda on:self.viewport.set_island_layer("spawns",on))
        self.ia_collision_check.toggled.connect(lambda on:self.viewport.set_island_layer("collision",on))
        self.ia_oob_check.toggled.connect(lambda on:self.viewport.set_island_layer("oob",on))
        self.ia_powerups_check.toggled.connect(lambda on:self.viewport.set_island_layer("powerups",on))
        self.ia_pfx_check.toggled.connect(lambda on:self.viewport.set_island_layer("pfx",on))
        self.ia_cameras_check.toggled.connect(lambda on:self.viewport.set_island_layer("cameras",on))
        self.ia_sectors_check.toggled.connect(lambda on:self.viewport.set_island_layer("sectors",on))
        self.ia_bounds_check.toggled.connect(lambda on:self.viewport.set_island_layer("bounds",on))
        self.ia_clear_selection_btn.clicked.connect(self.viewport.clear_map_selection)
        self.ia_focus_selected_btn.clicked.connect(self.viewport.focus_selected_map_object)
        self.ia_hide_selected_btn.clicked.connect(self.viewport.hide_selected_map_object)
        self.ia_isolate_selected_check.toggled.connect(self.viewport.set_isolate_selected_map)
        self.ia_show_all_btn.clicked.connect(self.restore_all_map_objects)
        self.ia_export_obj_btn.clicked.connect(self.export_selected_map_obj)
        self.diag_scan_btn.clicked.connect(self.run_asset_support_scanner)

        right=QTabWidget(); self.inspector_panel=right; right.setMinimumWidth(340)
        self.inspector=QTextEdit(); self.inspector.setReadOnly(True); self.inspector.setFont(QFont("Consolas",9)); right.addTab(self.inspector,"Inspector")
        self.raw_json=QTextEdit(); self.raw_json.setReadOnly(True); self.raw_json.setFont(QFont("Consolas",8)); right.addTab(self.raw_json,"Raw JSON")
        self.sources=QTextEdit(); self.sources.setReadOnly(True); self.sources.setFont(QFont("Consolas",9)); right.addTab(self.sources,"Sources")
        root.addWidget(left); root.addWidget(self.viewport); root.addWidget(right); root.setSizes([300,850,390])

    def on_map_selection_message(self, text: str):
        msg=str(text or "")
        first=msg.splitlines()[0] if msg else "No map object selected"
        self.ia_selected_label.setText(first.replace("Selected map object:","Selected object:"))
        selected=bool(msg and not msg.startswith("No map object"))
        for w in (self.ia_focus_selected_btn,self.ia_hide_selected_btn,self.ia_export_obj_btn):
            w.setEnabled(selected)
        if not selected and self.ia_isolate_selected_check.isChecked():
            self.ia_isolate_selected_check.blockSignals(True); self.ia_isolate_selected_check.setChecked(False); self.ia_isolate_selected_check.blockSignals(False)
        self.ia_isolate_selected_check.setEnabled(selected)
        if selected:
            self.inspector.setPlainText(msg)

    def run_asset_support_scanner(self):
        if not self.db:
            self.diag_report.setPlainText("Load an APF/APK/XAPK first.")
            return
        from collections import Counter
        type_counts=Counter()
        for rec in self.db.all_records():
            typ=rec.path.split("/",1)[0] if "/" in rec.path else "(root)"
            type_counts[typ]+=1
        dedicated={
            "VuAnimatedModelAsset","VuStaticModelAsset","VuMaterialAsset","VuTextureAsset","VuCubeTextureAsset",
            "VuAnimationAsset","VuProjectAsset","VuTemplateAsset","VuPfxAsset","VuAudioBankAsset",
            "VuDBAsset","VuSpreadsheetAsset","VuDrivingAnimationSetAsset"
        }
        json_or_raw={"VuShaderAsset","VuStringAsset","VuFontAsset","VuLocalizationAsset"}
        codec_counts=Counter(); codec_unknown=[]
        for name,arc in self.db.archives:
            for e in arc.entries:
                c=arc.codec(e); codec_counts[c]+=1
                if c=="unknown" and len(codec_unknown)<30:codec_unknown.append(f"{name}: {e.path} flags=0x{int(e.flags):08X}")
        lines=[
            "BBR Vector Studio Asset Support Scanner",
            f"Unique paths: {len(self.db.records_by_path)}",
            f"APFs: {len(self.db.archives)}",
            "",
            "Compression codecs:",
        ]
        for k,v in sorted(codec_counts.items()):lines.append(f"  {k}: {v}")
        lines += ["", "Asset types:"]
        for typ,count in sorted(type_counts.items(),key=lambda kv:(-kv[1],kv[0].lower())):
            state="typed" if typ in dedicated else ("raw/JSON" if typ in json_or_raw else "no dedicated decoder")
            lines.append(f"  {typ}: {count}  [{state}]")
        unknown=[(t,c) for t,c in type_counts.items() if t not in dedicated and t not in json_or_raw]
        lines += ["",f"Types without a dedicated Studio decoder: {len(unknown)}"]
        for typ,count in sorted(unknown,key=lambda kv:(-kv[1],kv[0].lower()))[:80]:
            lines.append(f"  {typ}: {count}")
        if codec_unknown:
            lines += ["","Unknown compression entries:"]+ ["  "+x for x in codec_unknown]
        if self.current_island_scene is not None:
            unresolved=[]
            for x in self.current_island_scene.unresolved_models:
                if x not in unresolved:unresolved.append(x)
            lines += ["",f"Current map unresolved model references: {len(unresolved)}"]
            lines += ["  "+x for x in unresolved[:80]]
        self.diag_report.setPlainText("\n".join(lines))
        self.diag_status.setText(f"Scan complete • {len(type_counts)} asset types • {len(unknown)} without dedicated decoder")

    def set_inspector_visible(self,on):
        on=bool(on)
        if not on and self.inspector_panel.isVisible():
            sizes=self.main_splitter.sizes()
            if len(sizes)>=3 and sizes[2]>0:
                self._inspector_restore_width=max(340,int(sizes[2]))
        self.inspector_panel.setVisible(on)
        if on:
            sizes=self.main_splitter.sizes()
            if len(sizes)>=3:
                total=max(1,sum(sizes))
                right=min(max(340,self._inspector_restore_width),max(340,total//2))
                left=max(260,sizes[0] if sizes[0]>0 else 300)
                center=max(320,total-left-right)
                self.main_splitter.setSizes([left,center,right])

    def clear_preview(self):
        self.current_vehicle=None; self.current_character=None; self.current_island_scene=None
        self.viewport.clear_scene()
        if self.viewport.materials:
            self.viewport.materials.set_vehicle("")
        for combo in (self.vehicle_combo,self.char_combo):
            combo.blockSignals(True); combo.setCurrentIndex(-1); combo.blockSignals(False)
        for combo in (self.skin_combo,self.anim_combo):
            combo.blockSignals(True); combo.clear(); combo.blockSignals(False)
        self.drive_check.blockSignals(True); self.drive_check.setChecked(False); self.drive_check.blockSignals(False)
        self.drive_check.setEnabled(False); self.reset_btn.setEnabled(False); self.remove_vehicle_btn.setEnabled(False)
        self.attachments_check.setEnabled(False); self.attachments_status.setText("Attachments: —")
        self.attachment_combo.blockSignals(True); self.attachment_combo.clear(); self.attachment_combo.addItem("(authored defaults)");
        if self.vresolver: self.attachment_combo.addItems(self.vresolver.attachment_names())
        self.attachment_combo.setCurrentIndex(0); self.attachment_combo.blockSignals(False); self.attachment_combo.setEnabled(False)
        self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear(); self.attachment_mount_combo.blockSignals(False); self.attachment_mount_combo.setEnabled(False)
        self.backfire_check.blockSignals(True); self.backfire_check.setChecked(False); self.backfire_check.blockSignals(False); self.backfire_check.setEnabled(False); self.viewport.set_backfire_preview(False)
        self.attach_check.blockSignals(True); self.attach_check.setChecked(False); self.attach_check.blockSignals(False); self.attach_check.setEnabled(False)
        self.char_remove_vehicle_btn.setEnabled(False); self.drive_pose_btn.setEnabled(False)
        self.char_ability_combo.clear(); self.char_ability_btn.setEnabled(False)
        if hasattr(self,"ia_focus_btn"):
            self.ia_focus_btn.setEnabled(False); self.ia_fit_btn.setEnabled(False)
        self.inspector.clear(); self.raw_json.clear()
        if hasattr(self,"drive_debug_label"): self.drive_debug_label.setText("Drive debug: —")
        if hasattr(self,"pfx_status"): self.pfx_status.setText("Source-driven PFX preview. Start/loop/end systems are read from VehicleEffectDB where available.")
        if self.db:
            self.sources.setPlainText(self.db.report())
            self.statusBar().showMessage("Preview cleared. Select a vehicle, character, Island Adventure scene, effect or raw asset.")

    def populate_island_scenes(self, *_args):
        resolver=self.iaresolver
        available=bool(resolver and resolver.available)
        self.ia_mode_combo.setEnabled(available)
        self.ia_scene_combo.blockSignals(True); self.ia_scene_combo.clear()
        if not available:
            self.ia_scene_combo.blockSignals(False)
            self.ia_scene_combo.setEnabled(False); self.ia_load_btn.setEnabled(False); self.ia_nodes_check.setEnabled(False)
            self.ia_focus_btn.setEnabled(False); self.ia_fit_btn.setEnabled(False)
            for w in (self.ia_environment_check,self.ia_animated_check,self.ia_gameplay_check,self.ia_water_check,self.ia_decals_check,
                      self.ia_spawns_check,self.ia_collision_check,self.ia_oob_check,self.ia_powerups_check,
                      self.ia_pfx_check,self.ia_cameras_check): w.setEnabled(False)
            profile=getattr(self.db,"game_variant","Unknown package") if self.db else "No package loaded"
            self.ia_status.setText(f"{profile}\nNo compatible VuProjectAsset/Tracks map projects were detected.")
            return
        mode=self.ia_mode_combo.currentText() or "Track"
        if mode == "Career Map":
            if resolver.has_career_map(): self.ia_scene_combo.addItem("Career Map")
        else:
            self.ia_scene_combo.addItems(resolver.tracks())
        self.ia_scene_combo.setCurrentIndex(0 if self.ia_scene_combo.count() else -1)
        self.ia_scene_combo.blockSignals(False)
        self.ia_scene_combo.setEnabled(self.ia_scene_combo.count()>0)
        self.ia_load_btn.setEnabled(self.ia_scene_combo.count()>0)
        self.ia_nodes_check.setEnabled(mode=="Track")
        for w in (self.ia_environment_check,self.ia_animated_check,self.ia_gameplay_check,self.ia_water_check,self.ia_decals_check,
                  self.ia_spawns_check,self.ia_collision_check,self.ia_oob_check,self.ia_powerups_check,
                  self.ia_pfx_check,self.ia_cameras_check):
            w.setEnabled(available)
        self.ia_status.setText(
            f"{getattr(self.db,'game_variant','BBR package')} • {len(resolver.tracks())} map/track projects" +
            (" • Career Map available" if resolver.has_career_map() else "")
        )

    def load_island_scene(self):
        if not (self.iaresolver and self.iaresolver.available):
            return
        mode=self.ia_mode_combo.currentText() or "Track"
        name=self.ia_scene_combo.currentText()
        if not name:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.clear_preview()
            spec=(self.iaresolver.resolve_career_map() if mode=="Career Map" else self.iaresolver.resolve_track(name))
            self.current_island_scene=spec
            self.viewport.set_show_island_track_nodes(self.ia_nodes_check.isChecked())
            self.viewport.set_island_layer("environment",self.ia_environment_check.isChecked())
            self.viewport.set_island_layer("animated",self.ia_animated_check.isChecked())
            self.viewport.set_island_layer("gameplay",self.ia_gameplay_check.isChecked())
            self.viewport.set_island_layer("water",self.ia_water_check.isChecked())
            self.viewport.set_island_layer("spawns",self.ia_spawns_check.isChecked())
            self.viewport.set_island_layer("collision",self.ia_collision_check.isChecked())
            self.viewport.set_island_layer("oob",self.ia_oob_check.isChecked())
            self.viewport.set_island_layer("powerups",self.ia_powerups_check.isChecked())
            self.viewport.set_island_layer("pfx",self.ia_pfx_check.isChecked())
            self.viewport.set_island_layer("cameras",self.ia_cameras_check.isChecked())
            self.viewport.set_environment(spec)
            self.ia_focus_btn.setEnabled(True); self.ia_fit_btn.setEnabled(True)
            report=spec.report()
            self.inspector.setPlainText(report)
            self.raw_json.setPlainText(
                f"{spec.project_path}\n\nComplete project graph expanded for v0.68 map preview. "
                "Use the Map layer toggles for gameplay, collision, OOB, route, water and markers. "
                "Use Raw Assets to inspect individual authored assets."
            )
            self.sources.setPlainText(self.db.report()+"\n\n"+report)
            self.statusBar().showMessage(
                f"Loaded {self.db.game_variant} {spec.kind}: {spec.name} • {len(spec.models)} model instances • {len(spec.spawn_points)} starts • {len(spec.collision_segments)} collision polylines"
            )
        except Exception as exc:
            self.inspector.setPlainText(traceback.format_exc())
            QMessageBox.critical(self,"Map preview failed",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def open_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,"Open BBR package or APF","","BBR files (*.xapk *.apks *.apk *.zip *.apf);;All files (*.*)")
        if path:self.load_database(path)

    def add_apf_dialog(self):
        if not self.db:return
        path,_=QFileDialog.getOpenFileName(
            self,"Add Hot Wheels or DLC APF","","Vector APF packs (*.apf);;All files (*.*)")
        if path:self.add_apf_pack(path)

    def add_apf_pack(self,path):
        if not self.db:return
        candidate=Path(path)
        if candidate.suffix.lower()!=".apf":
            QMessageBox.warning(self,"APF pack required","Choose an .apf asset pack.")
            return
        if any(arc.path.resolve()==candidate.resolve() for _,arc in self.db.archives):
            self.statusBar().showMessage(f"{candidate.name} is already loaded")
            return
        self.load_database(self.loaded_package_path, [*self.extra_apf_paths,str(candidate)])

    def repack_android_package(self):
        """Replace an existing APF inside APK/XAPK/APKS without touching game logic.

        Custom vehicle/character skins have their own Assets.apf exporter; this
        command remains the generic Android-container repacker for a completed APF.
        """
        source=str(self.loaded_package_path or "")
        if not source or Path(source).suffix.lower() not in (".apk",".xapk",".apks",".zip"):
            source,_=QFileDialog.getOpenFileName(
                self,"Choose original Android package","",
                "Android packages (*.apk *.xapk *.apks *.zip);;All files (*.*)")
            if not source:return
        try:
            targets=discover_apf_targets(source)
        except Exception as exc:
            QMessageBox.critical(self,"Package inspection failed",str(exc));return
        if not targets:
            QMessageBox.warning(self,"No APF found","No embedded .apf archive was found in this package.");return

        labels=[x.label for x in targets]
        chosen,ok=QInputDialog.getItem(
            self,"Select embedded APF",
            "Choose the APF entry to replace. Assets.apf is the normal game-content pack:",
            labels,0,False)
        if not ok or not chosen:return
        target=next(x for x in targets if x.label==chosen)

        replacement,_=QFileDialog.getOpenFileName(
            self,"Choose replacement APF","","Vector APF (*.apf);;All files (*.*)")
        if not replacement:return
        src=Path(source)
        default=str(src.with_name(src.stem+"_repacked"+src.suffix))
        output,_=QFileDialog.getSaveFileName(
            self,"Save rebuilt Android package",default,
            "Android package (*%s);;All files (*.*)" % src.suffix)
        if not output:return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            repack_with_apf(source,target,replacement,output)
        except Exception as exc:
            QMessageBox.critical(self,"Repack failed",f"{exc}\n\n{traceback.format_exc()}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        note=(
            f"Rebuilt package saved to:\n{output}\n\n"
            f"Replaced: {target.label}\n\n"
            + signing_guidance(source) +
            "\n\nCustom vehicle/character skin exports can now be created directly from the skin tabs as an Assets.apf."
        )
        QMessageBox.information(self,"Android package rebuilt",note)
        self.statusBar().showMessage(f"Repacked {Path(source).name} -> {Path(output).name}")

    def load_database(self,path,extra_apfs=None):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        new=None
        try:
            new=AssetDatabase.open(path)
            for pack in extra_apfs or []:
                candidate=Path(pack)
                if any(arc.path.resolve()==candidate.resolve() for _,arc in new.archives):
                    continue
                new._add_archive(candidate.name,candidate)
            new._detect_game_variant()
            if self.db:self.db.close()
            self.loaded_package_path=str(path); self.extra_apf_paths=list(extra_apfs or [])
            self._fmod_setup_lib=None; self._current_audio_data=None; self._current_audio_info=None
            self.db=new; self.vresolver=VehicleResolver50(new); self.cresolver=CharacterResolver50(new); self.effects=EffectLibrary50(new); self.pfxlib=PfxLibrary50(new); self.iaresolver=IslandAdventureResolver(new)
            self.add_pack_btn.setEnabled(True); self.add_pack_action.setEnabled(True)
            self.external_retarget_cache.clear(); self._active_external_animation=None
            self.viewport.set_database(new)
            self.current_vehicle=None; self.current_character=None; self.current_island_scene=None
            self.vehicle_combo.blockSignals(True); self.vehicle_combo.clear(); self.vehicle_combo.addItems(self.vresolver.vehicles()); self.vehicle_combo.setCurrentIndex(-1); self.vehicle_combo.blockSignals(False)
            self.vehicle_skin_combo.blockSignals(True); self.vehicle_skin_combo.clear(); self.vehicle_skin_combo.blockSignals(False); self.vehicle_skin_combo.setEnabled(False)
            self.char_combo.blockSignals(True); self.char_combo.clear(); self.char_combo.addItems(self.cresolver.characters()); self.char_combo.setCurrentIndex(-1); self.char_combo.blockSignals(False)
            self.skin_combo.clear(); self.anim_combo.clear()
            self.effect_combo.clear(); self.effect_combo.addItems([n for n in self.effects.names() if (self.effects.get(n) or {}).get("Type")])
            self.pfx_combo.clear(); self.pfx_combo.addItems(self.pfxlib.names())
            self.audio_list.blockSignals(True); self.audio_list.clear(); self.audio_list.addItems([p for p in sorted(new.records_by_path) if p.startswith(("VuAudioBankAsset/","VuAudioStreamAsset/"))]); self.audio_list.blockSignals(False); self.audio_export_btn.setEnabled(False); self.audio_sample_combo.clear(); self.audio_sample_combo.setEnabled(False); self.audio_sample_export_btn.setEnabled(False)
            self.drive_check.blockSignals(True); self.drive_check.setChecked(False); self.drive_check.blockSignals(False); self.drive_check.setEnabled(False); self.reset_btn.setEnabled(False); self.remove_vehicle_btn.setEnabled(False)
            self.attachments_check.setChecked(True); self.attachments_check.setEnabled(False); self.attachments_status.setText("Attachments: —")
            self.attachment_combo.blockSignals(True); self.attachment_combo.clear(); self.attachment_combo.addItem("(authored defaults)"); self.attachment_combo.addItems(self.vresolver.attachment_names()); self.attachment_combo.setCurrentIndex(0); self.attachment_combo.blockSignals(False); self.attachment_combo.setEnabled(False)
            self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear(); self.attachment_mount_combo.blockSignals(False); self.attachment_mount_combo.setEnabled(False)
            self.backfire_check.blockSignals(True); self.backfire_check.setChecked(False); self.backfire_check.blockSignals(False); self.backfire_check.setEnabled(False); self.viewport.set_backfire_preview(False)
            self.attach_check.blockSignals(True); self.attach_check.setChecked(False); self.attach_check.blockSignals(False); self.attach_check.setEnabled(False); self.viewport.set_attach_character(False)
            self.char_remove_vehicle_btn.setEnabled(False); self.drive_pose_btn.setEnabled(False); self.char_ability_combo.clear(); self.char_ability_btn.setEnabled(False)
            self.populate_island_scenes()
            self.filter_assets(self.search.text())
            profile=getattr(new,"game_variant","Vector Unit package")
            self.inspector.setPlainText(
                f"Package loaded: {profile}. Nothing is auto-selected.\n\n"
                "Choose a Vehicle, Character, Map/Track, Effect/PFX, Audio or Raw Asset from the left panel."
            )
            self.raw_json.clear(); self.sources.setPlainText(new.report())
            self.statusBar().showMessage(f"Loaded {len(new.archives)} APF pack(s) • {len(new.records_by_path)} unique assets • select what you want to preview")
        except Exception as exc:
            if new is not None and new is not self.db:new.close()
            QMessageBox.critical(self,"Open failed",f"{exc}\n\n{traceback.format_exc()}")
        finally: QApplication.restoreOverrideCursor()

    def _install_vehicle_spec(self, spec, preview_attachment="", preview_mount=""):
        self.current_island_scene=None
        self.current_vehicle=spec; self.viewport.set_vehicle(spec)
        self.drive_check.blockSignals(True); self.drive_check.setChecked(False); self.drive_check.blockSignals(False)
        self.drive_check.setEnabled(True); self.reset_btn.setEnabled(True); self.remove_vehicle_btn.setEnabled(True)
        self.attachments_check.setEnabled(True); self.backfire_check.setEnabled(True); self.attachment_combo.setEnabled(True)
        default_count=sum(1 for m in spec.models if m.role in ("Attachment","LooseMount"))
        preview_count=sum(1 for m in spec.models if m.role=="AttachmentPreview")
        if preview_attachment:
            self.attachments_status.setText(f"Attachments: {default_count} default + {preview_count} preview • {preview_attachment}")
        else:
            self.attachments_status.setText(f"Attachments: {default_count} authored/default")
        self.attach_check.setEnabled(self.current_character is not None); self.char_remove_vehicle_btn.setEnabled(self.current_character is not None)
        if self.current_character is not None and self.attach_check.isChecked():
            self.viewport.set_attach_character(True)
        extra = ""
        if preview_attachment:
            extra=f"\n\nCustom attachment preview: {preview_attachment}\nMount: {preview_mount}"
        self.inspector.setPlainText(spec.source_report()+extra); self.raw_json.setPlainText(_safe_json(spec.expanded_root))
        self.sources.setPlainText(self.db.report()+"\n\n"+spec.source_report()+extra)

    def import_custom_vehicle_skin(self):
        path,_=QFileDialog.getOpenFileName(self,"Import custom vehicle skin / texture","","Skin/image or Vector BIN (*.png *.jpg *.jpeg *.webp *.tga *.bmp *.dds *.bin);;All files (*)")
        if not path:
            path=QFileDialog.getExistingDirectory(self,"Select vehicle skin package folder")
        if not path or not self.current_vehicle:
            return
        try:
            textures,meta=load_skin_package(path)
            self.custom_skin_export_data={"kind":"vehicle","textures":textures,"meta":meta,"source":path}
            self.viewport.set_custom_skin_textures(textures, path, "vehicle")
            self.statusBar().showMessage(f"Custom vehicle skin loaded: {meta['count']} texture(s)")
            self.inspector.setPlainText(self.current_vehicle.source_report()+"\n\nCustom skin: "+path+f"\nDecoded textures: {meta['count']}\nIgnored model/non-texture files: {len(meta['ignored'])}")
        except Exception as exc:
            QMessageBox.critical(self,"Vehicle skin import failed",str(exc))

    def clear_custom_vehicle_skin(self):
        self.custom_skin_export_data=None
        self.viewport.clear_custom_skin()
        self.statusBar().showMessage("Custom vehicle skin cleared")

    def select_vehicle(self,name):
        if not name or not self.vresolver:return
        try:
            skins=self.vresolver.vehicle_skins(name)
            self.vehicle_skin_combo.blockSignals(True); self.vehicle_skin_combo.clear(); self.vehicle_skin_combo.addItems(skins); self.vehicle_skin_combo.setCurrentIndex(0); self.vehicle_skin_combo.blockSignals(False); self.vehicle_skin_combo.setEnabled(bool(skins))
            self.select_vehicle_skin(self.vehicle_skin_combo.currentText() or "Default")
        except Exception:
            self.inspector.setPlainText(traceback.format_exc())

    def select_vehicle_skin(self,skin_name):
        if not self.vresolver:
            return
        name=self.vehicle_combo.currentText()
        if not name:
            return
        try:
            # A skin switch rebuilds the authored project variant and resets
            # only the custom attachment preview, not the selected character.
            self.attachment_combo.blockSignals(True); self.attachment_combo.setCurrentIndex(0); self.attachment_combo.blockSignals(False)
            self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear(); self.attachment_mount_combo.blockSignals(False); self.attachment_mount_combo.setEnabled(False)
            self.viewport.clear_custom_skin()
            spec=self.vresolver.resolve(name,skin_name or "Default")
            self._install_vehicle_spec(spec)
        except Exception:
            self.inspector.setPlainText(traceback.format_exc())

    def select_custom_attachment(self, name):
        if not self.vresolver or not self.current_vehicle:
            return
        vehicle_name=self.vehicle_combo.currentText() or self.current_vehicle.name
        if not name or name.startswith("(authored"):
            try:
                spec=self.vresolver.resolve(vehicle_name,self.vehicle_skin_combo.currentText() or "Default")
                self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear(); self.attachment_mount_combo.blockSignals(False); self.attachment_mount_combo.setEnabled(False)
                self._install_vehicle_spec(spec)
            except Exception:
                self.inspector.setPlainText(traceback.format_exc())
            return
        try:
            base=self.vresolver.resolve(vehicle_name,self.vehicle_skin_combo.currentText() or "Default")
            mounts=self.vresolver.compatible_attachment_mounts(base,name)
            self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear()
            for path,label in mounts:
                self.attachment_mount_combo.addItem(label,path)
            self.attachment_mount_combo.blockSignals(False)
            self.attachment_mount_combo.setEnabled(bool(mounts))
            if not mounts:
                self._install_vehicle_spec(base)
                self.attachments_status.setText(f"Attachments: {name} has no compatible socket on {vehicle_name}")
                self.statusBar().showMessage(f"{name}: no compatible authored attachment mount on {vehicle_name}")
                return
            self.attachment_mount_combo.setCurrentIndex(0)
            self.apply_custom_attachment_preview()
        except Exception:
            self.inspector.setPlainText(traceback.format_exc())

    def apply_custom_attachment_preview(self, *_args):
        if not self.vresolver or not self.current_vehicle:
            return
        name=self.attachment_combo.currentText()
        if not name or name.startswith("(authored"):
            return
        mount_path=self.attachment_mount_combo.currentData()
        if not mount_path:
            return
        vehicle_name=self.vehicle_combo.currentText() or self.current_vehicle.name
        try:
            spec=self.vresolver.resolve(vehicle_name,self.vehicle_skin_combo.currentText() or "Default")
            label=self.vresolver.append_custom_attachment(spec,name,str(mount_path))
            self._install_vehicle_spec(spec,name,label)
            self.statusBar().showMessage(f"Attachment preview: {name} • {label}")
        except Exception:
            self.inspector.setPlainText(traceback.format_exc())

    def import_custom_character_skin(self):
        path,_=QFileDialog.getOpenFileName(self,"Import custom character skin / texture","","Skin/image or Vector BIN (*.png *.jpg *.jpeg *.webp *.tga *.bmp *.dds *.bin);;All files (*)")
        if not path:
            path=QFileDialog.getExistingDirectory(self,"Select character skin package folder")
        if not path or not self.current_character:
            return
        try:
            textures,meta=load_skin_package(path)
            self.custom_skin_export_data={"kind":"character","textures":textures,"meta":meta,"source":path}
            model=None
            models=meta.get("models",{})
            desired=str(self.current_character.skin or "").lower()
            if desired in models: model=models[desired]
            elif str(self.current_character.name or "").lower() in models: model=models[str(self.current_character.name or "").lower()]
            elif len(models)==1: model=next(iter(models.values()))
            self.viewport.set_custom_skin_textures(textures, path, "character", model=model)
            extra_model=f" • model BIN: {model.name}" if model is not None else ""
            self.skin_info_label.setText(f"Custom skin: {Path(path).name} • {meta['count']} texture(s){extra_model}")
            self.statusBar().showMessage(f"Custom character skin loaded: {meta['count']} texture(s), {meta.get('model_count',0)} model skin(s) • UV/skeleton preserved")
            self.inspector.setPlainText(self.current_character.report()+"\n\nCustom skin source: "+path+f"\nDecoded texture assets: {meta['count']}\nDecoded animated-model skin BINs: {meta.get('model_count',0)}\nIgnored files: {len(meta['ignored'])}")
        except Exception as exc:
            QMessageBox.critical(self,"Character skin import failed",str(exc))

    def export_custom_skin_apf(self, kind):
        """Export by replacing one existing, shipped skin texture only.

        The original APF remains untouched. The selected skin's existing
        VuTextureAsset is used as the binary template and its authored
        VuAnimatedModelAsset is never rewritten.
        """
        data=self.custom_skin_export_data
        if not data or data.get("kind") != kind:
            QMessageBox.warning(self,"No custom skin","Import a custom skin first.")
            return

        source,_=QFileDialog.getOpenFileName(
            self,"Choose original Assets.apf","",
            "Vector APF (*.apf);;All files (*)")
        if not source:
            return

        textures=data.get("textures",{}) or {}
        if not textures:
            QMessageBox.warning(self,"No texture","The imported skin contains no decoded texture.")
            return

        from vector_formats import APFArchive
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            arc=APFArchive(source)
            try:
                texture_entries=[e for e in arc.entries if e.path.lower().startswith("vutextureasset/")]
                if not texture_entries:
                    raise ValueError("The selected Assets.apf contains no VuTextureAsset entries.")

                # Build the reverse material -> texture map once. A texture is
                # not considered skin-owned merely because its filename looks
                # like a skin. This prevents ability/effect textures from being
                # accidentally overwritten.
                usage_index=build_texture_usage_index(source)

                context=[]
                if kind=="vehicle":
                    context=[str(self.vehicle_combo.currentText() or ""),
                             str(self.vehicle_skin_combo.currentText() or ""),
                             str(getattr(self.current_vehicle,"name","") or "")]
                    selected_model=getattr(self.current_vehicle,"model",None)
                else:
                    context=[str(self.char_combo.currentText() or ""),
                             str(self.skin_combo.currentText() or ""),
                             str(getattr(self.current_character,"name","") or ""),
                             str(getattr(self.current_character,"skin","") or "")]
                    selected_model=getattr(self.current_character,"model",None)

                tokens=[]
                for value in context:
                    value=str(value or "").lower().replace(" ","_").replace("\\","/")
                    if value and len(value)>1:
                        tokens.extend([value,Path(value).stem.lower()])
                tokens=list(dict.fromkeys(tokens))

                selected_skin=str(getattr(self.current_character,"skin","") or self.skin_combo.currentText() or "").lower().replace(" ","_")
                selected_char=str(getattr(self.current_character,"name","") or self.char_combo.currentText() or "").lower().replace(" ","_")

                # First resolve the selected model through its authored
                # VuMaterialAsset references. This is much safer than choosing
                # a texture solely from its filename.
                model_texture_paths=set(skin_texture_paths(source,selected_model,usage_index))
                model_entries=[e for e in texture_entries if e.path in model_texture_paths]

                def score(entry):
                    leaf=Path(entry.path.replace("\\","/")).stem.lower()
                    points=0
                    if entry.path in model_texture_paths: points += 500
                    if selected_skin and leaf==selected_skin: points += 100
                    if selected_skin and selected_skin in leaf: points += 40
                    if selected_char and leaf==selected_char: points += 20
                    if selected_char and selected_char in leaf: points += 5
                    points += sum(3 if tok==leaf else 1 for tok in tokens if tok in leaf)
                    return points

                # Remove textures that are also referenced by effect/ability
                # materials. They must never be edited in-place by the skin
                # exporter because that changes the ability appearance too.
                safe_model_entries=[]
                for e in model_entries:
                    diag=texture_usage_diagnostics(source,e.path,usage_index)
                    if not diag["ability_shared"]:
                        safe_model_entries.append(e)

                preferred=safe_model_entries or [
                    e for e in texture_entries
                    if not texture_usage_diagnostics(source,e.path,usage_index)["ability_shared"]
                    and score(e)>0
                ]

                if not preferred:
                    raise ValueError(
                        "No skin-owned texture was found that is safe to replace. "
                        "The matching texture(s) are shared with a character ability/effect. "
                        "The exporter stopped instead of changing the ability texture."
                    )

                ranked=sorted(preferred,key=lambda e:(-score(e),len(e.path),e.path.lower()))
                candidates=ranked[:1500]
                labels=[]
                for e in candidates:
                    diag=texture_usage_diagnostics(source,e.path,usage_index)
                    suffix=" • skin material"
                    if len(diag["material_refs"])>1:
                        suffix+=f" • shared by {len(diag['material_refs'])} materials"
                    labels.append(e.path+suffix)

                target_label,ok=QInputDialog.getItem(
                    self,
                    f"Select existing {kind} skin texture to replace",
                    "Skin-owned VuTextureAsset (ability-shared textures are hidden):",
                    labels,0,True)
                if not ok or not target_label:
                    return
                target=target_label.split(" • ",1)[0]

                custom_keys=list(textures.keys())
                custom_labels=[str(k) for k in custom_keys]
                custom_key,ok=QInputDialog.getItem(
                    self,
                    "Select custom texture",
                    "Imported texture:",
                    custom_labels,0,True)
                if not ok or not custom_key:
                    return
                custom_texture=textures[custom_key]
            finally:
                arc.close()

            default=str(Path(source).with_name(
                Path(source).stem + f"_{kind}_CustomSkin.apf"))
            output,_=QFileDialog.getSaveFileName(
                self,"Export modified Assets.apf",default,
                "Vector APF (*.apf);;All files (*)")
            if not output:
                return

            if Path(source).resolve()==Path(output).resolve():
                output=str(Path(source).with_name(
                    Path(source).stem + f"_{kind}_CustomSkin.apf"))
                n=2
                while Path(output).resolve()==Path(source).resolve() or Path(output).exists():
                    output=str(Path(source).with_name(
                        f"{Path(source).stem}_{kind}_CustomSkin_{n}.apf"))
                    n+=1

            replacements,detail=build_existing_skin_replacement(
                source,target,custom_texture)
            stats=write_apf_from_archive(source,output,replacements)

            # Runtime-oriented verification: the selected entry must decode,
            # while every other APF entry remains structurally readable.
            check=APFArchive(output)
            try:
                entry=check.get(target)
                if entry is None:
                    raise ValueError(f"Export verification failed: missing {target}")
                decoded=check.decode(entry)
                if len(decoded)!=int(entry.unpacked_size):
                    raise ValueError(f"Export verification failed: size mismatch for {target}")
            finally:
                check.close()

            fmt=detail.get("format")
            codec=detail.get("codec")
            msg=(
                f"Existing {kind} skin replaced successfully:\n{output}\n\n"
                f"Target: {target}\n"
                f"Format: {fmt} • APF codec: {codec}\n"
                f"Replaced entries: {stats['replaced']}\n\n"
                "Only the selected existing VuTextureAsset was changed. "
                "The original VuAnimatedModelAsset and all unrelated assets were left untouched.\n\n"
                "The original Assets.apf was not modified."
            )
            QMessageBox.information(self,"Custom skin APF export",msg)
            self.statusBar().showMessage(
                f"Custom {kind} skin exported by replacing existing asset: {Path(output).name}")
        except Exception as exc:
            QMessageBox.critical(
                self,"Custom skin APF export failed",
                f"{exc}\n\n{traceback.format_exc()}")
        finally:
            QApplication.restoreOverrideCursor()

    def clear_custom_character_skin(self):
        self.custom_skin_export_data=None
        self.viewport.clear_custom_skin()
        self.skin_info_label.setText("Custom skin: none")
        self.statusBar().showMessage("Custom character skin cleared")

    def select_character(self,name):
        if not name or not self.cresolver:return
        self.skin_combo.blockSignals(True); self.skin_combo.clear(); self.skin_combo.addItems(self.cresolver.skins(name));
        if name in self.cresolver.skins(name): self.skin_combo.setCurrentText(name)
        self.skin_combo.blockSignals(False); self.select_skin(self.skin_combo.currentText())

    def select_skin(self,skin):
        name=self.char_combo.currentText()
        if not name or not skin or not self.cresolver:return
        try:
            self.viewport.clear_custom_skin()
            self.skin_info_label.setText("Custom skin: none")
            spec=self.cresolver.resolve(name,skin); self.current_island_scene=None; self.current_character=spec; self.viewport.set_character(spec)
            self._active_external_animation=None
            self.attach_check.setEnabled(self.current_vehicle is not None)
            self.char_remove_vehicle_btn.setEnabled(self.current_vehicle is not None)
            self.drive_pose_btn.setEnabled(True)
            self.anim_combo.blockSignals(True); self.anim_combo.clear(); self.anim_combo.addItem("(bind pose)"); self.anim_combo.addItems(spec.animations); self.anim_combo.blockSignals(False)
            self.anim_info_label.setText("Animation: —")
            self.anim_force_loop.setEnabled(False); self.anim_replay_btn.setEnabled(False)
            self.ext_play_btn.setEnabled(self.external_motion is not None and self.external_motion.fbx_clip is not None)
            self.ext_map_apply_btn.setEnabled(self.ext_play_btn.isEnabled())
            self.ext_frame_slider.setEnabled(False); self.ext_replay_btn.setEnabled(False)
            if self.external_motion is not None:
                self._fill_external_map([b.name for b in spec.model.bones])
                self.ext_anim_report.setPlainText(mapping_report(self.external_motion,[b.name for b in spec.model.bones]))
            self._populate_character_abilities(name,skin)
            self.inspector.setPlainText(spec.report()); self.raw_json.setPlainText("")
        except Exception:self.inspector.setPlainText(traceback.format_exc())

    def select_animation(self,ref):
        if ref and ref==self._external_item_title and self.external_motion is not None:
            self.play_external_animation()
            return
        if isinstance(self.current_character,CustomCharacter):
            self._active_external_animation=None
            self.ext_frame_slider.setEnabled(False); self.ext_replay_btn.setEnabled(False)
            self.ext_pause_check.setChecked(False)
            self.viewport.set_external_root_follow(False)
            self.viewport.set_character_animation(None)
            self.anim_info_label.setText("Animation: —")
            self.anim_force_loop.setEnabled(False); self.anim_replay_btn.setEnabled(False)
            self.inspector.setPlainText(self.current_character.report())
            return
        if not self.cresolver:return
        self._active_external_animation=None
        self.ext_frame_slider.setEnabled(False); self.ext_replay_btn.setEnabled(False)
        self.ext_pause_check.setChecked(False)
        self.viewport.set_external_root_follow(False)
        anim=None if not ref or ref.startswith("(") else self.cresolver.animation(ref)
        preview_fps=(self.cresolver.authored_preview_fps(
            self.current_character.name,self.current_character.skin,ref,float(anim.fps or 30.0)
        ) if (anim is not None and self.current_character is not None) else 30.0)
        self.viewport.set_character_animation(anim, ref, preview_fps=preview_fps)
        active=anim is not None
        if not active:self._update_animation_timeline(0,0)
        self.anim_force_loop.setEnabled(active)
        self.anim_replay_btn.setEnabled(active)
        if active:
            rate_note=(f" • preview {preview_fps:g} FPS" if abs(float(preview_fps)-float(anim.fps))>1e-6 else "")
            self.anim_info_label.setText(anim.codec_report()+rate_note)
            base=self.current_character.report() if self.current_character is not None else ""
            self.inspector.setPlainText(
                base + "\n\nAnimation:\n" + str(ref) + "\n" + anim.codec_report() +
                (f"\nAuthored preview rate: {preview_fps:g} FPS (timeline-derived)" if rate_note else "") +
                "\nQuaternion: int16 x/y/z/w ÷ 32767 (native deserialize)\n" +
                "Playback: " + ("forced authored-cycle loop" if self.anim_force_loop.isChecked() else ("asset loop" if anim.looping else "asset one-shot / hold last frame"))
            )
        else:
            self.anim_info_label.setText("Animation: —")
            if self.current_character is not None:
                self.inspector.setPlainText(self.current_character.report())

    def _populate_character_abilities(self, character_name: str, skin_name: str):
        self.char_ability_combo.blockSignals(True); self.char_ability_combo.clear()
        choices=[]
        if self.effects:
            names=set(self.effects.names())
            # The VehicleEffectDB uses the playable skin name when a skin has
            # its own ability tuning, and the base character name otherwise.
            # Victim records are reactions, not abilities the driver can fire.
            for candidate in (str(skin_name or ""), str(character_name or "")):
                if candidate and candidate in names and "victim" not in candidate.lower():
                    e=self.effects.get(candidate)
                    if e and e.get("Type") and candidate not in choices:
                        choices.append(candidate)
        self.char_ability_combo.addItems(choices)
        self.char_ability_combo.blockSignals(False)
        self.char_ability_btn.setEnabled(bool(choices))

    def preview_character_driving_pose(self):
        if self.current_character is None:
            return
        # With a vehicle present, "driving position" means the real vehicle
        # driver mount + that vehicle's authored DrivingAnimationSet.  Without
        # a vehicle it becomes a character-only pose/animation preview.
        if self.current_vehicle is not None:
            set_name=str(self.current_vehicle.driving_anim_set or "Standard")
            i=self.drive_anim_combo.findText(set_name)
            if i>=0:
                self.drive_anim_combo.setCurrentIndex(i)
            self.attach_check.setChecked(True)
            self.viewport.set_attach_character(True)
            self.viewport.fit_scene()
            self.statusBar().showMessage(f"Driving position: {set_name} • attached to {self.current_vehicle.name}")
            return
        set_name=self.drive_anim_combo.currentText() or "Standard"
        if self.viewport.set_character_driving_preview(set_name,True):
            self.anim_combo.blockSignals(True); self.anim_combo.setCurrentIndex(-1); self.anim_combo.blockSignals(False)
            self.statusBar().showMessage(f"Character-only driving pose: {set_name}")
        else:
            self.statusBar().showMessage(f"No compatible {set_name} driving animation for this character")

    def remove_vehicle_keep_character(self):
        if self.current_vehicle is None:
            return
        was_attached=bool(self.attach_check.isChecked() and self.current_character is not None)
        old_set=str(self.current_vehicle.driving_anim_set or "Standard")
        self.current_vehicle=None
        self.viewport.clear_vehicle_keep_character()
        self.vehicle_combo.blockSignals(True); self.vehicle_combo.setCurrentIndex(-1); self.vehicle_combo.blockSignals(False)
        self.vehicle_skin_combo.blockSignals(True); self.vehicle_skin_combo.clear(); self.vehicle_skin_combo.blockSignals(False); self.vehicle_skin_combo.setEnabled(False)
        self.drive_check.blockSignals(True); self.drive_check.setChecked(False); self.drive_check.blockSignals(False)
        self.drive_check.setEnabled(False); self.reset_btn.setEnabled(False); self.remove_vehicle_btn.setEnabled(False)
        self.attachments_check.setEnabled(False); self.attachments_status.setText("Attachments: —")
        self.attachment_combo.setEnabled(False); self.attachment_combo.blockSignals(True); self.attachment_combo.setCurrentIndex(0); self.attachment_combo.blockSignals(False)
        self.attachment_mount_combo.setEnabled(False); self.attachment_mount_combo.blockSignals(True); self.attachment_mount_combo.clear(); self.attachment_mount_combo.blockSignals(False)
        self.backfire_check.blockSignals(True); self.backfire_check.setChecked(False); self.backfire_check.blockSignals(False); self.backfire_check.setEnabled(False); self.viewport.set_backfire_preview(False)
        self.attach_check.blockSignals(True); self.attach_check.setChecked(False); self.attach_check.blockSignals(False); self.attach_check.setEnabled(False)
        self.char_remove_vehicle_btn.setEnabled(False)
        if was_attached:
            i=self.drive_anim_combo.findText(old_set)
            if i>=0: self.drive_anim_combo.setCurrentIndex(i)
            self.viewport.set_character_driving_preview(old_set,True)
        if self.current_character is not None:
            self.inspector.setPlainText(self.current_character.report()+"\n\nVehicle removed: character-only preview active.")
        self.statusBar().showMessage("Vehicle removed. Character, skin and selected pose/animation were kept.")

    def preview_effect_name(self, name: str):
        if not self.effects or not name:return
        e=self.effects.get(name)
        if not e:return
        etype=str(e.get("Type") or "")
        self.viewport.stop_ability_preview()
        built=0; detail=""
        if etype=="VuVehicleDropBallsEffect":
            spawns=self.effects.drop_ball_spawns(name)
            self.viewport.trigger_effect(spawns)
            built=len(spawns)
            detail=f"{built} source-authored balls"
        else:
            cues=self.effects.pfx_cues(name)
            built=self.viewport.trigger_ability_pfx(cues)
            if built:
                detail=f"{built} source-authored PFX cue(s)"
            else:
                detail="no directly previewable PFX reference was found in this effect row"
        self.pfx_status.setText(f"{name} • {etype} • {detail}")
        self.statusBar().showMessage(f"Previewing {name} ({etype}) • {detail}")
        cue_report=[]
        if etype!="VuVehicleDropBallsEffect":
            for c in self.effects.pfx_cues(name):
                cue_report.append(f"{c.role}: {c.ref or '[mount '+c.mount+']'} bone={c.bone or '-'} start={c.start_time:g}s duration={c.duration:g}s source={c.source_key}")
        extra=("\n\nPFX preview cues:\n"+"\n".join(cue_report)) if cue_report else ""
        self.inspector.setPlainText(f"Effect: {name}\nType: {etype}\nSource: VuDBAsset/VehicleEffectDB\nPreview: {detail}\n\n"+_safe_json(e)+extra)
        self.raw_json.setPlainText(_safe_json(e))

    def preview_character_ability(self):
        self.preview_effect_name(self.char_ability_combo.currentText())

    def preview_effect(self):
        self.preview_effect_name(self.effect_combo.currentText())

    def preview_selected_pfx(self):
        name=self.pfx_combo.currentText()
        if not name:return
        cue=AbilityPfxCue(str(name),"loop",np.zeros(3,dtype=np.float32),np.zeros(3,dtype=np.float32),duration=4.0,source_key="Direct PFX preview")
        self.viewport.stop_ability_preview()
        built=self.viewport.trigger_ability_pfx([cue])
        self.pfx_status.setText(f"Direct PFX preview: {name}" if built else f"Could not build PFX: {name}")
        if built:self.statusBar().showMessage(f"Previewing PFX: {name}")

    def inspect_pfx(self,name):
        if not name or not self.pfxlib:return
        s=self.pfxlib.build(name)
        if s:
            self.inspector.setPlainText(s.report()); self.raw_json.setPlainText(_safe_json(self.db.pfx(name)))

    def inspect_audio_asset(self,path):
        self.audio_sample_combo.blockSignals(True); self.audio_sample_combo.clear(); self.audio_sample_combo.blockSignals(False)
        self.audio_sample_combo.setEnabled(False); self.audio_sample_export_btn.setEnabled(False)
        self._current_audio_data=None; self._current_audio_info=None
        if not path or not self.db:
            self.audio_export_btn.setEnabled(False); return
        rec=self.db.get(path)
        if not rec:
            self.audio_export_btn.setEnabled(False); return
        try:
            dec=self.db.decode(rec); info=inspect_audio_payload(dec.data)
            self._current_audio_data=dec.data; self._current_audio_info=info
            if info.bank is not None and info.bank.samples:
                self.audio_sample_combo.blockSignals(True)
                for sm in info.bank.samples:
                    self.audio_sample_combo.addItem(f"{sm.index:03d} • {sm.name} • {sm.duration:.2f}s",sm.index)
                self.audio_sample_combo.setCurrentIndex(0); self.audio_sample_combo.blockSignals(False)
                self.audio_sample_combo.setEnabled(True); self.audio_sample_export_btn.setEnabled(True)
            elif info.kind=="MP3":
                self.audio_sample_combo.addItem("MP3 stream",0); self.audio_sample_combo.setEnabled(True); self.audio_sample_export_btn.setEnabled(True)
            self.inspector.setPlainText(f"Asset: {rec.path}\nSource: {rec.source_name}\nStored: {rec.stored_size} bytes\nUnpacked: {rec.unpacked_size} bytes\n\n"+info.report())
            self.raw_json.setPlainText("(binary audio asset; metadata decoded by Vector Studio)")
            self.audio_export_btn.setEnabled(True)
        except Exception:
            self.audio_export_btn.setEnabled(False); self.inspector.setPlainText(traceback.format_exc())

    def export_audio_sample(self):
        data=self._current_audio_data; info=self._current_audio_info
        if data is None or info is None: return
        try:
            if info.kind=="MP3":
                blob=bytes(data[info.payload_offset:info.payload_offset+info.payload_size]); ext=".mp3"; base="stream"
            elif info.kind=="FSB5" and info.bank is not None:
                idx=int(self.audio_sample_combo.currentData() if self.audio_sample_combo.currentData() is not None else 0)
                if not (0 <= idx < len(info.bank.samples)): return
                sm=info.bank.samples[idx]; base=sm.name or f"sample_{idx:03d}"
                if info.bank.mode_name=="VORBIS":
                    if self._fmod_setup_lib is None and self.loaded_package_path:
                        try: self._fmod_setup_lib=FmodVorbisSetupLibrary.from_game_package(self.loaded_package_path)
                        except Exception: self._fmod_setup_lib=False
                    setup=(self._fmod_setup_lib.get(sm.vorbis_crc32) if self._fmod_setup_lib not in (None,False) else None)
                    if setup:
                        blob=rebuild_fsb5_vorbis_ogg(data,info.bank,sm,setup); ext=".ogg"
                    else:
                        QMessageBox.information(self,"FMOD setup unavailable","This FSB5 Vorbis sample needs the matching FMOD setup packet. Open the full mobile APK/XAPK instead of a bare APF, or export the FSB bank and decode it with vgmstream.")
                        return
                else:
                    QMessageBox.information(self,"Sample codec",f"Direct sample export for FSB5 mode {info.bank.mode_name} is not implemented. Export the FSB bank and use vgmstream.")
                    return
            else:
                return
            safe="".join(c if (c.isalnum() or c in "._-") else "_" for c in str(base)).strip("_") or "audio_sample"
            out,_=QFileDialog.getSaveFileName(self,"Export selected audio sample",safe+ext,f"Audio (*{ext});;All files (*.*)")
            if not out:return
            Path(out).write_bytes(blob); self.statusBar().showMessage(f"Exported audio sample: {Path(out).name}")
        except Exception as exc:
            QMessageBox.critical(self,"Audio sample export failed",f"{exc}\n\n{traceback.format_exc()}")

    def export_audio_asset(self):
        if not self.db: return
        path=self.audio_list.currentItem().text() if self.audio_list.currentItem() else ""
        rec=self.db.get(path) if path else None
        if not rec: return
        try:
            dec=self.db.decode(rec); ext,blob,info=extract_audio_payload(dec.data)
            base=Path(path.split("/",1)[-1]).name or "audio"
            out,_=QFileDialog.getSaveFileName(self,"Export embedded audio",base+ext,f"Audio (*{ext});;All files (*.*)")
            if not out: return
            Path(out).write_bytes(blob)
            self.statusBar().showMessage(f"Exported {info.kind}: {Path(out).name}")
        except Exception as exc:
            QMessageBox.critical(self,"Audio export failed",str(exc))

    def filter_assets(self,text):
        if not self.db:return
        q=(text or "").strip().lower(); self.asset_list.clear()
        paths=sorted(self.db.records_by_path)
        if q:paths=[p for p in paths if q in p.lower()]
        self.asset_list.addItems(paths[:20000])

    def inspect_asset(self,path):
        if not path or not self.db:return
        rec=self.db.get(path)
        if not rec:return
        if path.startswith(("VuAudioBankAsset/","VuAudioStreamAsset/")):
            self.inspect_audio_asset(path); return
        typed=""
        try:
            if path.startswith("VuTextureAsset/"):
                tex=self.db.texture(rec)
                if tex is not None:
                    typed=f"\nTexture: {tex.width}x{tex.height} • mips={tex.mip_count} • format={tex.format_code} • decoder={tex.format_name}\n"
            elif path.startswith("VuCubeTextureAsset/"):
                tex=self.db.cube_texture(rec)
                if tex is not None:
                    typed=f"\nCube texture: {tex.width}x{tex.height} • faces={len(tex.faces)} • mips={tex.mip_count} • format={tex.format_code} • decoder={tex.format_name}\n"
        except Exception as exc:
            typed=f"\nTyped decode error: {exc}\n"
        obj=self.db.json(rec)
        refs=self.db.references_from_json(obj) if obj is not None else []
        self.inspector.setPlainText(
            f"Asset: {rec.path}\nSource: {rec.source_name}\nStored: {rec.stored_size} bytes\nUnpacked: {rec.unpacked_size} bytes\n"+typed
            +("\nReferences:\n"+"\n".join(f"  {k}: {r.path} [{r.source_name}]" for k,r in refs) if refs else "")
        )
        self.raw_json.setPlainText(_safe_json(obj) if obj is not None else "(binary asset; typed metadata shown in Inspector)")

    def closeEvent(self,e):
        if self.db:self.db.close()
        super().closeEvent(e)


def main():
    fmt=QSurfaceFormat(); fmt.setDepthBufferSize(24); fmt.setSamples(4); QSurfaceFormat.setDefaultFormat(fmt)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Grafix.BBRVectorStudio")
        except Exception:
            pass
    app=QApplication(sys.argv); app.setApplicationName(APP_TITLE)
    try:
        if APP_ICON_PATH.exists():
            app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    except Exception:
        pass
    w=MainWindow(); w.show()
    if len(sys.argv)>1 and Path(sys.argv[1]).exists(): QTimer.singleShot(0,lambda:w.load_database(sys.argv[1]))
    sys.exit(app.exec())


if __name__=="__main__": main()
