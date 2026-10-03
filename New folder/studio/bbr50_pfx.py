from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import zlib
import numpy as np

from bbr50_assets import AssetDatabase, AssetRecord
from vector_formats import ModelData, TextureData


# v0.73 simulates the Vector pattern types that are sufficiently decoded for a
# trustworthy editor preview: quads, geometry, trails and light patterns. Other
# pattern types remain visible in the inspector instead of being silently
# converted to generic quads (a source of square/rectangle artefacts in older builds).
SUPPORTED_PATTERN_TYPES = {"VuPfxQuadPattern", "VuPfxGeomPattern", "VuPfxTrailPattern", "VuPfxLightPattern"}
SUPPORTED_EMITTER_TYPES = {"VuPfxEmitQuadFountain", "VuPfxEmitRotFountain", "VuPfxEmitDirectionalQuadFountain"}

# These processor equations are now taken from the reverse-engineered native
# VuPfx implementations.  They are intentionally small and literal: the goal is
# to preserve the authored data, not invent prettier editor-only motion.
SUPPORTED_TICK_TYPES = {
    "VuPfxTickLinearAcceleration",
    "VuPfxTickDampenVelocity",
    "VuPfxTickScale",
    "VuPfxTickAlpha",
    "VuPfxTickAlphaInOut",
    "VuPfxTickAlphaLifeTime",
    "VuPfxSoftKillFade",
    "VuPfxSoftKillScale",
    "VuPfxTickWorldScaleZ",
}

DECODED_NOT_SIMULATED = {
    "VuPfxBoingScale",
    "VuPfxPowerUpBubbleScale",
    "VuPfxSpringConstraint",
    "VuPfxDecelerateToWorldFrame",
    "VuPfxOrientDirGeom",
}


def _v3(v, default=(0.0, 0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return np.asarray(v[:3], dtype=np.float32)
        except Exception:
            pass
    return np.asarray(default, dtype=np.float32)


def _color(v, default=(1, 1, 1, 1)):
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        try:
            return tuple(float(x) for x in v[:4])
        except Exception:
            pass
    return tuple(float(x) for x in default)


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _b(v, d=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(d)
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(d)


def _transform_point(M: Optional[np.ndarray], p: np.ndarray) -> np.ndarray:
    if M is None:
        return np.asarray(p, dtype=np.float32).copy()
    q = np.asarray(M, dtype=np.float32) @ np.asarray((p[0], p[1], p[2], 1.0), dtype=np.float32)
    return np.asarray(q[:3], dtype=np.float32)


def _transform_vector(M: Optional[np.ndarray], v: np.ndarray) -> np.ndarray:
    if M is None:
        return np.asarray(v, dtype=np.float32).copy()
    R = np.asarray(M, dtype=np.float32)[:3, :3]
    return np.asarray(R @ np.asarray(v, dtype=np.float32), dtype=np.float32)


@dataclass
class PfxParticle:
    pattern_name: str
    kind: str                      # quad | geom
    position: np.ndarray
    velocity: np.ndarray
    color: Tuple[float, float, float, float]
    scale: float
    rotation: np.ndarray           # xyz degrees
    angular_velocity: np.ndarray   # xyz deg/sec
    age: float
    lifespan: float
    texture_ref: str = ""
    model_ref: str = ""
    model: Optional[ModelData] = None
    world_space: bool = False
    directional_stretch: float = 0.0
    world_scale_z: float = 1.0


@dataclass
class PfxEmitter50:
    type: str
    name: str
    properties: Dict[str, Any]
    spawn_accumulator: float = 0.0
    spawned_total: int = 0


@dataclass
class PfxProcessor50:
    type: str
    name: str
    properties: Dict[str, Any]


@dataclass
class PfxPattern50:
    name: str
    type: str
    max_particles: int
    properties: Dict[str, Any] = field(default_factory=dict)
    texture_ref: str = ""
    texture_record: Optional[AssetRecord] = None
    texture: Optional[TextureData] = None
    tile_texture_ref: str = ""
    tile_texture_record: Optional[AssetRecord] = None
    tile_texture: Optional[TextureData] = None
    tile_scroll_speed_v: float = 0.0
    model_ref: str = ""
    model_record: Optional[AssetRecord] = None
    model: Optional[ModelData] = None
    emitters: List[PfxEmitter50] = field(default_factory=list)
    processors: List[PfxProcessor50] = field(default_factory=list)
    unsupported_processors: List[str] = field(default_factory=list)
    blend_mode: str = "Additive"
    space: str = "World"
    inherit_system_velocity: bool = True
    respect_scale: bool = True
    start_delay: float = 0.0


@dataclass
class PfxTrailPoint:
    pattern_name: str
    position: np.ndarray
    age: float = 0.0


@dataclass
class PfxSystem50:
    name: str
    source: str
    patterns: List[PfxPattern50]
    unsupported_patterns: List[str] = field(default_factory=list)
    particles: List[PfxParticle] = field(default_factory=list)
    trail_points: List[PfxTrailPoint] = field(default_factory=list)
    active: bool = True
    elapsed: float = 0.0
    soft_kill_age: float = 0.0
    was_active: bool = True

    def report(self) -> str:
        lines = [f"PFX: {self.name}", f"Source: {self.source}", ""]
        for p in self.patterns:
            lines.append(
                f"{p.type} {p.name}: max={p.max_particles} "
                f"space={p.space} blend={p.blend_mode} delay={p.start_delay:g}"
            )
            if p.texture_ref:
                lines.append(f"  Texture: {p.texture_ref}")
            if p.model_ref:
                lines.append(f"  Model: {p.model_ref}")
            lines.append("  Emitters: " + (", ".join(f"{e.type} {e.name}" for e in p.emitters) or "(none)"))
            if p.processors:
                lines.append("  Native processors: " + ", ".join(x.type for x in p.processors))
            if p.unsupported_processors:
                lines.append("  Decoded but not simulated: " + ", ".join(p.unsupported_processors))
        if self.unsupported_patterns:
            lines += ["", "Unsupported pattern types skipped:"] + ["  " + x for x in self.unsupported_patterns]
        return "\n".join(lines)


class PfxLibrary50:
    def __init__(self, db: AssetDatabase):
        self.db = db

    def names(self):
        return sorted(self.db.pfx_systems())

    def build(self, ref: str) -> Optional[PfxSystem50]:
        root = self.db.pfx(ref)
        if not root:
            return None
        patterns: List[PfxPattern50] = []
        unsupported: List[str] = []
        for node in root.get("ChildNodes") or []:
            if not isinstance(node, dict):
                continue
            typ = str(node.get("Type") or "")
            name = str(node.get("Name") or typ)
            if typ not in SUPPORTED_PATTERN_TYPES:
                unsupported.append(f"{typ} {name}")
                continue
            pp = node.get("Properties") if isinstance(node.get("Properties"), dict) else {}
            pat = PfxPattern50(
                name=name,
                type=typ,
                max_particles=max(0, int(pp.get("Max Particle Count") or 0)),
                properties=dict(pp),
                blend_mode=str(pp.get("Blend Mode") or "Additive"),
                space=str(pp.get("Space") or "World"),
                inherit_system_velocity=_b(pp.get("Inherit System Velocity"), True),
                respect_scale=_b(pp.get("Respect Scale"), True),
                start_delay=max(0.0, _f(pp.get("Start Delay"), 0.0)),
            )
            if typ in ("VuPfxQuadPattern", "VuPfxTrailPattern"):
                pat.texture_ref = str(pp.get("Texture Asset") or "")
                if pat.texture_ref:
                    pat.texture_record = self.db.resolve_texture(pat.texture_ref)
                    if pat.texture_record:
                        try:
                            pat.texture = self.db.texture(pat.texture_record)
                        except Exception:
                            pat.texture = None
                # A number of authored flame particles use a colour/animation
                # tile plus a flare mask.  Keep both references so the viewport
                # can prefer the actual fire tile instead of rendering every
                # flame as a plain white flare card.
                pat.tile_texture_ref = str(pp.get("Tile Texture Asset") or "")
                pat.tile_scroll_speed_v = _f(pp.get("Tile Scroll Speed V"), 0.0)
                if pat.tile_texture_ref:
                    pat.tile_texture_record = self.db.resolve_texture(pat.tile_texture_ref)
                    if pat.tile_texture_record:
                        try:
                            pat.tile_texture = self.db.texture(pat.tile_texture_record)
                        except Exception:
                            pat.tile_texture = None
            elif typ == "VuPfxGeomPattern":
                pat.model_ref = str(pp.get("Model Asset") or "")
                if pat.model_ref:
                    pat.model_record = self.db.resolve_model(pat.model_ref)
                    if pat.model_record:
                        try:
                            pat.model = self.db.model(pat.model_record)
                        except Exception:
                            pat.model = None

            for proc in node.get("ChildNodes") or []:
                if not isinstance(proc, dict):
                    continue
                pt = str(proc.get("Type") or "")
                pn = str(proc.get("Name") or pt)
                props = proc.get("Properties") if isinstance(proc.get("Properties"), dict) else {}
                if pt in SUPPORTED_EMITTER_TYPES:
                    pat.emitters.append(PfxEmitter50(pt, pn, dict(props)))
                elif pt in SUPPORTED_TICK_TYPES:
                    pat.processors.append(PfxProcessor50(pt, pn, dict(props)))
                elif pt in DECODED_NOT_SIMULATED or pt:
                    pat.unsupported_processors.append(pt)
            patterns.append(pat)
        return PfxSystem50(str(root.get("Name") or ref), "VuPfxAsset/Generic", patterns, unsupported)

    def reset(self, sys: PfxSystem50, clear_particles: bool = True):
        if clear_particles:
            sys.particles.clear()
            sys.trail_points.clear()
        sys.elapsed = 0.0
        sys.soft_kill_age = 0.0
        sys.was_active = sys.active
        for p in sys.patterns:
            for e in p.emitters:
                e.spawn_accumulator = 0.0
                e.spawned_total = 0

    @staticmethod
    def _seed(sys: PfxSystem50, seed: int, tick_index: int) -> int:
        base = zlib.crc32(sys.name.encode("utf-8", "replace")) & 0xFFFFFFFF
        return base ^ (int(seed) & 0xFFFFFFFF) ^ (int(tick_index) * 0x9E3779B1 & 0xFFFFFFFF)

    def _spawn_one(
        self,
        sys: PfxSystem50,
        pat: PfxPattern50,
        emitter: PfxEmitter50,
        rng: random.Random,
        world_matrix: Optional[np.ndarray],
        system_velocity: Optional[np.ndarray],
    ):
        e = emitter.properties
        minpos = _v3(e.get("Min Position")); maxpos = _v3(e.get("Max Position"), minpos)
        minvel = _v3(e.get("Min Linear Velocity")); maxvel = _v3(e.get("Max Linear Velocity"), minvel)
        pos = np.asarray([rng.uniform(float(a), float(b)) for a, b in zip(minpos, maxpos)], dtype=np.float32)
        vel = np.asarray([rng.uniform(float(a), float(b)) for a, b in zip(minvel, maxvel)], dtype=np.float32)

        mins = _f(e.get("Min Scale"), 1.0); maxs = _f(e.get("Max Scale"), mins)
        life_min = _f(e.get("Min Lifespan"), _f(e.get("Max Lifespan"), 1.0))
        life_max = _f(e.get("Max Lifespan"), life_min)
        c0 = np.asarray(_color(e.get("Min Color")), dtype=np.float32)
        c1 = np.asarray(_color(e.get("Max Color"), c0), dtype=np.float32)
        t = rng.random(); col = tuple((c0 + (c1 - c0) * t).tolist())

        rmin = e.get("Min Rotation", 0); rmax = e.get("Max Rotation", rmin)
        if isinstance(rmin, (list, tuple)) or isinstance(rmax, (list, tuple)):
            a0 = _v3(rmin); a1 = _v3(rmax, a0)
            rot = np.asarray([rng.uniform(float(min(a, b)), float(max(a, b))) for a, b in zip(a0, a1)], dtype=np.float32)
        else:
            lo = _f(rmin, 0); hi = _f(rmax, lo)
            rot = np.asarray((0, 0, rng.uniform(min(lo, hi), max(lo, hi))), dtype=np.float32)

        av0 = e.get("Min Angular Velocity", 0); av1 = e.get("Max Angular Velocity", av0)
        if isinstance(av0, (list, tuple)) or isinstance(av1, (list, tuple)):
            a0 = _v3(av0); a1 = _v3(av1, a0)
            av = np.asarray([rng.uniform(float(min(a, b)), float(max(a, b))) for a, b in zip(a0, a1)], dtype=np.float32)
        else:
            lo = _f(av0); hi = _f(av1, lo)
            av = np.asarray((0, 0, rng.uniform(min(lo, hi), max(lo, hi))), dtype=np.float32)

        world_space = pat.space.strip().lower() != "object"
        if world_space:
            pos = _transform_point(world_matrix, pos)
            vel = _transform_vector(world_matrix, vel)
            if pat.inherit_system_velocity and system_velocity is not None:
                vel = vel + np.asarray(system_velocity, dtype=np.float32)

        stretch_min = max(0.0, _f(e.get("Min Directional Stretch"), 0.0))
        stretch_max = max(stretch_min, _f(e.get("Max Directional Stretch"), stretch_min))
        # Directional fountain particles are velocity-oriented even when the
        # asset omits the explicit stretch properties.  Use a tiny nonzero
        # factor in that case: it preserves authored direction without turning
        # smoke into long spark streaks.
        if emitter.type == "VuPfxEmitDirectionalQuadFountain" and stretch_max <= 0.0:
            stretch_min = stretch_max = 0.015
        wsz0 = max(0.01, _f(e.get("Min World Scale Z"), 1.0))
        wsz1 = max(0.01, _f(e.get("Max World Scale Z"), wsz0))

        sys.particles.append(PfxParticle(
            pattern_name=pat.name,
            kind=("quad" if pat.type == "VuPfxQuadPattern" else "light" if pat.type == "VuPfxLightPattern" else "geom"),
            position=pos,
            velocity=vel,
            color=col,
            scale=rng.uniform(min(mins, maxs), max(mins, maxs)),
            rotation=rot,
            angular_velocity=av,
            age=0.0,
            lifespan=max(0.01, rng.uniform(min(life_min, life_max), max(life_min, life_max))),
            texture_ref=pat.texture_ref,
            model_ref=pat.model_ref,
            model=pat.model,
            world_space=world_space,
            directional_stretch=rng.uniform(stretch_min, stretch_max) if stretch_max > 0.0 else 0.0,
            world_scale_z=rng.uniform(min(wsz0, wsz1), max(wsz0, wsz1)),
        ))
        emitter.spawned_total += 1

    @staticmethod
    def _processor_started(proc: PfxProcessor50, particle: PfxParticle) -> bool:
        p = proc.properties
        delay = _f(p.get("Start Delay"), _f(p.get("Delay"), 0.0))
        return particle.age > delay

    def _apply_processor(self, proc: PfxProcessor50, x: PfxParticle, dt: float, soft_killing: bool):
        p = proc.properties
        pt = proc.type

        if pt == "VuPfxTickLinearAcceleration":
            if not self._processor_started(proc, x):
                return
            acc = np.asarray((_f(p.get("Accel X")), _f(p.get("Accel Y")), _f(p.get("Accel Z"))), dtype=np.float32)
            x.velocity += acc * dt
            return

        if pt == "VuPfxTickDampenVelocity":
            if not self._processor_started(proc, x):
                return
            # Native equation: velocity *= 1 - min(dt * Amount, 1).
            amount = max(0.0, _f(p.get("Amount"), 0.0))
            factor = 1.0 - min(dt * amount, 1.0)
            x.velocity *= factor
            return

        if pt == "VuPfxTickScale":
            if not self._processor_started(proc, x):
                return
            x.scale += dt * _f(p.get("Rate"), 0.0)
            x.scale = max(0.0, x.scale)
            return

        if pt == "VuPfxTickAlpha":
            if not self._processor_started(proc, x):
                return
            c = list(x.color)
            c[3] = max(0.0, min(1.0, c[3] + dt * _f(p.get("Rate"), 0.0)))
            x.color = tuple(c)
            return

        if pt == "VuPfxTickAlphaInOut":
            # Native: fade-in rate is applied while age <= Fade In Duration;
            # fade-out rate starts once age exceeds Fade Out Start Time.
            delta = 0.0
            if x.age <= _f(p.get("Fade In Duration"), 1.0):
                delta += dt * _f(p.get("Fade In Rate"), 1.0)
            if x.age > _f(p.get("Fade Out Start Time"), 2.0):
                delta += dt * _f(p.get("Fade Out Rate"), -1.0)
            if delta:
                c = list(x.color); c[3] = max(0.0, min(1.0, c[3] + delta)); x.color = tuple(c)
            return

        if pt == "VuPfxTickAlphaLifeTime":
            # Native: fade-out begins at (particle_lifespan - Fade Out Duration).
            delta = 0.0
            if x.age <= _f(p.get("Fade In Duration"), 1.0):
                delta += dt * _f(p.get("Fade In Rate"), 1.0)
            if x.age > x.lifespan - _f(p.get("Fade Out Duration"), 1.0):
                delta += dt * _f(p.get("Fade Out Rate"), -1.0)
            if delta:
                c = list(x.color); c[3] = max(0.0, min(1.0, c[3] + delta)); x.color = tuple(c)
            return

        if pt == "VuPfxSoftKillFade":
            if not soft_killing or not self._processor_started(proc, x):
                return
            c = list(x.color)
            c[3] = max(0.0, min(1.0, c[3] + dt * _f(p.get("Rate"), -1.0)))
            x.color = tuple(c)
            return

        if pt == "VuPfxSoftKillScale":
            if not soft_killing or not self._processor_started(proc, x):
                return
            x.scale = max(0.0, x.scale + dt * _f(p.get("Rate"), -1.0))
            return

        if pt == "VuPfxTickWorldScaleZ":
            if not self._processor_started(proc, x):
                return
            x.world_scale_z = max(0.01, float(x.world_scale_z) + dt * _f(p.get("Rate"), 0.0))
            return

    def step(
        self,
        sys: PfxSystem50,
        dt: float,
        seed: int = 0,
        world_matrix: Optional[np.ndarray] = None,
        system_velocity: Optional[np.ndarray] = None,
    ):
        dt = max(0.0, min(float(dt), 0.1))

        just_activated = bool(sys.active and not sys.was_active)
        if just_activated:
            sys.elapsed = 0.0
            sys.soft_kill_age = 0.0
            # A native PFX restart resets spawn counters, but already-emitted
            # world particles can continue their natural lifetime.
            for pat in sys.patterns:
                for emitter in pat.emitters:
                    emitter.spawn_accumulator = 0.0
                    emitter.spawned_total = 0

        if sys.active:
            sys.elapsed += dt
            sys.soft_kill_age = 0.0
        else:
            sys.soft_kill_age += dt

        tick_index = int(round((sys.elapsed if sys.active else sys.soft_kill_age) * 1000.0))
        rng = random.Random(self._seed(sys, seed, tick_index))

        if sys.active:
            for pat in sys.patterns:
                if sys.elapsed < pat.start_delay or not pat.emitters or pat.max_particles <= 0:
                    continue
                current = sum(1 for x in sys.particles if x.pattern_name == pat.name)
                for emitter in pat.emitters:
                    ep = emitter.properties
                    raw_rate = ep.get("Spawn Per Second", None)
                    rate = max(0.0, _f(raw_rate, 0.0))
                    maxspawn = int(ep.get("Max Spawn Count") or 0)
                    if raw_rate is None and maxspawn > 0 and emitter.spawned_total == 0 and maxspawn <= pat.max_particles:
                        # Native one-shot/static emitters commonly omit Spawn
                        # Per Second entirely.  v0.60 treated that as rate=0,
                        # which is why Tribal's authored Head Geom mask never
                        # spawned.  A finite Max Spawn Count <= pattern capacity
                        # is an unambiguous burst request.
                        n = maxspawn
                    else:
                        emitter.spawn_accumulator += rate * dt
                        n = int(emitter.spawn_accumulator)
                        emitter.spawn_accumulator -= n
                    for _ in range(n):
                        if current >= pat.max_particles:
                            break
                        if maxspawn > 0 and emitter.spawned_total >= maxspawn:
                            break
                        self._spawn_one(sys, pat, emitter, rng, world_matrix, system_velocity)
                        current += 1

        # VuPfxTrailPattern has no emitter node: the trail is generated from
        # motion of the PFX system itself.  Keep a short world-space history so
        # AstroDog's authored Fire_Tile jet ribbons appear instead of being
        # silently skipped.
        trail_patterns = [p for p in sys.patterns if p.type == "VuPfxTrailPattern"]
        for tp in sys.trail_points:
            tp.age += dt
        if sys.active and world_matrix is not None:
            origin = _transform_point(world_matrix, np.zeros(3, dtype=np.float32))
            for pat in trail_patterns:
                pts = [x for x in sys.trail_points if x.pattern_name == pat.name]
                should_add = not pts
                if pts:
                    should_add = float(np.linalg.norm(origin - pts[-1].position)) > 0.01
                if should_add:
                    sys.trail_points.append(PfxTrailPoint(pat.name, origin.copy(), 0.0))
                    pts.append(sys.trail_points[-1])
                maxp = max(2, int(pat.max_particles or 30))
                if len(pts) > maxp:
                    remove_ids = set(id(x) for x in pts[:-maxp])
                    sys.trail_points = [x for x in sys.trail_points if id(x) not in remove_ids]
        # Trail points fade after the system is stopped; while active, the
        # capacity limit above naturally drops the oldest points.
        keep_trails=[]
        for tp in sys.trail_points:
            pat=next((p for p in trail_patterns if p.name==tp.pattern_name),None)
            fade=max(0.05,_f((pat.properties if pat else {}).get("Fade Out Time"),0.9))
            if sys.active or tp.age <= fade:
                keep_trails.append(tp)
        sys.trail_points=keep_trails

        pat_by = {p.name: p for p in sys.patterns}
        kept: List[PfxParticle] = []
        soft_killing = not sys.active
        for x in sys.particles:
            x.age += dt
            if x.age >= x.lifespan:
                continue
            pat = pat_by.get(x.pattern_name)
            if pat:
                for proc in pat.processors:
                    self._apply_processor(proc, x, dt, soft_killing)
            x.position += x.velocity * dt
            x.rotation += x.angular_velocity * dt
            # Fully faded/shrunk soft-killed particles can be discarded early.
            if soft_killing and (x.color[3] <= 0.0001 or x.scale <= 0.0001):
                continue
            kept.append(x)
        sys.particles = kept
        sys.was_active = bool(sys.active)
        return sys.particles
