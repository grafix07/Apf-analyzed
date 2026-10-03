"""Read binary FBX animation curves and retarget poses for Studio preview."""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vector_formats import AnimationData, bind_globals, quat_matrix_engine


FBX_SECOND = 46186158000
ORDERS = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")


@dataclass
class _Node:
    name: str
    props: list = field(default_factory=list)
    children: list = field(default_factory=list)

    def one(self, name):
        return next((n for n in self.children if n.name == name), None)


class _Reader:
    def __init__(self, path, include_geometry=False):
        self.data = Path(path).read_bytes()
        self.include_geometry = include_geometry
        if self.data[:23] != b"Kaydara FBX Binary  \x00\x1a\x00":
            raise ValueError("This FBX is not binary. Export as binary FBX 2014 or newer.")
        self.version = struct.unpack_from("<I", self.data, 23)[0]
        if self.version < 7100 or self.version > 7900:
            raise ValueError(f"Unsupported binary FBX version {self.version}.")
        self.header = 25 if self.version >= 7500 else 13
        self.nodes = []
        at = 27
        while at + self.header <= len(self.data) and any(self.data[at:at+self.header]):
            node, at = self._node(at)
            self.nodes.append(node)

    def one(self, name):
        return next((n for n in self.nodes if n.name == name), None)

    def _node(self, at):
        start = at
        if self.header == 25:
            end, count, prop_size, name_size = struct.unpack_from("<QQQB", self.data, at)
        else:
            end, count, prop_size, name_size = struct.unpack_from("<IIIB", self.data, at)
        if end <= start or end > len(self.data) or count > 100000:
            raise ValueError("Invalid FBX node range or property count.")
        at += self.header
        name = self.data[at:at+name_size].decode("utf-8", "replace")
        at += name_size
        # Motion-only imports skip large mesh and media payloads.
        if not self.include_geometry and name in {"Geometry", "Deformer", "Material", "Texture", "Video", "NodeAttribute"}:
            return _Node(name), end
        prop_end = at + prop_size
        props = []
        for _ in range(count):
            p, at = self._prop(at)
            props.append(p)
        if at != prop_end or at > end:
            raise ValueError(f"Invalid FBX properties in {name}.")
        children = []
        while at < end - self.header:
            child, at = self._node(at)
            children.append(child)
        if at < end and any(self.data[at:end]):
            raise ValueError(f"Invalid FBX child boundary in {name}.")
        return _Node(name, props, children), end

    def _prop(self, at):
        kind = chr(self.data[at]); at += 1
        scalar = {"Y":"h", "C":"?", "I":"i", "F":"f", "D":"d", "L":"q"}
        if kind in scalar:
            fmt = "<" + scalar[kind]
            return struct.unpack_from(fmt, self.data, at)[0], at + struct.calcsize(fmt)
        if kind in ("R", "S"):
            size = struct.unpack_from("<I", self.data, at)[0]; at += 4
            raw = self.data[at:at+size]; at += size
            return (raw.decode("utf-8", "replace") if kind == "S" else
                    (raw if self.include_geometry else None)), at
        types = {"f":"<f4", "d":"<f8", "i":"<i4", "l":"<i8", "b":"u1", "c":"u1"}
        if kind in types:
            count, encoding, size = struct.unpack_from("<III", self.data, at); at += 12
            if count > 10000000 or size > len(self.data) - at or encoding not in (0, 1):
                raise ValueError("Invalid FBX animation array.")
            raw = self.data[at:at+size]; at += size
            raw = zlib.decompress(raw) if encoding else raw
            dtype = np.dtype(types[kind])
            if len(raw) != count * dtype.itemsize:
                raise ValueError("FBX array length does not match its declared size.")
            return np.frombuffer(raw, dtype=dtype).copy(), at
        raise ValueError(f"Unsupported FBX property type {kind!r}.")


def _properties(node):
    section = node.one("Properties70")
    return {n.props[0]: n.props[4:] for n in section.children if n.name == "P" and n.props} if section else {}


def _vector(values, default):
    return np.asarray(values[-3:] if len(values) >= 3 else default, dtype=np.float64)


def _rotation(angles, order):
    """FBX Euler orders are intrinsic; multiplication reverses the axis list."""
    if not 0 <= order < len(ORDERS):
        raise ValueError(f"Unsupported FBX rotation order {order}.")
    order = ORDERS[order]
    angles = np.radians(angles)
    result = np.eye(3, dtype=np.float64)
    for axis in reversed(order):
        c, s = math.cos(angles["XYZ".index(axis)]), math.sin(angles["XYZ".index(axis)])
        if axis == "X": r = np.asarray(((1,0,0),(0,c,-s),(0,s,c)))
        elif axis == "Y": r = np.asarray(((c,0,s),(0,1,0),(-s,0,c)))
        else: r = np.asarray(((c,-s,0),(s,c,0),(0,0,1)))
        result = result @ r
    return result


def _translation(v):
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = v
    return out


def _rot4(r):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    return out


def _local_matrix(translation, rotation, scale, prop):
    order = int(prop.get("RotationOrder", [0])[-1])
    pre = _rotation(_vector(prop.get("PreRotation", []), (0,0,0)), order)
    post = _rotation(_vector(prop.get("PostRotation", []), (0,0,0)), order)
    rot = _rotation(rotation, order)
    rp = _vector(prop.get("RotationPivot", []), (0,0,0))
    ro = _vector(prop.get("RotationOffset", []), (0,0,0))
    sp = _vector(prop.get("ScalingPivot", []), (0,0,0))
    so = _vector(prop.get("ScalingOffset", []), (0,0,0))
    sc = np.diag([*scale, 1.0])
    return (_translation(translation) @ _translation(ro) @ _translation(rp)
            @ _rot4(pre @ rot @ post.T) @ _translation(-rp)
            @ _translation(so) @ _translation(sp) @ sc @ _translation(-sp))


def _axis_matrix(settings):
    def axis(name, sign):
        index = int(settings.get(name, [0])[-1]); direction = int(settings.get(sign, [1])[-1])
        if index not in (0,1,2) or direction not in (-1,1):
            raise ValueError("Invalid FBX scene axis metadata.")
        unit = np.zeros(3); unit[index] = direction
        return unit
    up = axis("UpAxis", "UpAxisSign")
    forward = axis("FrontAxis", "FrontAxisSign")
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 0.5:
        raise ValueError("FBX scene up and front axes overlap.")
    return np.stack([right, forward, up])


@dataclass
class FbxJoint:
    name: str
    parent: int
    properties: dict
    bind: np.ndarray


@dataclass
class FbxClip:
    path: str
    name: str
    joints: list[FbxJoint]
    fps: float
    ticks: np.ndarray
    channels: dict
    basis: np.ndarray
    centimeters_per_unit: float
    skipped_setup_frame: bool

    @property
    def frame_count(self):
        return len(self.ticks)

    def global_frames(self):
        """Evaluate the FBX transform stack on the imported 30 FPS timeline."""
        count = len(self.ticks)
        local = {}
        for ji, joint in enumerate(self.joints):
            p = joint.properties
            for prop, default in (("Lcl Translation", (0,0,0)),
                                  ("Lcl Rotation", (0,0,0)),
                                  ("Lcl Scaling", (1,1,1))):
                base = _vector(p.get(prop, []), default)
                xyz = np.tile(base, (count, 1))
                for axis_index, axis in enumerate("XYZ"):
                    curve = self.channels.get((ji, prop, axis))
                    if curve is not None:
                        ts, vs = curve
                        xyz[:, axis_index] = np.interp(self.ticks, ts, vs)
                local[ji, prop] = xyz
        matrices = np.empty((count, len(self.joints), 4, 4), dtype=np.float64)
        for frame in range(count):
            for ji, joint in enumerate(self.joints):
                m = _local_matrix(*(local[ji, p][frame] for p in
                                    ("Lcl Translation", "Lcl Rotation", "Lcl Scaling")),
                                  joint.properties)
                matrices[frame, ji] = matrices[frame, joint.parent] @ m if joint.parent >= 0 else m
        return matrices


def load_fbx_clip(path):
    reader = _Reader(path)
    objects = reader.one("Objects")
    if objects is None:
        raise ValueError("FBX contains no scene objects.")
    by_id = {n.props[0]: n for n in objects.children if n.props and isinstance(n.props[0], int)}
    links = [n.props for n in reader.one("Connections").children if n.name == "C"]
    model_nodes = [n for n in objects.children if n.name == "Model" and
                   len(n.props) >= 3 and n.props[2] not in ("Mesh", "Camera", "Light")]
    if not model_nodes:
        raise ValueError("FBX contains no character joints.")
    model_ids = {n.props[0] for n in model_nodes}
    parents = {c[1]: c[2] for c in links if c[0] == "OO" and c[1] in model_ids and c[2] in model_ids}
    ordered = []
    def visit(n):
        if n.props[0] in ordered: return
        parent = parents.get(n.props[0])
        if parent in model_ids: visit(by_id[parent])
        ordered.append(n.props[0])
    for n in model_nodes: visit(n)
    lookup = {obj_id: i for i, obj_id in enumerate(ordered)}
    pose = {}
    for p in (n for n in objects.children if n.name == "Pose"):
        if len(p.props) >= 3 and p.props[2] != "BindPose": continue
        for item in p.children:
            node, mat = item.one("Node"), item.one("Matrix")
            if node and mat and len(mat.props[0]) == 16:
                pose[node.props[0]] = np.asarray(mat.props[0], dtype=np.float64).reshape(4,4).T
    joints = []
    for obj_id in ordered:
        node = by_id[obj_id]
        prop = _properties(node)
        joints.append(FbxJoint(node.props[1].split("\x00", 1)[0],
                               lookup.get(parents.get(obj_id), -1), prop,
                               pose.get(obj_id)))
    bind_computed = []
    for j in joints:
        local = _local_matrix(_vector(j.properties.get("Lcl Translation", []), (0,0,0)),
                              _vector(j.properties.get("Lcl Rotation", []), (0,0,0)),
                              _vector(j.properties.get("Lcl Scaling", []), (1,1,1)), j.properties)
        bind_computed.append(bind_computed[j.parent] @ local if j.parent >= 0 else local)
        if j.bind is None: j.bind = bind_computed[-1]

    stacks = [n for n in objects.children if n.name == "AnimationStack"]
    selected_stack = stacks[0].props[0] if stacks else None
    selected_layers = {c[1] for c in links if c[0] == "OO" and c[2] == selected_stack}
    selected_nodes = {c[1] for c in links if c[0] == "OO" and c[2] in selected_layers}
    curve_node_target = {c[1]: (lookup[c[2]], c[3]) for c in links
                         if c[0] == "OP" and len(c) > 3 and c[1] in by_id and
                         by_id[c[1]].name == "AnimationCurveNode" and c[2] in lookup and
                         (not selected_nodes or c[1] in selected_nodes)}
    channels = {}
    for c in links:
        if (c[0] != "OP" or len(c) < 4 or c[2] not in curve_node_target or
                c[1] not in by_id or by_id[c[1]].name != "AnimationCurve"):
            continue
        node = by_id[c[1]]
        times, values = node.one("KeyTime"), node.one("KeyValueFloat")
        if times is None or values is None or not len(times.props[0]): continue
        if len(times.props[0]) != len(values.props[0]):
            raise ValueError("FBX key time/value counts do not match.")
        if np.any(np.diff(times.props[0]) < 0):
            raise ValueError("FBX key times are out of order.")
        ji, prop = curve_node_target[c[2]]
        channels[(ji, prop, c[3][-1])] = (times.props[0], values.props[0])
    animated = [(t,v) for (j,p,a),(t,v) in channels.items() if len(t) > 1 and
                p in ("Lcl Rotation", "Lcl Translation") and np.ptp(v) > 1e-5]
    if not animated:
        raise ValueError("FBX has a skeleton but no changing animation curves.")
    times = max(animated, key=lambda item: len(item[0]))[0]
    if len(times) > 1:
        typical = np.median(np.diff(times)[1:] if len(times) > 2 else np.diff(times))
    else: typical = FBX_SECOND / 30
    global_settings = _properties(reader.one("GlobalSettings")) if reader.one("GlobalSettings") else {}
    custom_fps = float(global_settings.get("CustomFrameRate", [0])[-1] or 0)
    inferred_fps = FBX_SECOND / typical if typical > 0 else 30.0
    custom_mode = int(global_settings.get("TimeMode", [14])[-1]) == 14
    fps = custom_fps if custom_mode and 1 <= custom_fps <= 120 else inferred_fps
    if fps < 1 or fps > 120: fps = custom_fps if 1 <= custom_fps <= 120 else 30.0
    fps = max(1.0, min(120.0, fps))
    skip = len(times) > 3 and times[0] == 0 and times[1]-times[0] > 1.5*typical
    begin, end = int(times[1 if skip else 0]), int(times[-1])
    frame_count = round((end-begin) * fps / FBX_SECOND) + 1
    if frame_count < 2 or frame_count > 20000:
        raise ValueError("Invalid or excessively long FBX animation timeline.")
    sample_ticks = begin + np.arange(frame_count, dtype=np.float64) * FBX_SECOND/fps
    sample_ticks[-1] = end
    unit_scale = float(global_settings.get("UnitScaleFactor", [1])[-1] or 1)
    if unit_scale <= 0: unit_scale = 1
    name = next((n.props[1].split("\x00")[0] for n in stacks
                 if len(n.props) >= 2), Path(path).stem)
    return FbxClip(str(path), name, joints, fps, sample_ticks, channels,
                   _axis_matrix(global_settings), unit_scale, skip)


def _quat_for_engine_rotation(r):
    """The BBR decoder transposes its quaternion matrix; store the inverse."""
    r = r.T
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(max(0.,trace+1.)) * 2
        q = [(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s,s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(max(0.,1+r[i,i]-r[j,j]-r[k,k])) * 2
        q = np.zeros(4); q[i] = s/4
        q[j] = (r[i,j]+r[j,i])/s
        q[k] = (r[i,k]+r[k,i])/s
        q[3] = (r[k,j]-r[j,k])/s
    q = np.asarray(q, dtype=np.float64)
    return (q / max(np.linalg.norm(q),1e-10)).astype(np.float32)


def retarget_fbx(clip: FbxClip, model, mapping: dict[str,str], root_motion=False):
    """Return a normal Studio AnimationData for its existing skinned renderer."""
    if not model.bones:
        raise ValueError("Select a skinned BBR character first.")
    source_names = {j.name: i for i,j in enumerate(clip.joints)}
    linked = {i: source_names[mapping[b.name]] for i,b in enumerate(model.bones)
              if b.name in mapping and mapping[b.name] in source_names}
    required = ("Hips", "Head", "L_Shoulder", "R_Shoulder", "L_Leg", "R_Leg")
    target_names = {b.name for b in model.bones}
    if any(name in target_names and next((i for i,b in enumerate(model.bones) if b.name == name),-1)
           not in linked for name in required):
        raise ValueError("Bone map is missing a hip, head, arm or leg. Check the FBX rig names.")
    source_hips = source_names[mapping["Hips"]]
    target_hips = next(i for i,b in enumerate(model.bones) if b.name == "Hips")
    source = clip.global_frames()
    bind_src = [j.bind for j in clip.joints]
    target_bind = [np.asarray(x,dtype=np.float64) for x in bind_globals(model)]
    bind_local = [np.linalg.inv(target_bind[b.parent]) @ target_bind[i] if b.parent >= 0
                  else target_bind[i] for i,b in enumerate(model.bones)]
    basis = clip.basis
    # Match the character's authored leg length when source/target proportions differ.
    source_ankle = next((source_names[mapping[k]] for k in ("L_Ankle", "R_Ankle")
                         if k in mapping and mapping[k] in source_names), source_hips)
    target_ankle = next((i for i,b in enumerate(model.bones) if b.name in ("L_Ankle", "R_Ankle")), target_hips)
    src_len = abs((basis @ (bind_src[source_hips][:3,3]-bind_src[source_ankle][:3,3]))[2])
    dst_len = abs(target_bind[target_hips][2,3]-target_bind[target_ankle][2,3])
    meters_per_unit = 0.01*clip.centimeters_per_unit
    proportion = dst_len / (src_len*meters_per_unit) if src_len*meters_per_unit > 1e-6 else 1.
    motion_scale = meters_per_unit * max(0.25,min(4.0,proportion))
    n, nb = clip.frame_count, len(model.bones)
    trans = np.zeros((n,nb,3),np.float32)
    rot = np.zeros((n,nb,4),np.float32)
    scale = np.ones((n,nb,3),np.float32)
    base_hips = source[0,source_hips,:3,3]
    bind_r = [m[:3,:3] for m in target_bind]
    inverse_src = {si:np.linalg.inv(bind_src[si][:3,:3]) for si in set(linked.values())}
    for fi in range(n):
        desired_rot = [None]*nb
        offset = basis @ (source[fi,source_hips,:3,3]-base_hips) * motion_scale
        for bi,b in enumerate(model.bones):
            parent = b.parent
            parent_r = desired_rot[parent] if parent >= 0 else np.eye(3)
            si = linked.get(bi)
            if bi == 0:
                world_r = bind_r[bi]
            elif si is not None:
                delta = basis @ source[fi,si,:3,:3] @ inverse_src[si] @ basis.T
                world_r = delta @ bind_r[bi]
            else:
                world_r = parent_r @ bind_local[bi][:3,:3]
            # Keep rotations orthogonal even when an FBX bind matrix has rounding error.
            u, _, vt = np.linalg.svd(world_r)
            world_r = u @ vt
            desired_rot[bi] = world_r
            local_r = parent_r.T @ world_r
            trans[fi,bi] = bind_local[bi][:3,3]
            if bi == 0 and root_motion:
                trans[fi,bi,:2] += offset[:2]
            if bi == target_hips:
                trans[fi,bi] += parent_r.T @ np.asarray((0,0,offset[2]))
            rot[fi,bi] = _quat_for_engine_rotation(local_r)
            scale[fi,bi] = np.asarray([np.linalg.norm(bind_local[bi][:3,c]) for c in range(3)],np.float32)
    return AnimationData(name=Path(clip.path).stem, bone_count=nb, frame_count=n,
                         fps=clip.fps, translations=trans, rotations=rot,
                         scales=scale, loop_flag=0)


def retarget_custom_fbx(clip: FbxClip, model, mapping: dict[str, str], root_motion=False):
    """Apply FBX world-space bind-pose deltas to an imported character rig."""
    if not model.bones:
        raise ValueError("Import a skinned FBX character first.")
    source_names = {joint.name: i for i, joint in enumerate(clip.joints)}
    linked = {i: source_names[mapping[b.name]] for i, b in enumerate(model.bones)
              if b.name in mapping and mapping[b.name] in source_names}
    if len(linked) < 3:
        raise ValueError("Too few matching bones. Check the Target = Source bone map.")
    source = clip.global_frames()
    target_bind = [np.asarray(x, dtype=np.float64) for x in bind_globals(model)]
    bind_local = [np.linalg.inv(target_bind[b.parent]) @ target_bind[i]
                  if b.parent >= 0 else target_bind[i] for i, b in enumerate(model.bones)]
    basis = clip.basis
    inverse_src = {si: np.linalg.inv(clip.joints[si].bind[:3, :3])
                   for si in set(linked.values())}
    # Prefer the pelvis/root translation channel. Imported models often have
    # helper nodes above the hips that carry no motion of their own.
    motion_bone = next((i for i in linked if model.bones[i].name.split(":")[-1].lower()
                        in ("hips", "pelvis")), None)
    if motion_bone is None:
        motion_bone = next((i for i, si in linked.items()
                            if any(key[0] == si and key[1] == "Lcl Translation"
                                   for key in clip.channels)), min(linked))
    source_motion = linked[motion_bone]
    # Match model height when the motion and imported mesh use different units
    # or proportions. A matching rig produces a factor of approximately one.
    source_height = np.ptp([float((basis @ clip.joints[si].bind[:3, 3])[2])
                            for si in linked.values()])
    target_height = np.ptp([float(target_bind[i][2, 3]) for i in linked])
    unit = .01 * clip.centimeters_per_unit
    ratio = target_height / (source_height * unit) if source_height * unit > 1e-7 else 1.
    motion_scale = unit * max(.25, min(4., ratio))
    count, bone_count = clip.frame_count, len(model.bones)
    translations = np.zeros((count, bone_count, 3), np.float32)
    rotations = np.zeros((count, bone_count, 4), np.float32)
    scales = np.ones((count, bone_count, 3), np.float32)
    base_motion = source[0, source_motion, :3, 3]
    for fi in range(count):
        desired = [None] * bone_count
        displacement = basis @ (source[fi, source_motion, :3, 3] - base_motion) * motion_scale
        if not root_motion:
            displacement[:2] = 0
        for bi, bone in enumerate(model.bones):
            parent_rotation = desired[bone.parent] if bone.parent >= 0 else np.eye(3)
            si = linked.get(bi)
            if si is not None:
                world_rotation = (basis @ source[fi, si, :3, :3] @ inverse_src[si]
                                  @ basis.T @ target_bind[bi][:3, :3])
            else:
                world_rotation = parent_rotation @ bind_local[bi][:3, :3]
            u, _, vt = np.linalg.svd(world_rotation)
            world_rotation = u @ vt
            if np.linalg.det(world_rotation) < 0:
                u[:, -1] *= -1
                world_rotation = u @ vt
            desired[bi] = world_rotation
            local_rotation = parent_rotation.T @ world_rotation
            translations[fi, bi] = bind_local[bi][:3, 3]
            if bi == motion_bone:
                translations[fi, bi] += parent_rotation.T @ displacement
            rotations[fi, bi] = _quat_for_engine_rotation(local_rotation)
            scales[fi, bi] = np.linalg.norm(bind_local[bi][:3, :3], axis=0)
    return AnimationData(name=Path(clip.path).stem, bone_count=bone_count,
                         frame_count=count, fps=clip.fps, translations=translations,
                         rotations=rotations, scales=scales, loop_flag=0)
