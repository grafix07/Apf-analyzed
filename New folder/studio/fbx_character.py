"""Import a skinned binary FBX character into the Studio's preview model."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fbx_motion import (_Reader, _axis_matrix, _local_matrix, _properties,
                        _quat_for_engine_rotation, _vector)
from vector_formats import Bone, MeshData, ModelData, TextureData, bind_globals, recompute_mesh_normals


@dataclass
class CustomCharacter:
    name: str
    model: ModelData
    source: str
    skin: str = "Imported FBX"
    animations: tuple = ()

    def report(self):
        vertices = sum(len(m.positions) for m in self.model.meshes)
        triangles = sum(len(m.indices) // 3 for m in self.model.meshes)
        weighted = sum(m.weights is not None for m in self.model.meshes)
        textured = sum(getattr(m, "custom_texture", None) is not None for m in self.model.meshes)
        return (f"Custom character: {self.name}\nFile: {self.source}\n"
                f"Meshes: {len(self.model.meshes)} ({weighted} skinned)\n"
                f"Meshes with image textures: {textured}\n"
                f"Vertices: {vertices:,} (including UV/normal seams)\n"
                f"Triangles: {triangles:,}\nBones: {len(self.model.bones)}\n"
                "Load a motion FBX in External Animation and adjust the bone map if needed.")


def _array(node, name):
    child = node.one(name)
    return np.asarray(child.props[0]) if child and child.props else np.empty(0)


def _name(node):
    return str(node.props[1]).split("\x00", 1)[0]


def _texture_at(texture_node, source_path):
    if texture_node is None:
        return None
    candidate = next((str(n.props[0]) for key in ("RelativeFilename", "FileName")
                      for n in [texture_node.one(key)] if n and n.props and n.props[0]), None)
    folder = Path(source_path).resolve().parent
    normalized = candidate.replace("\\", "/") if candidate else ""
    files = [folder / normalized, folder / Path(normalized).name] if normalized else []
    path = next((p for p in files if p.resolve().is_relative_to(folder) and p.is_file()), None)
    content = texture_node.one("Content")
    embedded = content.props[0] if content and content.props and isinstance(content.props[0], bytes) else None
    if path is None and not embedded:
        return None
    try:
        with Image.open(path if path is not None else BytesIO(embedded)) as im:
            im.thumbnail((8192, 8192))
            rgba = np.asarray(im.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        return TextureData(path.name if path else "Embedded FBX image", rgba.shape[1], rgba.shape[0], 1, 0, rgba)
    except (OSError, ValueError):
        return None


def _bind_pose(objects):
    pose = {}
    for p in (n for n in objects.children if n.name == "Pose"):
        if len(p.props) >= 3 and p.props[2] != "BindPose":
            continue
        for item in p.children:
            node, mat = item.one("Node"), item.one("Matrix")
            if node and mat and mat.props and len(mat.props[0]) == 16:
                pose[node.props[0]] = np.asarray(mat.props[0], dtype=np.float64).reshape(4, 4).T
    return pose


def _layer_values(geometry, layer_name, data_name, index_name, width,
                  corners, control_points, default):
    layer = next((x for x in geometry.children if x.name == layer_name), None)
    if layer is None:
        return np.tile(default, (len(corners), 1)).astype(np.float32), False
    raw = _array(layer, data_name)
    if not len(raw) or len(raw) % width:
        return np.tile(default, (len(corners), 1)).astype(np.float32), False
    values = raw.reshape(-1, width)
    mapping = layer.one("MappingInformationType")
    mapping = mapping.props[0] if mapping and mapping.props else "ByPolygonVertex"
    if mapping == "ByPolygonVertex":
        selected = np.asarray(corners, dtype=np.int64)
    elif mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        selected = np.asarray(control_points, dtype=np.int64)
    elif mapping == "AllSame":
        selected = np.zeros(len(corners), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported FBX {layer_name} mapping {mapping}.")
    reference = layer.one("ReferenceInformationType")
    reference = reference.props[0] if reference and reference.props else "Direct"
    if reference in ("IndexToDirect", "Index"):
        indices = _array(layer, index_name)
        if len(indices) <= int(selected.max(initial=-1)):
            raise ValueError(f"FBX {layer_name} index array is incomplete.")
        selected = indices[selected]
    elif reference != "Direct":
        raise ValueError(f"Unsupported FBX {layer_name} reference {reference}.")
    if np.any(selected < 0) or np.any(selected >= len(values)):
        raise ValueError(f"FBX {layer_name} refers outside its data array.")
    return values[selected].astype(np.float32), True


def _triangulate(encoded, vertex_count, flip=False):
    control, corners, triangles, triangle_faces = [], [], [], []
    face = []
    face_id = 0
    for at, raw in enumerate(encoded):
        index = int(raw if raw >= 0 else -raw - 1)
        if not 0 <= index < vertex_count:
            raise ValueError("FBX polygon index refers outside the mesh vertices.")
        control.append(index)
        corners.append(at)
        face.append(at)
        if raw < 0:
            for t in range(1, len(face) - 1):
                tri = (face[0], face[t + 1], face[t]) if flip else (face[0], face[t], face[t + 1])
                triangles.extend(tri)
                triangle_faces.append(face_id)
            face = []
            face_id += 1
    if face:
        raise ValueError("FBX polygon has no end marker.")
    return (np.asarray(control, dtype=np.int64), corners,
            np.asarray(triangles, dtype=np.uint32).reshape(-1, 3),
            np.asarray(triangle_faces, dtype=np.int64))


def _material_slots(geometry, triangle_faces):
    layer = next((x for x in geometry.children if x.name == "LayerElementMaterial"), None)
    if layer is None:
        return np.zeros(len(triangle_faces), dtype=np.int64)
    raw = _array(layer, "Materials")
    mapping = layer.one("MappingInformationType")
    mode = mapping.props[0] if mapping and mapping.props else "AllSame"
    if not len(raw):
        return np.zeros(len(triangle_faces), dtype=np.int64)
    if mode == "AllSame":
        return np.full(len(triangle_faces), int(raw[0]), dtype=np.int64)
    if mode == "ByPolygon" and int(triangle_faces.max(initial=-1)) < len(raw):
        return raw[triangle_faces].astype(np.int64)
    return np.zeros(len(triangle_faces), dtype=np.int64)


def load_fbx_character(path):
    """Read mesh, bind pose and skin weights from a binary FBX scene.

    Source geometry is moved to world space and then into the Studio's metre,
    Z-up basis. The skeleton bind matrices undergo the same conversion.
    """
    reader = _Reader(path, include_geometry=True)
    objects = reader.one("Objects")
    connections = reader.one("Connections")
    if objects is None or connections is None:
        raise ValueError("FBX contains no scene objects or connections.")
    by_id = {n.props[0]: n for n in objects.children if n.props and isinstance(n.props[0], int)}
    links = [n.props for n in connections.children if n.name == "C" and len(n.props) >= 3]
    parent = {c[1]: c[2] for c in links if c[0] == "OO" and c[1] in by_id}
    models = {i: n for i, n in by_id.items() if n.name == "Model"}
    joints = {i: n for i, n in models.items() if len(n.props) >= 3 and n.props[2] not in ("Mesh", "Camera", "Light")}
    geometries = [(i, n) for i, n in by_id.items() if n.name == "Geometry" and len(n.props) >= 3 and n.props[2] == "Mesh"]
    if not geometries:
        raise ValueError("FBX has no polygon mesh. Export a character with its mesh and skeleton.")
    if not joints:
        raise ValueError("FBX has no skeleton. Export a rigged character to play animations.")
    settings_node = reader.one("GlobalSettings")
    settings = _properties(settings_node) if settings_node else {}
    basis = _axis_matrix(settings)
    unit = float(settings.get("UnitScaleFactor", [1])[-1] or 1) * .01
    if unit <= 0:
        raise ValueError("FBX has an invalid unit scale.")
    pose = _bind_pose(objects)

    computed = {}
    visiting = set()
    def model_world(i):
        if i in computed:
            return computed[i]
        if i in visiting:
            raise ValueError("FBX has a cycle in the model hierarchy.")
        visiting.add(i)
        prop = _properties(models[i])
        local = _local_matrix(_vector(prop.get("Lcl Translation", []), (0, 0, 0)),
                              _vector(prop.get("Lcl Rotation", []), (0, 0, 0)),
                              _vector(prop.get("Lcl Scaling", []), (1, 1, 1)), prop)
        p = parent.get(i)
        result = model_world(p) @ local if p in models else local
        visiting.remove(i)
        computed[i] = result
        return result

    ordered = []
    def visit(i):
        if i in ordered:
            return
        p = parent.get(i)
        if p in joints:
            visit(p)
        ordered.append(i)
    for i in joints:
        visit(i)
    joint_lookup = {i: at for at, i in enumerate(ordered)}
    bones = []
    for at, i in enumerate(ordered):
        world = pose.get(i)
        if world is None:
            world = model_world(i)
        linear = basis @ world[:3, :3] @ basis.T
        scales = np.linalg.norm(linear, axis=0)
        if np.any(scales < 1e-8):
            raise ValueError("FBX skeleton contains a bone with zero scale.")
        rotation = linear / scales
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
            scales[2] *= -1
        bones.append(Bone(at, _name(joints[i]), joint_lookup.get(parent.get(i), -1),
                          (basis @ world[:3, 3] * unit).astype(np.float32),
                          _quat_for_engine_rotation(rotation), scales.astype(np.float32)))

    # Geometry -> Model and Cluster -> Skin -> Geometry links; joint -> Cluster.
    geometry_models = {c[1]: c[2] for c in links if c[0] == "OO" and c[1] in by_id
                       and by_id[c[1]].name == "Geometry" and c[2] in models}
    skins = {i: parent.get(i) for i, n in by_id.items() if n.name == "Deformer"
             and len(n.props) >= 3 and n.props[2] == "Skin"}
    cluster_bones = {c[2]: joint_lookup[c[1]] for c in links if c[0] == "OO"
                     and c[1] in joint_lookup and c[2] in by_id
                     and by_id[c[2]].name == "Deformer"}
    model_materials = {}
    for c in links:
        if c[0] == "OO" and c[1] in by_id and by_id[c[1]].name == "Material" and c[2] in models:
            model_materials.setdefault(c[2], []).append(c[1])
    material_textures = {}
    for c in links:
        if (c[0] in ("OP", "OO") and c[1] in by_id and by_id[c[1]].name == "Texture"
                and c[2] in by_id and by_id[c[2]].name == "Material"):
            if c[0] == "OO" or len(c) < 4 or "diffuse" in str(c[3]).lower():
                material_textures[c[2]] = c[1]
    video_textures = {c[2]: c[1] for c in links if c[0] == "OO" and c[1] in by_id
                      and by_id[c[1]].name == "Video" and c[2] in by_id
                      and by_id[c[2]].name == "Texture"}
    texture_cache = {}
    meshes = []
    for geometry_id, geometry in geometries:
        raw_positions = _array(geometry, "Vertices")
        encoded = _array(geometry, "PolygonVertexIndex")
        if not len(raw_positions) or len(raw_positions) % 3 or not len(encoded):
            continue
        source_points = raw_positions.reshape(-1, 3)
        mesh_model_id = geometry_models.get(geometry_id)
        mesh_world = (pose.get(mesh_model_id, model_world(mesh_model_id)) if mesh_model_id in models
                      else np.eye(4, dtype=np.float64))
        mesh_prop = _properties(models[mesh_model_id]) if mesh_model_id in models else {}
        geometric = _local_matrix(_vector(mesh_prop.get("GeometricTranslation", []), (0, 0, 0)),
                                  _vector(mesh_prop.get("GeometricRotation", []), (0, 0, 0)),
                                  _vector(mesh_prop.get("GeometricScaling", []), (1, 1, 1)), {})
        mesh_world = mesh_world @ geometric
        flip = np.linalg.det(mesh_world[:3, :3]) * np.linalg.det(basis) < 0
        control, corners, triangles, triangle_faces = _triangulate(encoded, len(source_points), flip=flip)
        if not len(triangles):
            continue
        points = source_points[control]
        positions = ((points @ mesh_world[:3, :3].T + mesh_world[:3, 3]) @ basis.T * unit).astype(np.float32)
        normals, has_normals = _layer_values(geometry, "LayerElementNormal", "Normals", "NormalsIndex", 3,
                                             corners, control, (0, 0, 1))
        if has_normals:
            normal_matrix = np.linalg.inv(mesh_world[:3, :3]).T
            normals = (normals @ normal_matrix.T @ basis.T).astype(np.float32)
            length = np.linalg.norm(normals, axis=1)
            normals /= np.maximum(length[:, None], 1e-12)
        else:
            normals = recompute_mesh_normals(positions, triangles)
        uvs, _ = _layer_values(geometry, "LayerElementUV", "UV", "UVIndex", 2,
                               corners, control, (0, 0))

        influences = [[] for _ in source_points]
        for cluster_id, cluster in by_id.items():
            if (cluster.name != "Deformer" or len(cluster.props) < 3 or
                    cluster.props[2] != "Cluster" or skins.get(parent.get(cluster_id)) != geometry_id):
                continue
            bi = cluster_bones.get(cluster_id)
            if bi is None:
                continue
            vertices, weights = _array(cluster, "Indexes"), _array(cluster, "Weights")
            if len(vertices) != len(weights):
                raise ValueError("FBX skin cluster has mismatched index and weight arrays.")
            for vi, weight in zip(vertices, weights):
                if 0 <= int(vi) < len(influences) and float(weight) > 0:
                    influences[int(vi)].append((float(weight), bi))
        weights_out = joints_out = None
        if any(influences):
            weights_out = np.zeros((len(control), 4), dtype=np.float32)
            joints_out = np.zeros((len(control), 4), dtype=np.int32)
            for vi, cp in enumerate(control):
                strongest = sorted(influences[int(cp)], reverse=True)[:4]
                total = sum(weight for weight, _ in strongest)
                if total > 1e-9:
                    for ci, (weight, bi) in enumerate(strongest):
                        weights_out[vi, ci] = weight / total
                        joints_out[vi, ci] = bi
        mesh_name = _name(models[mesh_model_id]) if mesh_model_id in models else _name(geometry)
        materials = model_materials.get(mesh_model_id, [])
        slots = _material_slots(geometry, triangle_faces)
        for slot in np.unique(slots):
            selected = triangles[slots == slot]
            if not len(selected):
                continue
            vertices, remapped = np.unique(selected.reshape(-1), return_inverse=True)
            label = mesh_name if len(np.unique(slots)) == 1 else f"{mesh_name} [{int(slot)}]"
            mesh = MeshData(label, positions[vertices], normals[vertices], uvs[vertices],
                            remapped.astype(np.uint32),
                            weights=weights_out[vertices] if weights_out is not None else None,
                            joints=joints_out[vertices] if joints_out is not None else None)
            material_id = materials[int(slot)] if 0 <= int(slot) < len(materials) else None
            if material_id is not None:
                props = _properties(by_id[material_id])
                color = _vector(props.get("DiffuseColor", []), (1, 1, 1))
                factor = float(props.get("DiffuseFactor", [1])[-1] if props.get("DiffuseFactor") else 1)
                mesh.custom_color = np.asarray([*np.clip(color * factor, 0, 1), 1], dtype=np.float32)
                texture_id = material_textures.get(material_id)
                if texture_id is not None:
                    texture_node = by_id.get(texture_id)
                    video_node = by_id.get(video_textures.get(texture_id))
                    for node in (texture_node, video_node):
                        if node is None: continue
                        key = (texture_id, node.props[0])
                        if key not in texture_cache:
                            texture_cache[key] = _texture_at(node, path)
                        if texture_cache[key] is not None:
                            mesh.custom_texture = texture_cache[key]
                            break
            meshes.append(mesh)
    if not meshes:
        raise ValueError("FBX contains no drawable polygons.")
    if not any(m.weights is not None for m in meshes):
        raise ValueError("FBX mesh has no skin weights. Export a skinned character, not just a skeleton or prop.")
    model = ModelData(Path(path).stem, meshes, bones, str(path))
    return CustomCharacter(Path(path).stem, model, str(path))


def skin_custom_mesh(mesh, model, current_globals, bone_visible=None):
    """Vectorized skinning for the polygon-corner vertices typical of FBX."""
    if mesh.weights is None or mesh.joints is None:
        return mesh.positions, mesh.normals
    bind = bind_globals(model)
    if len(bind) != len(current_globals):
        return mesh.positions, mesh.normals
    skin = np.asarray([np.asarray(g, np.float32) @ np.linalg.inv(b)
                       for g, b in zip(current_globals, bind)], dtype=np.float32)
    normal_mats = np.asarray([np.linalg.inv(s[:3, :3]).T for s in skin], dtype=np.float32)
    positions = np.asarray(mesh.positions, dtype=np.float32)
    normals = np.asarray(mesh.normals, dtype=np.float32)
    p4 = np.concatenate((positions, np.ones((len(positions), 1), np.float32)), axis=1)
    out_pos = np.zeros_like(positions)
    out_norm = np.zeros_like(normals)
    used = np.zeros(len(positions), np.float32)
    for channel in range(min(4, mesh.weights.shape[1])):
        weights = mesh.weights[:, channel]
        joints = mesh.joints[:, channel]
        active = (weights > 1e-7) & (joints >= 0) & (joints < len(skin))
        if bone_visible is not None:
            flags = np.asarray(bone_visible, dtype=bool)
            active &= flags[np.clip(joints, 0, len(flags)-1)]
        if not np.any(active):
            continue
        matrix = skin[joints[active]]
        nmatrix = normal_mats[joints[active]]
        weight = weights[active, None]
        out_pos[active] += np.einsum("nij,nj->ni", matrix, p4[active])[:, :3] * weight
        out_norm[active] += np.einsum("nij,nj->ni", nmatrix, normals[active]) * weight
        used[active] += weights[active]
    weighted = used > 1e-8
    out_pos[weighted] /= used[weighted, None]
    out_norm[weighted] /= used[weighted, None]
    unweighted = mesh.weights.sum(axis=1) <= 1e-8
    out_pos[unweighted] = positions[unweighted]
    out_norm[unweighted] = normals[unweighted]
    lengths = np.linalg.norm(out_norm, axis=1)
    good = lengths > 1e-8
    out_norm[good] /= lengths[good, None]
    return out_pos, out_norm
