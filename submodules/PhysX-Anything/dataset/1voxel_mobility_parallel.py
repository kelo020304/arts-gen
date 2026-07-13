"""
Parallelized version of 1voxel_mobility.py
- Global transform computed once (sequential)
- Per-part voxelization parallelized across CPU cores
"""
import open3d as o3d
import numpy as np
import trimesh
import os
import json
import pickle
import logging
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def movtran(orimesh, offset, scale):
    orimesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    orimesh.apply_translation([offset[0], offset[1], offset[2]])
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    orimesh.apply_transform(rotation_matrix)
    return orimesh

def generate_voxel(mesh, voxel_define=64):
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=81920, init_factor=3, pcl=None)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        point_cloud, voxel_size=1/voxel_define,
        min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < voxel_define)
    vertices = (vertices + 0.5) / voxel_define - 0.5
    indices = ((vertices + 0.5) * voxel_define)
    indices = np.asarray(indices, dtype=np.int64)
    return indices, vertices

def load_obj_geometry_fast(path):
    V, F = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                V.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                F.append(idx)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return trimesh.Trimesh(V, F, process=False)

def process_part(args):
    """Process a single part at all 3 resolutions. Runs in a worker process."""
    nameind, obj_names, objpath, offset, scale, tmpdir, index = args

    # Load and transform part mesh
    oripart1 = trimesh.Trimesh([])
    for meshname in obj_names:
        eachpart1 = load_obj_geometry_fast(os.path.join(objpath, meshname + '.obj'))
        oripart1 = trimesh.util.concatenate([oripart1, eachpart1])
    movtran(oripart1, offset, scale)

    results = {}
    for res in [16, 32, 64]:
        savepath = os.path.join(tmpdir, 'partseg', index, str(res))
        os.makedirs(savepath, exist_ok=True)

        # Export and reload as open3d mesh
        ply_path = os.path.join(savepath, f'mesh_new_{nameind}.ply')
        oripart1.export(ply_path)
        part = o3d.io.read_triangle_mesh(ply_path)
        indices1, vertices1 = generate_voxel(part, res)

        np.save(os.path.join(savepath, f'ind_{nameind}.npy'), indices1)
        results[(nameind, res)] = (indices1, vertices1)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind", type=int, default=0)
    parser.add_argument("--range", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (cpu_count)")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else min(cpu_count(), 64)

    tmpdir = 'tmp_mobility/'
    basepath = './PhysX_mobility'
    os.makedirs(tmpdir, exist_ok=True)

    indexlist = os.listdir(os.path.join(basepath, 'partseg'))
    logger = get_logger(f'./tmp_mobility/exp_1render_mobility_par{args.ind}.log', verbosity=1)
    logger.info(f'start (parallel, {num_workers} workers)')
    os.makedirs(os.path.join(tmpdir, 'partseg'), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, 'finaljson'), exist_ok=True)

    indexlist = indexlist[args.ind * args.range:(args.ind + 1) * args.range]

    for index in indexlist:
        logger.info('begin: ' + index)
        objpath = os.path.join(basepath, 'partseg', index, 'objs')
        namelist = [f for f in os.listdir(objpath) if f.endswith('.obj')]
        jsonfile = os.path.join(basepath, 'finaljson', index + '.json')

        with open(jsonfile, 'r') as fp:
            jsondata = json.load(fp)

        if os.path.exists(os.path.join(tmpdir, 'partseg', index, '64', 'allind.npy')):
            logger.info('skip: ' + index)
            continue

        # === Sequential: compute global transform ===
        orimesh = trimesh.Trimesh([])
        for name in namelist:
            oripart = load_obj_geometry_fast(os.path.join(objpath, name))
            orimesh = trimesh.util.concatenate([orimesh, oripart])
        orimesh.merge_vertices()

        bbox_max = np.array(orimesh.vertices).max(0)
        bbox_min = np.array(orimesh.vertices).min(0)
        scale = 1 / max(bbox_max - bbox_min)
        offset = -(bbox_min + bbox_max) / 2
        movtran(orimesh, offset, scale)
        M_scale = trimesh.transformations.scale_matrix(scale)
        M_trans = trimesh.transformations.translation_matrix(offset)
        M_rot = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        M = M_rot @ M_trans @ M_scale

        def transfer(point):
            ph = np.append(point, 1.0)
            return (M @ ph)[:3]

        # Refine joint coordinates (same as original)
        needrefineb, needrefinec, needrefined, needrefinecb = {}, {}, {}, {}
        if len(jsondata['group_info']) > 1:
            for groupind in range(1, len(jsondata['group_info'])):
                gi = jsondata['group_info'][str(groupind)]
                if gi[-1] == 'B':
                    needrefineb[str(groupind)] = gi[-2][:3]
                if gi[-1] == 'C':
                    needrefinec[str(groupind)] = gi[-2][:6]
                if gi[-1] == 'D':
                    needrefined[str(groupind)] = gi[-2][3:6]
                if gi[-1] == 'CB':
                    needrefinecb[str(groupind)] = gi[-2][:6] + gi[-2][8:11]

        for groupind in needrefineb:
            point2 = needrefineb[groupind][:3]
            jsondata['group_info'][groupind][-2][:3] = transfer(point2) - transfer([0,0,0])
        for groupind in needrefinec:
            point2 = needrefinec[groupind][:3]
            point3 = needrefinec[groupind][3:6]
            jsondata['group_info'][groupind][-2][:3] = transfer(point2) - transfer([0,0,0])
            jsondata['group_info'][groupind][-2][3:6] = transfer(point3)
        for groupind in needrefined:
            point1 = needrefined[groupind][:3]
            jsondata['group_info'][groupind][-2][3:6] = transfer(point1)
        for groupind in needrefinecb:
            point2 = needrefinecb[groupind][:3]
            point3 = needrefinecb[groupind][3:6]
            point4 = needrefinecb[groupind][6:]
            jsondata['group_info'][groupind][-2][:3] = transfer(point2) - transfer([0,0,0])
            jsondata['group_info'][groupind][-2][3:6] = transfer(point3)
            jsondata['group_info'][groupind][-2][8:11] = transfer(point4) - transfer([0,0,0])

        with open(os.path.join(tmpdir, 'finaljson', index + '.json'), 'w') as file:
            json.dump(jsondata, file, indent=4, ensure_ascii=False)

        # === Parallel: voxelize all parts ===
        num_parts = len(jsondata['parts'])
        part_args = []
        for nameind in range(num_parts):
            obj_names = jsondata['parts'][nameind]['obj']
            part_args.append((nameind, obj_names, objpath, offset, scale, tmpdir, index))

        actual_workers = min(num_workers, num_parts)
        logger.info(f'{index}: {num_parts} parts, using {actual_workers} workers')

        with Pool(actual_workers) as pool:
            all_results = pool.map(process_part, part_args)

        # Merge results per resolution
        for res in [16, 32, 64]:
            savepath = os.path.join(tmpdir, 'partseg', index, str(res))
            allind = np.empty((0, 3))
            alldict_vert = {}
            for result in all_results:
                for (ni, r), (indices, vertices) in result.items():
                    if r == res:
                        alldict_vert[str(ni)] = vertices
                        allind = np.concatenate([allind, indices], 0)

            with open(os.path.join(savepath, 'alldict_vert.pkl'), 'wb') as f:
                pickle.dump(alldict_vert, f)
            np.save(os.path.join(savepath, 'allind.npy'), allind)
            logger.info(f'{index} res={res}: {allind.shape}')

        logger.info(f'done: {index}')
