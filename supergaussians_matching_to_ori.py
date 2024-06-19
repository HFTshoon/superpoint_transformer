import os
import sys
from tqdm import tqdm

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

import torch
from src.datasets.s3dis import CLASS_NAMES, CLASS_COLORS, STUFF_CLASSES
from src.datasets.s3dis import S3DIS_NUM_CLASSES as NUM_CLASSES
from src.transforms import *

from plyfile import PlyData, PlyElement
import numpy as np
from colorhash import ColorHash

scene_name_list = [
	"bicycle",
	"room",
	"stump"
]

# scene_name = scene_name_list[5]
for scene_name in scene_name_list:
    print("="*50)
    print(scene_name)
    nag = torch.load(f"/workspace/superpoint_transformer/supergaussians/{scene_name}.pt")
    pt = PlyData.read(f"/workspace/gaussian-splatting/output/{scene_name}_4/point_cloud/iteration_30000/point_cloud.ply")
    change_pt = PlyData.read(f"/workspace/gaussian-splatting/output/{scene_name}_3/point_cloud/iteration_30000/point_cloud.ply")
    input_pt = PlyData.read(f"/workspace/gaussian-splatting/output/{scene_name}_4/input.ply")
    output_path = f"/workspace/superpoint_transformer/supergaussians/{scene_name}.ply"

    input_x = input_pt["vertex"]["x"]
    input_y = input_pt["vertex"]["y"]
    input_z = input_pt["vertex"]["z"]
    output_x = pt["vertex"]["x"]
    output_y = pt["vertex"]["y"]
    output_z = pt["vertex"]["z"]

    input_xyz = np.stack((input_x, input_y, input_z), axis=1)
    output_xyz = np.stack((output_x, output_y, output_z), axis=1)

    match_xyz = []
    for cur_xyz in tqdm(input_xyz):
        distance = np.linalg.norm(output_xyz - cur_xyz, axis=1)
        nearest_idx = np.argmin(distance)
        match_xyz.append(nearest_idx)
    # print(match_xyz)

    diff_cnt = 0
    for a, b in enumerate(match_xyz):
        if a!=b:
            diff_cnt += 1
    print(diff_cnt)

    pt_num_points = int(np.shape(input_pt["vertex"]["x"])[0])
    print(f"Num of Points(Input): {pt_num_points}")

    num_points = int(nag[0].sub.num_points)
    print(f"Num of Points(NAG): {num_points}")
    for i_level, data in enumerate(nag):
        print(f"Level-{i_level}:{data}\n")

    level_dict = []
    for i_level, data in enumerate(nag):
        level_dict.append({})
        for cluster_idx in tqdm(range(len(nag[i_level].sub))):
            cluster = nag[i_level].sub[cluster_idx]
            for point in cluster.points:
                level_dict[i_level][int(point)] = cluster_idx
                
    colors = []
    C = 0.28209479177387814
    for i in tqdm(range(num_points)):
        r,g,b = ColorHash(i).rgb
        colors.append(((r/255.0 - 0.5)/C, (g/255.0 - 0.5)/C, (b/255.0 - 0.5)/C))
            
    cluster_dict = {}
    color_dict = {}
    for point_idx in tqdm(range(num_points)):
        level_0 = level_dict[0][point_idx]
        level_1 = level_dict[1][level_0]
        level_2 = level_dict[2][level_1]
        level_3 = level_dict[3][level_2]
        cluster_dict[point_idx] = [level_0, level_1, level_2, level_3]
        color_dict[point_idx] = [colors[level_0], colors[level_1], colors[level_2], colors[level_3]]
    print(cluster_dict[0])
    print(color_dict[0])
        
    attributes_names = [p for p in pt['vertex'].properties]
    f_dc_names = [p.name for p in attributes_names if p.name.startswith("f_dc_")]
    f_dc_names = sorted(f_dc_names, key = lambda x: int(x.split('_')[-1]))
    f_rest_names = [p.name for p in attributes_names if p.name.startswith("f_rest_")]
    f_rest_names = sorted(f_rest_names, key = lambda x: int(x.split('_')[-1]))
    scales_names = [p.name for p in attributes_names if p.name.startswith("scale_")]
    scales_names = sorted(scales_names, key = lambda x: int(x.split('_')[-1]))
    rot_names = [p.name for p in attributes_names if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))

    for name in f_rest_names:
        pt['vertex'][name] = np.zeros_like(pt['vertex'][name])

    x = change_pt['vertex']['x']
    y = change_pt['vertex']['y']
    z = change_pt['vertex']['z']
    xyz = np.stack((x, y, z), axis=1)
    normals = np.zeros_like(xyz)

    f_rest = np.zeros((pt_num_points, len(f_rest_names)))
        
    opacities = change_pt['vertex']['opacity'].reshape(-1, 1)

    scale = np.zeros((pt_num_points, len(scales_names)))
    for name in scales_names:
        scale[:, int(name.split('_')[-1])] = change_pt['vertex'][name]
        
    rotation = np.zeros((pt_num_points, len(rot_names)))
    for name in rot_names:
        rotation[:, int(name.split('_')[-1])] = change_pt['vertex'][name]

    for level in range(4):
        f_dc = np.zeros((pt_num_points, len(f_dc_names)))
        for i, name in enumerate(f_dc_names):
            new_dc = np.zeros((pt_num_points))
            for j in range(pt_num_points):
                new_dc[j] = color_dict[match_xyz[j]][level][i]
            f_dc[:, i] = new_dc
        
        dtype_full = [(attribute.name, 'f4') for attribute in attributes_names]
        elements = np.empty(pt_num_points, dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path.replace(".ply", f"_level_{level}.ply"))