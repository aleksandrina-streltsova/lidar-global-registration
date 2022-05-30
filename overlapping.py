import sys
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from typing import List
from open3d.cuda.pybind.geometry import PointCloud


def calculate_overlap(pcd1: PointCloud, pcd2: PointCloud, voxel_size: float) -> float:
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
    count1 = 0
    count2 = 0
    for p in pcd1.points:
        [_, _, dist2] = pcd2_tree.search_knn_vector_3d(p, 1)
        if dist2[0] ** 0.5 < 2 * voxel_size:
            count1 += 1
    for p in pcd2.points:
        [_, _, dist2] = pcd1_tree.search_knn_vector_3d(p, 1)
        if dist2[0] ** 0.5 < 2 * voxel_size:
            count2 += 1
    return max(count1 / len(pcd1.points), count2 / len(pcd2.points))


def find_best_pairs(voxel_size: float, dirpath: str, filenames: List[str]):
    max_overlap = 0
    indices = (-1, -1)
    overlapping_matrix = np.ones((len(filenames), len(filenames)))
    for i1, f_i1 in enumerate(tqdm(filenames)):
        pcd_i1 = o3d.io.read_point_cloud(os.path.join(dirpath, f_i1))
        for j1, f_j1 in enumerate(filenames[:i1]):
            pcd_j1 = o3d.io.read_point_cloud(os.path.join(dirpath, f_j1))
            overlap = calculate_overlap(pcd_i1, pcd_j1, voxel_size)
            overlapping_matrix[i1, j1] = overlap
            overlapping_matrix[j1, i1] = overlap
            if overlap > max_overlap:
                max_overlap = overlap
                indices = (i1, j1)
    df = pd.DataFrame(data=overlapping_matrix, columns=filenames, index=pd.Index(filenames, name='reading'))
    df.to_csv(os.path.join(dirpath, 'overlapping.csv'))

    i1, j1 = indices
    f_i1, f_j1 = filenames[i1], filenames[j1]
    pcd_i1 = o3d.io.read_point_cloud(os.path.join(dirpath, filenames[i1]))
    pcd_j1 = o3d.io.read_point_cloud(os.path.join(dirpath, filenames[j1]))
    filenames = filenames[:i1] + filenames[i1 + 1:]
    filenames = filenames[:j1] + filenames[j1 + 1:]
    overlapping_matrix = np.zeros((len(filenames), len(filenames)))

    for i2, f_i2 in enumerate(tqdm(filenames)):
        pcd_i2: PointCloud = o3d.io.read_point_cloud(os.path.join(dirpath, filenames[i2]))
        pcd_i2 += pcd_i1
        for j2, f_j2 in enumerate(filenames):
            if j2 == i2:
                continue
            pcd_j2: PointCloud = o3d.io.read_point_cloud(os.path.join(dirpath, filenames[j2]))
            pcd_j2 += pcd_j1
            overlap = calculate_overlap(pcd_i2, pcd_j2, voxel_size)
            overlapping_matrix[i2, j2] = overlap
    df = pd.DataFrame(data=overlapping_matrix, columns=filenames, index=pd.Index(filenames, name='reading'))
    df.to_csv(os.path.join(dirpath, 'overlapping_pairs.csv'))
    i2, j2 = np.unravel_index(np.argmax(overlapping_matrix, axis=None), overlapping_matrix.shape)
    f_i2, f_j2 = filenames[i2], filenames[j2]
    print(f'best pairs: [{f_i1}, {f_i2}] [{f_j1}, {f_j2}]')


def main():
    if len(sys.argv) != 3:
        print("Syntax is: dir_path voxel_size!")
        exit(1)
    dirpath = sys.argv[1]
    if not os.path.exists(dirpath):
        print(f'Directory {dirpath} doesn\'t exist!')
        exit(1)
    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply', os.listdir(dirpath))))
    voxel_size = float(sys.argv[2])
    find_best_pairs(voxel_size, dirpath, filenames)
    # overlaps = np.ones((len(filenames), len(filenames)))
    # for i, f1 in enumerate(tqdm(filenames)):
    #     pcd1 = o3d.io.read_point_cloud(os.path.join(dirpath, f1))
    #     for j, f2 in enumerate(filenames[:i]):
    #         pcd2 = o3d.io.read_point_cloud(os.path.join(dirpath, f2))
    #         overlap = calculate_overlap(pcd1, pcd2, voxel_size)
    #         overlaps[i][j] = overlap
    #         overlaps[j][i] = overlap
    # df = pd.DataFrame(data=overlaps, columns=filenames, index=pd.Index(filenames, name='reading'))
    # df.to_csv(os.path.join(dirpath, 'overlapping.csv'))


if __name__ == '__main__':
    main()
