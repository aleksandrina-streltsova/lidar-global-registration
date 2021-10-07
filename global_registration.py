import time
import os
import sys
import yaml
from typing import NamedTuple, List, Tuple

import numpy as np
import pandas as pd
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud, AxisAlignedBoundingBox
from open3d.cpu.pybind.pipelines.registration import Feature, RegistrationResult

RegistrationData = NamedTuple('RegistrationData', [
    ('pcd_down', PointCloud),
    ('pcd_fpfh', Feature),
    ('filename', str)
])


def get_transformation(csv_path: str, src_filename: str, tgt_filename: str):
    df = pd.read_csv(csv_path)
    gt = {}
    for _, row in df.iterrows():
        gt[row[0]] = np.array(list(map(float, row[1:].values))).reshape((4, 4))
    return gt[tgt_filename] @ np.linalg.inv(gt[src_filename])


def preprocess_point_cloud(pcd: PointCloud, voxel_size: int, model_size: float, config) -> Tuple[PointCloud, Feature]:
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"    Pointcloud down sampled from {len(pcd.points)} points to {len(pcd_down.points)} points.")

    radius_normal = config['normal_radius'] * model_size
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = config['feature_radius'] * model_size
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def load_dataset(voxel_size: int, config) -> List[RegistrationData]:
    dataset = []
    for i, filename in enumerate(sorted(os.listdir(config['path']))):
        print(f"::  Processing {filename}")
        pcd = o3d.io.read_point_cloud(os.path.join(config['path'], filename))
        if i == 0:
            bbox = AxisAlignedBoundingBox().create_from_points(pcd.points)
            model_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, model_size, config)
        dataset.append(RegistrationData(pcd_down, pcd_fpfh, filename))
        del pcd
    return dataset


def load_source_and_target(voxel_size: int, config) -> Tuple[RegistrationData, RegistrationData]:
    dataset = []
    for i, key in enumerate(['source', 'target']):
        filepath = config[key]
        filename = os.path.basename(filepath)
        print(f"::  Processing {filename}")
        pcd = o3d.io.read_point_cloud(filepath)
        if i == 0:
            bbox = AxisAlignedBoundingBox().create_from_points(pcd.points)
            model_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, model_size, config)
        dataset.append(RegistrationData(pcd_down, pcd_fpfh, filename))
        del pcd
    return dataset[0], dataset[1]


def execute_global_registration(source: RegistrationData, target: RegistrationData,
                                voxel_size: int, config) -> RegistrationResult:
    print(f"::  RANSAC global registration on downsampled point clouds: {source.filename} and {target.filename}.")
    start = time.time()
    distance_threshold = config['distance_thr'] * voxel_size
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source.pcd_down, target.pcd_down, source.pcd_fpfh, target.pcd_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(config['iteration'], config['confidence']))
    print("    Global registration took: %.3f sec." % (time.time() - start))
    print(f"    fitness: {result.fitness}\n"
          f"    inlier_rmse: {result.inlier_rmse}\n"
          f"    inliers: {len(result.correspondence_set)}/{round(len(result.correspondence_set) / result.fitness)}")
    print(result.transformation)
    return result


def save_clouds(source_filename: str, target_filename: str, transformation: np.ndarray, config):
    source_pcd = o3d.io.read_point_cloud(os.path.join(config['path'], source_filename))
    target_pcd = o3d.io.read_point_cloud(os.path.join(config['path'], target_filename))
    source_pcd.transform(transformation)
    source_pcd += target_pcd
    o3d.io.write_point_cloud(f"{source_filename[:-4]}_{target_filename[:-4]}.ply", source_pcd,
                             compressed=True, print_progress=True)
    del source_pcd, target_pcd


def run_global_registration_and_save_ply():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream)
    dataset = load_dataset(config['voxel_size'], config)
    for i in range(len(dataset) - 1):
        source = dataset[i]
        target = dataset[i + 1]
        result = execute_global_registration(source, target, config['voxel_size'], config)
        save_clouds(source.filename, target.filename, result.transformation, config)


def run_global_registration():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    source, target = load_source_and_target(config['voxel_size'], config)
    execute_global_registration(source, target, config['voxel_size'], config)
    print(get_transformation(config['ground_truth'], source.filename, target.filename))


if __name__ == '__main__':
    run_global_registration()
