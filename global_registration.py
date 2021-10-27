import time
import os
import sys
import yaml
import copy
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


def count_correct_correspondences(source: RegistrationData, target: RegistrationData,
                                  correspondence_set: np.ndarray, transformation_gt: np.ndarray,
                                  error_threshold: float):
    source_correspondences = np.asarray(copy.deepcopy(source.pcd_down).transform(transformation_gt).points)[correspondence_set[:, 0]]
    target_correspondences = np.asarray(target.pcd_down.points)[correspondence_set[:, 1]]
    errors = np.linalg.norm(source_correspondences - target_correspondences, axis=1)
    return np.count_nonzero(errors < error_threshold)


def get_transformation(csv_path: str, src_filename: str, tgt_filename: str):
    df = pd.read_csv(csv_path)
    gt = {}
    for _, row in df.iterrows():
        gt[row[0]] = np.array(list(map(float, row[1:].values))).reshape((4, 4))
    return np.linalg.inv(gt[tgt_filename]) @ gt[src_filename]


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
                                voxel_size: int, config, testname: str) -> RegistrationResult:
    print(f"::  RANSAC global registration on downsampled point clouds: {source.filename} and {target.filename}.")
    start = time.time()
    distance_threshold = config['distance_thr'] * voxel_size
    result: RegistrationResult = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source.pcd_down, target.pcd_down, source.pcd_fpfh, target.pcd_fpfh, config['reciprocal'],
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(config['iteration'], config['confidence']))
    transformation_gt = get_transformation(config['ground_truth'], source.filename, target.filename)
    correspondence_set = np.asarray(result.correspondence_set)
    print("    Global registration took: %.3f sec." % (time.time() - start))
    print(f"    fitness: {result.fitness}\n"
          f"    inlier_rmse: {result.inlier_rmse}\n"
          f"    inliers: {len(result.correspondence_set)}/{round(len(result.correspondence_set) / result.fitness)}\n"
          f"    correct inliers: {count_correct_correspondences(source, target, np.asarray(result.correspondence_set), transformation_gt, float(config['error_thr']))}\n")
    print(f"transformation: \n\n{result.transformation}\n")
    print(f"transformation (ground truth): \n\n{transformation_gt}\n")
    save_clouds(result.transformation, config, testname)
    save_correspondence_distances(source, target, correspondence_set, transformation_gt, voxel_size, testname)
    return result


def save_clouds(transformation: np.ndarray, config, testname: str):
    source_pcd = o3d.io.read_point_cloud(config['source'])
    target_pcd = o3d.io.read_point_cloud(config['target'])
    source_pcd.paint_uniform_color([1, 0.706, 0])
    target_pcd.paint_uniform_color([0, 0.651, 0.929])
    source_pcd.transform(transformation)
    source_pcd += target_pcd
    o3d.io.write_point_cloud(testname + "_aligned_open3d.ply", source_pcd,
                             compressed=True, print_progress=True)
    del source_pcd, target_pcd


def save_correspondence_distances(source: RegistrationData, target: RegistrationData, correspondence_set: np.ndarray,
                                  transformation_gt: np.ndarray, voxel_size: float, testname: str):
    source_correspondences = np.asarray(copy.deepcopy(source.pcd_down).transform(transformation_gt).points)[correspondence_set[:, 0]]
    target_correspondences = np.asarray(target.pcd_down.points)[correspondence_set[:, 1]]
    errors = np.linalg.norm(source_correspondences - target_correspondences, axis=1) / voxel_size
    df = pd.DataFrame(errors, columns=['distance'])
    df.to_csv(testname + '_distances.csv', index=False)


def run_global_registration_and_save_ply():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream)
    dataset = load_dataset(config['voxel_size'], config)
    for i in range(len(dataset) - 1):
        source = dataset[i]
        target = dataset[i + 1]
        execute_global_registration(source, target, config['voxel_size'], config)


def run_global_registration():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    testname = os.path.basename(sys.argv[1])[:-5]
    source, target = load_source_and_target(config['voxel_size'], config)
    execute_global_registration(source, target, config['voxel_size'], config, testname)


if __name__ == '__main__':
    run_global_registration()
