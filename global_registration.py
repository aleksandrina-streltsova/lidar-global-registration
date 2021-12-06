import time
import os
import sys

import pyntcloud
import yaml
import copy

from open3d.cuda.pybind.geometry import PointCloud
from open3d.cuda.pybind.pipelines.registration import Feature, RegistrationResult
from typing import NamedTuple, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import open3d as o3d

RegistrationData = NamedTuple('RegistrationData', [
    ('pcd', PointCloud),
    ('pcd_fpfh', Optional[Feature]),
    ('filename', str)
])
GROUND_TRUTH_COLUMNS = ['reading', 'gT00', 'gT01', 'gT02', 'gT03', 'gT10', 'gT11', 'gT12', 'gT13', 'gT20', 'gT21',
                        'gT22', 'gT23', 'gT30', 'gT31', 'gT32', 'gT33']


def count_correct_correspondences(source: RegistrationData, target: RegistrationData,
                                  correspondence_set: np.ndarray, transformation_gt: np.ndarray,
                                  error_threshold: float):
    source_correspondences = np.asarray(copy.deepcopy(source.pcd).transform(transformation_gt).points)[
        correspondence_set[:, 0]]
    target_correspondences = np.asarray(target.pcd.points)[correspondence_set[:, 1]]
    errors = np.linalg.norm(source_correspondences - target_correspondences, axis=1)
    return np.count_nonzero(errors < error_threshold)


def get_transformation(csv_path: str, src_filename: str, tgt_filename: str):
    df = pd.read_csv(csv_path)
    gt = {}
    for _, row in df.iterrows():
        gt[row[0]] = np.array(list(map(float, row[1:].values))).reshape((4, 4))
    return np.linalg.inv(gt[tgt_filename]) @ gt[src_filename]


def preprocess_point_cloud(pcd: PointCloud, voxel_size: int, config) -> Tuple[PointCloud, Feature]:
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"    Pointcloud down sampled from {len(pcd.points)} points to {len(pcd_down.points)} points.")

    radius_normal = config['normal_radius_coef'] * voxel_size
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=radius_normal))

    radius_feature = config['feature_radius_coef'] * voxel_size
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamRadius(radius=radius_feature))
    return pcd_down, pcd_fpfh


def load_dataset(voxel_size: int, config, is_local: bool = False) -> List[RegistrationData]:
    dataset = []
    for i, filename in enumerate(sorted(os.listdir(config['path']))):
        print(f"::  Processing {filename}")
        pcd = o3d.io.read_point_cloud(os.path.join(config['path'], filename))
        if is_local:
            dataset.append(RegistrationData(pcd, None, filename))
        else:
            pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, config)
            dataset.append(RegistrationData(pcd_down, pcd_fpfh, filename))
            del pcd
    return dataset


def load_source_and_target(voxel_size: int, config, is_local: bool = False) -> Tuple[
    RegistrationData, RegistrationData]:
    dataset = []
    for i, key in enumerate(['source', 'target']):
        filepath = config[key]
        filename = os.path.basename(filepath)
        print(f"::  Processing {filename}")
        pcd = o3d.io.read_point_cloud(filepath)
        if is_local:
            dataset.append(RegistrationData(pcd, None, filename))
        else:
            pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, config)
            dataset.append(RegistrationData(pcd_down, pcd_fpfh, filename))
            del pcd
    return dataset[0], dataset[1]


def execute_global_registration(source: RegistrationData, target: RegistrationData,
                                voxel_size: int, config, testname: str) -> RegistrationResult:
    print(f"::  RANSAC global registration on downsampled point clouds: {source.filename} and {target.filename}.")
    start = time.time()
    distance_threshold = config['distance_thr_coef'] * voxel_size
    result: RegistrationResult = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source.pcd, target.pcd, source.pcd_fpfh, target.pcd_fpfh, config['reciprocal'],
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(config['iteration'], config['confidence']))
    correspondence_set = np.asarray(result.correspondence_set)
    print("    Global registration took: %.3f sec." % (time.time() - start))
    print(f"    fitness: {result.fitness}\n"
          f"    inlier_rmse: {result.inlier_rmse}\n"
          f"    inliers: {len(result.correspondence_set)}/{round(len(result.correspondence_set) / result.fitness)}")
    save_clouds(result.transformation, config, testname)
    if 'ground_truth' in config:
        transformation_gt = get_transformation(config['ground_truth'], source.filename, target.filename)
        print(
            f"    correct inliers: {count_correct_correspondences(source, target, np.asarray(result.correspondence_set), transformation_gt, distance_threshold)}\n")
        print(f"transformation: \n\n{result.transformation}\n")
        print(f"transformation (ground truth): \n\n{transformation_gt}\n")
        save_correspondence_distances(source, target, correspondence_set, transformation_gt, voxel_size, testname)
    return result


def execute_local_registration(source: RegistrationData, target: RegistrationData,
                               transformation: np.array, config) -> RegistrationResult:
    print(f"::  Apply point-to-point ICP: {source.filename} and {target.filename}.")
    start = time.time()

    result = o3d.pipelines.registration.registration_icp(
        source.pcd, target.pcd, config['icp_thr'], transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False))

    print("    Local registration took: %.3f sec." % (time.time() - start))
    print(f"    fitness: {result.fitness}\n"
          f"    inlier_rmse: {result.inlier_rmse}\n")
    print(f"transformation: \n\n{result.transformation}\n")
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
    source_correspondences = np.asarray(copy.deepcopy(source.pcd).transform(transformation_gt).points)[
        correspondence_set[:, 0]]
    target_correspondences = np.asarray(target.pcd.points)[correspondence_set[:, 1]]
    errors = np.linalg.norm(source_correspondences - target_correspondences, axis=1) / voxel_size
    df = pd.DataFrame(errors, columns=['distance'])
    df.to_csv(testname + '_distances.csv', index=False)


def run_global_registration_and_save_ply():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream)
    testname = os.path.basename(sys.argv[1])[:-5]
    dataset = load_dataset(config['voxel_size'], config)
    for i in range(len(dataset) - 1):
        source = dataset[i]
        target = dataset[i + 1]
        execute_global_registration(source, target, config['voxel_size'], config, testname)


def run_global_registration():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    testname = os.path.basename(sys.argv[1])[:-5]
    source, target = load_source_and_target(config['voxel_size'], config)
    execute_global_registration(source, target, config['voxel_size'], config, testname)


def run_global_and_local_registration():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    testname = os.path.basename(sys.argv[1])[:-5]
    source, target = load_source_and_target(config['voxel_size'], config)
    result = execute_global_registration(source, target, config['voxel_size'], config, testname)
    source, target = load_source_and_target(config['voxel_size'], config, is_local=True)
    execute_local_registration(source, target, result.transformation, config)


def estimate_and_save_ground_truth():
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    if os.path.exists(os.path.join(config['path'], 'ground_truth.csv')):
        return
    testname = os.path.basename(sys.argv[1])[:-5]
    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply' and f.startswith(testname), os.listdir(config['path']))))
    transformations = [np.eye(4)]
    voxel_size = config['voxel_size']

    print(f"::  Processing {filenames[0]}")
    pcd = o3d.io.read_point_cloud(os.path.join(config['path'], filenames[0]))
    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, config)
    source = RegistrationData(pcd_down, pcd_fpfh, filenames[0])
    source_local = RegistrationData(pcd, None, filenames[0])

    for i, filename in enumerate(filenames[1:]):
        print(f"::  Processing {filename}")
        pcd = o3d.io.read_point_cloud(os.path.join(config['path'], filename))
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, config)
        target = RegistrationData(pcd_down, pcd_fpfh, filename)
        target_local = RegistrationData(pcd, None, filename)

        result = execute_global_registration(source, target, voxel_size, config, testname)
        result = execute_local_registration(source_local, target_local, result.transformation, config)
        transformations.append(transformations[-1] @ np.linalg.inv(result.transformation))

        source = target
        source_local = target_local

    values = []
    for filename, transformation in zip(filenames, transformations):
        values.append([filename] + list(transformation.flatten()))
    df = pd.DataFrame(values, columns=GROUND_TRUTH_COLUMNS, index=None)
    df.to_csv(os.path.join(config['path'], 'ground_truth.csv'), index=False)


def downsample_and_transform_point_clouds(with_transformation: bool = True):
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    dataset_name = os.path.basename(sys.argv[1])[:-5]
    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply' and f.startswith(dataset_name), os.listdir(config['path']))))
    voxel_size = config['voxel_size']

    gt = {}
    if with_transformation:
        df = pd.read_csv(config['ground_truth'])
        for _, row in df.iterrows():
            gt[row[0]] = np.array(list(map(float, row[1:].values))).reshape((4, 4))

    path = os.path.join(config['path'], 'downsampled_' + str(voxel_size))
    if not os.path.exists(path):
        os.mkdir(path)

    iter_pbar = tqdm(filenames)
    for i, filename in enumerate(iter_pbar):
        iter_pbar.set_description(f'Processing {filename}..')
        pcd = o3d.io.read_point_cloud(os.path.join(config['path'], filename))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if with_transformation:
            pcd_down = pcd_down.transform(gt[filename])
        pcd_down.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        pyntcloud.PyntCloud.from_instance("open3d", pcd_down).to_file(os.path.join(path, filename))


if __name__ == '__main__':
    run_global_registration()
