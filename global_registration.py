import time
import os
import sys

import pyntcloud
import yaml
import copy

from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.pipelines.registration import Feature, RegistrationResult
from typing import NamedTuple, List, Tuple, Optional
from tqdm import tqdm
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt

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


def get_vertex_label(s: str, prefix: str, start=True) -> str:
    if start:
        res = s[:s.find('_' + prefix)]
    else:
        res = s[s.find('_' + prefix) + 1:]
    res = res[len(prefix):]
    if res.find('_') >= 0:
        res = res[:res.find('_')]
    return res


def transform_to_global_coordinate_system(test_prefix: str, error_thr: float, pcds_path: str, gt_path: str):
    df_tns = pd.read_csv('data/debug/transformations.csv')
    df = pd.read_csv('data/debug/test_results.csv')
    df = df[(df['version'] == 17) & (df.index > 1870) & (df['testname'].apply(lambda s: s.startswith(test_prefix)))]
    # df['success'] = (df['overlap_rmse'] < error_thr) & (df['t_err'] < error_thr)
    df['success'] = df['inliers'] / df['correspondences'] > 0.05
    df['ratio'] = df['correspondences'] / df['inliers']
    df = df[df['success']]
    vertices = set()
    G = nx.Graph()
    edge_tns = {}

    # for testname, overlap_error in df[['testname', 'overlap_rmse']].values:
    for testname, overlap_error in df[['testname', 'ratio']].values:
        v1 = get_vertex_label(testname, test_prefix)
        v2 = get_vertex_label(testname, test_prefix, False)
        transformation = df_tns[df_tns['reading'].apply(lambda s: test_prefix + v1 in s and test_prefix + v2 in s and 'gt' not in s)].values.flatten()[1:].reshape((4, 4)).astype(float)
        edge_tns[v1 + '_' + v2] = transformation
        edge_tns[v2 + '_' + v1] = np.linalg.inv(transformation)
        G.add_edge(v1, v2, weight=overlap_error)
        vertices.add(v1)
        vertices.add(v2)
    adj_mtx = np.full((len(vertices), len(vertices)), float('inf'))
    inv_dict = {}
    vs_dict = {}
    for i, v in enumerate(sorted(list(vertices))):
        vs_dict[v] = i
        inv_dict[i] = v
    # for testname, overlap_error in df[['testname', 'overlap_rmse']].values:
    for testname, overlap_error in df[['testname', 'ratio']].values:
        v1 = get_vertex_label(testname, test_prefix)
        v2 = get_vertex_label(testname, test_prefix, False)
        adj_mtx[vs_dict[v1]][vs_dict[v2]] = overlap_error
        adj_mtx[vs_dict[v2]][vs_dict[v1]] = overlap_error
    np.fill_diagonal(adj_mtx, 0)
    for k in range(len(vertices)):
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if adj_mtx[i][j] > adj_mtx[i][k] + adj_mtx[k][j]:
                    adj_mtx[i][j] = adj_mtx[i][k] + adj_mtx[k][j]
    independent_sets = [{0}]
    for i in range(1, len(vertices)):
        in_set = False
        for s in independent_sets:
            if adj_mtx[i, next(iter(s))] != float('inf'):
                s.add(i)
                in_set = True
                break
        if not in_set:
            independent_sets.append({i})
    for s in independent_sets:
        print('set:', end=' ')
        for j in s:
            print(inv_dict[j], end=' ')
        print()
    roots = []
    T = nx.minimum_spanning_tree(G)
    edges = defaultdict(list)
    for u, v in T.edges:
        edges[u].append(v)
        edges[v].append(u)
    transformations = {}
    for s in independent_sets:
        local_vs = sorted(list(s))
        local_adj_mtx = adj_mtx
        local_adj_mtx = local_adj_mtx[local_vs]
        local_adj_mtx = local_adj_mtx[:, local_vs]
        root = inv_dict[local_vs[np.argmax(np.sum(local_adj_mtx, axis=0))]]
        print('root:', root)
        roots.append(root)
        q = deque()
        visited = defaultdict(lambda: False)
        q.append(root)
        transformations[root] = np.eye(4)
        while len(q) > 0:
            v = q.popleft()
            visited[v] = True
            for u in edges[v]:
                if not visited[u]:
                    q.append(u)
                    transformations[u] = transformations[v] @ edge_tns[u + '_' + v]
    gt = pd.read_csv(gt_path)
    # pos2 = {}
    # for node in nx.nodes(T):
    #     gt_test = gt[gt['reading'].apply(lambda s: test_prefix + node in s)]
    #     pos2[node] = np.array([gt_test['gT03'].values[0], gt_test['gT13'].values[0]])
    edges, weights = zip(*nx.get_edge_attributes(T,'weight').items())
    weights /= np.median(weights)
    pos1 = nx.spring_layout(T, k=0.05, scale=3, seed=10)

    plt.figure(1, figsize=(20,12))
    nx.draw(T, pos1, edgelist=edges, edge_color=weights, with_labels=True, edge_cmap=plt.cm.Blues)
    plt.savefig(f'{test_prefix}forest.png')

    # plt.figure(2, figsize=(20,12))
    # nx.draw(T, pos2, edgelist=edges, edge_color=weights, with_labels=True, edge_cmap=plt.cm.Blues)
    # plt.savefig(f'{test_prefix}forest_xy.png')

    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply', os.listdir(pcds_path))))
    voxel_size = 0.05

    path = os.path.join(pcds_path, 'transformed_' + str(voxel_size))
    if not os.path.exists(path):
        os.mkdir(path)

    iter_pbar = tqdm(filenames)
    for i, filename in enumerate(iter_pbar):
        iter_pbar.set_description(f'Processing {filename}..')
        pcd = o3d.io.read_point_cloud(os.path.join(pcds_path, filename))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        v = filename[len(test_prefix):-4]
        if v.find('_') >= 0:
            v = v[:v.find('_')]
        if not v in transformations:
            continue
        pcd_down = pcd_down.transform(transformations[v])
        pcd_down.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        pyntcloud.PyntCloud.from_instance("open3d", pcd_down).to_file(os.path.join(path, filename))


def downsample_and_transform_point_clouds(with_transformation: bool = True):
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    dataset_name = os.path.basename(sys.argv[1])[:-5]
    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply', os.listdir(config['path']))))
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
        if with_transformation and filename in gt:
            pcd_down = pcd_down.transform(gt[filename])
        pcd_down.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        pyntcloud.PyntCloud.from_instance("open3d", pcd_down).to_file(os.path.join(path, filename))


if __name__ == '__main__':
    transform_to_global_coordinate_system(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4])
