import os

import click
import yaml
import pandas as pd
import numpy as np
import pye57
from scipy.spatial.transform import Rotation

from typing import List

from tqdm import tqdm
from pyntcloud import PyntCloud


ETH_GT_FILENAME = 'icpList.csv'
ETH_GT_COLUMN_PC = 'reading'
COMMON_GT_COLUMN_PC = ETH_GT_COLUMN_PC
COMMON_GT_FILENAME = 'ground_truth.csv'
GROUND_TRUTH_COLUMNS = ['reading', 'gT00', 'gT01', 'gT02', 'gT03', 'gT10', 'gT11', 'gT12', 'gT13', 'gT20', 'gT21',
                        'gT22', 'gT23', 'gT30', 'gT31', 'gT32', 'gT33']


@click.group()
def cli():
    pass


def _remove_suffix_starting_with_substring(s: str, sub: str) -> str:
    start = s.rfind(sub)
    if start != -1:
        return s[:start]
    return s


def _remove_extension(filename: str) -> str:
    return _remove_suffix_starting_with_substring(filename, '.')


def _remove_number(filename: str) -> str:
    return _remove_suffix_starting_with_substring(filename, '_')


def _get_test_name_eth(filename: str) -> str:
    return _remove_number(_remove_extension(filename))


def _get_test_name_stanford(conf_filename: str) -> str:
    return _remove_extension(conf_filename)


def _get_csv_column_names() -> List[str]:
    columns = [COMMON_GT_COLUMN_PC]
    for i in range(4):
        for j in range(4):
            columns.append(f'gT{i}{j}')
    return columns


def _parse_conf(conf_path: str):
    data_rows = []
    with open(conf_path, 'r') as conf:
        for line in conf.readlines():
            line_list = line.split()
            if len(line_list) < 2 or not line_list[1].endswith('.ply'):
                continue
            transformation = np.eye(4)
            translation = list(map(float, line_list[2:5]))
            rotation_quat = list(map(float, line_list[5:9]))
            rotation_mtx = np.linalg.inv(Rotation.from_quat(rotation_quat).as_matrix())
            transformation[:3, :3] = rotation_mtx
            transformation[:3, 3] = translation
            data_rows.append([line_list[1]] + transformation.flatten().tolist())
    gt_df = pd.DataFrame(data_rows, columns=_get_csv_column_names())
    return gt_df


@cli.command('eth_gt')
@click.argument('path', type=click.Path(exists=True, file_okay=False))
def parse_gt_eth(path):
    ply_names = sorted(list(map(lambda s: s[:s.find('.')], filter(lambda s: s.endswith('.ply'), os.listdir(path)))))
    with open(os.path.join(path, 'ground_truth.csv'), 'w+') as file:
        file.write(','.join(GROUND_TRUTH_COLUMNS) + '\n')
        file.write(ply_names[0] + '.ply,' + ','.join(map(str, np.eye(4).flatten())) + '\n')
        for i, ply_name in enumerate(ply_names[1:]):
            with open(os.path.join(path, 'groundtruth', f'{ply_name}-{ply_names[0]}.tfm'), 'r') as tn_file:
                file.write(ply_name + '.ply')
                for line in tn_file.readlines():
                    file.write(',' + ','.join(line.split()))
                file.write('\n')


@cli.command('stanford')
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False))
def stanford_to_common(input_dir, output_dir):
    conf_filenames = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.conf'):
            conf_filenames.append(file)
    if len(conf_filenames) == 0:
        print(f'No .conf file was found in f{input_dir}')
        return

    if output_dir is None:
        output_dir = _get_test_name_stanford(conf_filenames[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_df = pd.concat([_parse_conf(os.path.join(input_dir, conf_filename)) for conf_filename in conf_filenames])

    iter_pbar = tqdm(gt_df[COMMON_GT_COLUMN_PC])
    for filename in iter_pbar:
        iter_pbar.set_description(f'Copying {filename}..')
        cloud = PyntCloud.from_file(os.path.join(input_dir, filename))
        cloud.points = cloud.points.dropna()
        cloud.to_file(os.path.join(output_dir, filename))
    gt_df.to_csv(os.path.join(output_dir, COMMON_GT_FILENAME), index=False)


@cli.command('eth')
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False))
def eth_to_common(input_dir, output_dir):
    gt_df = pd.read_csv(os.path.join(input_dir, ETH_GT_FILENAME))

    if output_dir is None:
        output_dir = _get_test_name_eth(gt_df[ETH_GT_COLUMN_PC][0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iter_pbar = tqdm(gt_df[ETH_GT_COLUMN_PC])
    for filename in iter_pbar:
        iter_pbar.set_description(f'Processing {filename}..')
        cloud_df = pd.read_csv(os.path.join(input_dir, filename))
        cloud_df[['red', 'green', 'blue']] = 0.5
        cloud = PyntCloud(cloud_df.loc[:, ['x', 'y', 'z', 'red', 'green', 'blue']])
        cloud.to_file(os.path.join(output_dir, _remove_extension(filename) + '.ply'))

    gt_df[ETH_GT_COLUMN_PC] = gt_df[ETH_GT_COLUMN_PC].apply(lambda pc: _remove_extension(pc) + '.ply')
    gt_df.to_csv(os.path.join(output_dir, COMMON_GT_FILENAME), index=False)


@cli.command('other')
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
def other_to_common(input_dir):
    output_dir = input_dir
    for filename in sorted(os.listdir(input_dir)):
        if filename[-4:] == '.ply':
            cloud = PyntCloud.from_file(os.path.join(input_dir, filename))
            cloud.points = cloud.points.dropna()
            cloud.to_file(os.path.join(output_dir, filename))


@cli.command('e57')
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
@click.option('--indices', '-i', multiple=True, type=int)
def e57_to_common(input_dir, indices):
    output_dir = input_dir
    for f in tqdm(os.listdir(input_dir)):
        if f[-4:] != '.e57':
            continue
        e57_path = os.path.join(input_dir, f)
        e57 = pye57.E57(str(e57_path))
        input_dir = os.path.dirname(e57_path)
        testname = os.path.basename(e57_path)[:-4]
        gt_path = os.path.join(input_dir, COMMON_GT_FILENAME)
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)
        else:
            df = pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)
        rng = range(e57.scan_count) if len(indices) == 0 else indices
        for i in rng:
            scan = e57.read_scan(i, transform=False)
            header = e57.get_header(i)
            data = np.stack((scan['cartesianX'], scan['cartesianY'], scan['cartesianZ']), axis=1)
            cloud_df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            cloud = PyntCloud(cloud_df)
            filename = f'{testname}_{i}.ply'
            cloud.to_file(os.path.join(output_dir, filename))
            transformation_gt = np.eye(4)
            if header.has_pose():
                transformation_gt[:3, :3] = header.rotation_matrix
                transformation_gt[:3, 3] = header.translation
            df.drop(df[df['reading'] == filename].index, inplace=True)
            df = df.append(pd.DataFrame([[filename] + transformation_gt.flatten().tolist()], columns=df.columns))
        df.to_csv(gt_path, index=False)


@cli.command('las')
@click.argument('las-path', type=click.Path(exists=True, file_okay=False))
def las_to_common(las_path):
    filenames = [filename for filename in os.listdir(las_path) if filename.endswith('.las')]
    for filename in tqdm(filenames):
        pcd: PyntCloud = PyntCloud.from_file(os.path.join(las_path, filename))
        pcd_name = filename[:-4]
        pcd.to_file(os.path.join(las_path, f'{pcd_name}.ply'))


def transform_and_save(load_from: str, save_to: str, transformation: np.ndarray):
    pynt_cloud = PyntCloud.from_file(load_from)
    with_normals = 'nx' in pynt_cloud.points.columns
    columns = ['x', 'y', 'z'] + (['nx', 'ny', 'nz'] if with_normals else [])
    data = pynt_cloud.points[columns].values
    data[:, :3] = (transformation[:3, :3] @ data[:, :3].T).T + transformation[:3, 3]
    if with_normals:
        data[:, 3:] = (transformation[:3, :3] @ data[:, 3:].T).T
    pynt_cloud.points = pd.DataFrame(data, columns=columns, index=None)
    pynt_cloud.to_file(save_to)


@cli.command('perturb')
@click.argument('config-path', type=click.File())
@click.option('--with-translation/--without-translation', default=False)
@click.option('--with-rotation/--without-rotation', default=True)
def generate_and_save_random_perturbation(config_path, with_translation: bool = False, with_rotation: bool = True):
    config = yaml.load(config_path, Loader=yaml.Loader)
    if with_rotation:
        rmat = Rotation.from_euler('zyx', angles=[180 * np.random.rand(), 0, 0], degrees=True).as_matrix()
    else:
        rmat = np.eye(3)
    if with_translation:
        tvec = np.random.rand(3) * 10
    else:
        tvec = np.zeros(3)
    transformation = np.eye(4)
    transformation[:3, :3] = rmat
    transformation[:3, 3] = tvec
    dirpath = os.path.dirname(config['transform'])
    suffix = ("_r" if with_rotation else "") + ("_t" if with_translation else "")
    filename = os.path.basename(config['transform'])[:-4] + f'_transformed{suffix}.ply'
    transform_and_save(config['transform'], os.path.join(dirpath, filename), transformation)
    df = pd.read_csv(config['ground_truth'])
    df.drop(df[df['reading'] == filename].index, inplace=True)
    transformation_gt = df[df['reading'] == os.path.basename(config['transform'])].values[0, 1:].astype(float).reshape((4, 4))
    df = df.append(pd.DataFrame([[filename] + (transformation_gt @ np.linalg.inv(transformation)).flatten().tolist()], columns=df.columns))
    df.to_csv(config['ground_truth'], index=False)


@cli.command('transform')
@click.argument('config-path', type=click.Path(dir_okay=False, exists=True))
@click.option('--current', type=click.Choice(['local', 'global'], case_sensitive=False), default='global')
def transform(config_path, current):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    dirpath = os.path.dirname(str(config_path))
    dataset_name = os.path.basename(str(config_path))[:-5]
    filenames = list(sorted(filter(lambda f: f[-4:] == '.ply' and f.startswith(dataset_name), os.listdir(dirpath))))

    gt = {}
    df = pd.read_csv(config['ground_truth'])
    for _, row in df.iterrows():
        gt[row[0]] = np.array(list(map(float, row[1:].values))).reshape((4, 4))

    for filename in tqdm(filenames):
        if current == "local":
            transformation = gt[filename]
        else:
            transformation = np.linalg.inv(gt[filename])
        filepath = os.path.join(dirpath, filename)
        transform_and_save(filepath, filepath, transformation)


if __name__ == '__main__':
    cli()
