import os
import click
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

from typing import List

from tqdm import tqdm
from pyntcloud import PyntCloud


ETH_GT_FILENAME = 'icpList.csv'
ETH_GT_COLUMN_PC = 'reading'
COMMON_GT_COLUMN_PC = ETH_GT_COLUMN_PC
COMMON_GT_FILENAME = 'ground_truth.csv'


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


if __name__ == '__main__':
    cli()
