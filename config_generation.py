import pandas as pd
import os
import click

CONFIG_PATH = 'data/kizhi/kizhi_difficult.yaml'
OVERLAPPING_PATH = 'data/kizhi/downsampled/overlapping.csv'
DIFFICULTY_LEVELS_PATH = 'data/test_levels.csv'
DIRPATH = 'data/kizhi/downsampled_0.02'
PARAMETERS = '''
        voxel_size: 0.02
        downsample: false
        iteration: 10000000
        normal_radius_coef: 4
        feature_radius_coef: 15
        distance_thr_coef: 1.5
        edge_thr: 0.95
        confidence: 0.999
        n_samples: 3
        reciprocal: true
        randomness: 1
        inlier_fraction: 0.1
        ground_truth: data/kizhi/ground_truth.csv
        descriptor: [fpfh, shot, rops]
        bf: true
        debug: true
'''
MIN_OVERLAP = 0.2


# def generate_config():
#     df = pd.read_csv(OVERLAPPING_PATH, index_col='reading')
#     with open(CONFIG_PATH, 'w') as file:
#         file.write('tests:\n')
#         overlaps = df.values
#         filenames = df.columns
#         for i, f1 in enumerate(filenames):
#             for j, f2 in enumerate(filenames[:i]):
#                 if overlaps[i][j] > MIN_OVERLAP:
#                     file.write('    - test:')
#                     file.write(PARAMETERS)
#                     file.write(f'        source: {os.path.join(DIRPATH, f1)}\n')
#                     file.write(f'        target: {os.path.join(DIRPATH, f2)}\n')

@click.command()
@click.option('-l', '--level', default=2)
def generate_config(level):
    df = pd.read_csv(DIFFICULTY_LEVELS_PATH)
    df = df[df['level'] == level]
    with open(CONFIG_PATH, 'w') as file:
        file.write('tests:\n')
        testnames = df['testname'].values
        for testname in testnames:
            f1 = testname[:testname.rfind('kig') - 1] + '.ply'
            f2 = testname[testname.rfind('kig'):] + '.ply'
            file.write('    - test:')
            file.write(PARAMETERS)
            file.write(f'        source: {os.path.join(DIRPATH, f1)}\n')
            file.write(f'        target: {os.path.join(DIRPATH, f2)}\n')


if __name__ == '__main__':
    generate_config()
