import pandas as pd
import os
import click

CONFIG_PATH = 'data/tests.yaml'
OVERLAPPING_PATH = 'data/kizhi/downsampled/overlapping.csv'
DIFFICULTY_LEVELS_PATH = 'data/test_levels.csv'
# DIRPATHS = ['data/kizhi', 'data/office', 'data/arch', 'data/trees']
DIRPATHS = ['data/1-SubwayStation', 'data/2-HighSpeedRailway', 'data/3-Mountain',
            'data/5-Park', 'data/6-Campus', 'data/7-Residence', 'data/8-RiverBank',
            'data/9-HeritageBuilding', 'data/10-UndergroundExcavation', 'data/11-Tunnel']
PARAMETERS = '''
        voxel_size: 0.04
        keypoint: iss
        density: 0.005
        downsample: true
        matching: lr
        metric: correspondences
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
    with open(CONFIG_PATH, 'a') as file:
        file.write('tests:\n')
        for f1, f2, gror_iss_coef in df[['source', 'target', 'gror_iss_coef']].values:
            for dirpath in DIRPATHS:
                if os.path.exists(os.path.join(dirpath, f1)):
                    file.write('    - test:')
                    file.write(PARAMETERS)
                    file.write(f'        gror_iss_coef: {gror_iss_coef}\n')
                    file.write(f'        ground_truth: ' + os.path.join(dirpath, 'ground_truth.csv') + '\n')
                    file.write(f'        source: {os.path.join(dirpath, f1)}\n')
                    file.write(f'        target: {os.path.join(dirpath, f2)}\n')
    # filenames = list(sorted(filter(lambda s: s.endswith('.ply'), os.listdir(DIRPATH))))
    # with open(CONFIG_PATH, 'a') as file:
    #     for i, f1 in enumerate(filenames):
    #         for f2 in filenames[:i]:
    #             file.write('    - test:')
    #             file.write(PARAMETERS)
    #             file.write(f'        source: {os.path.join(DIRPATH, f1)}\n')
    #             file.write(f'        target: {os.path.join(DIRPATH, f2)}\n')


if __name__ == '__main__':
    generate_config()
