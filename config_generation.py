import pandas as pd
import os
import click

DIFFICULTY_LEVELS_PATH = 'data/test_levels.csv'

DIRPATHS = ['data/kizhi', 'data/office', 'data/arch', 'data/trees',
            'data/1-SubwayStation', 'data/2-HighSpeedRailway', 'data/3-Mountain',
            'data/5-Park', 'data/6-Campus', 'data/7-Residence', 'data/8-RiverBank',
            'data/9-HeritageBuilding', 'data/10-UndergroundExcavation', 'data/11-Tunnel',
            '/media/agisoft/nas2_dataset/processing/polarnick/students/global_registration/potemkin/exported']

PARAMETERS = '''
        iteration: 1000000
        metric: uniformity
        lrf: gravity
        bf: true
        matching: cluster
        alignment: ransac
        block_size: 200000
'''


@click.command()
@click.argument('config-path', type=click.Path(dir_okay=False))
@click.option('--selected/--all', type=bool, default=False)
@click.option('-l', '--level', default=2)
def generate_config(config_path, selected, level):
    df = pd.read_csv(DIFFICULTY_LEVELS_PATH)
    df = df[df['level'] >= level]
    df = df.fillna(0)
    df['selected'] = df['selected'].astype(bool)
    with open(config_path, 'a') as file:
        file.write('tests:\n')
        for f1, f2, feature_radius, s, with_vp in df[['source', 'target', 'feature_radius', 'selected', 'with_vp']].values:
            if not s and selected:
                continue
            for dirpath in DIRPATHS:
                if os.path.exists(os.path.join(dirpath, f1)):
                    file.write('    - test:')
                    file.write(PARAMETERS)
                    file.write(f'        ground_truth: ' + os.path.join(dirpath, 'ground_truth.csv') + '\n')
                    file.write(f'        source: {os.path.join(dirpath, f1)}\n')
                    file.write(f'        target: {os.path.join(dirpath, f2)}\n')
                    if with_vp == 1:
                        file.write(f'        viewpoints: ' + os.path.join(dirpath, 'viewpoints.csv') + '\n')


if __name__ == '__main__':
    generate_config()
