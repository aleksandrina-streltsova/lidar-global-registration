import pandas as pd
import matplotlib.pyplot as plt
import click


@click.group()
def cli():
    pass


@cli.command('histogram')
@click.argument('values_path', type=click.Path(dir_okay=False, exists=True))
@click.argument('hist_path', type=click.Path(dir_okay=False))
def histogram(values_path: str, hist_path: str):
    df = pd.read_csv(values_path)
    values = df.values
    plt.grid(linestyle='--')
    plt.hist(values, bins=256)
    plt.savefig(hist_path, bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    cli()