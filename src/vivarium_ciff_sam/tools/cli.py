"""
Make model specifications

click application that takes a template model specification file
and locations for which to create model specs and uses jinja2 to
render model specs with the correct location parameters plugged in.

It will look for the model spec template in "model_spec.in" in the directory
``src/vivarium_ciff_sam/model_specifications``.
Add location strings to the ``src/vivarium_ciff_sam/constants/metadata.py``
file. By default, specifications for all locations will be built. You can
choose to make a model specification for a single location by specifying
that location. However, the location string must exist in the list in
`src/vivarium_ciff_sam/constants/metadata.py``.

The application will look for the model spec based on the python environment
that is active and these files don't need to be specified if the
default names and location are used.
"""
import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_ciff_sam import paths
from vivarium_ciff_sam.constants import metadata
from vivarium_ciff_sam.tools import (build_artifacts,
                                                 build_model_specifications,
                                                 build_results,
                                                 configure_logging_to_terminal)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(metadata.LOCATIONS + ['all']),
              help=('Location for which to make an artifact. Note: prefer building archives on the cluster.\n'
                    'If you specify location "all" you must be on a cluster node.'))
@click.option('-o', '--output-dir',
              default=str(paths.ARTIFACT_ROOT),
              show_default=True,
              type=click.Path(),
              help='Specify an output directory. Directory must exist.')
@click.option('-a', '--append',
              is_flag=True,
              help='Append to the artifact instead of overwriting.')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_artifacts(location: str, output_dir: str, append: bool, verbose: int, with_debugger: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_artifacts, logger, with_debugger=with_debugger)
    main(location, output_dir, append, verbose)


@click.command()
@click.argument('output_file', type=click.Path(exists=True))
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
@click.option('-s', '--single', 'single_run',
              default=False,
              is_flag=True,
              help='Results are from a single, non-parallel run.')
def make_results(output_file: str, verbose: int, with_debugger: bool, single_run: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_results, logger, with_debugger=with_debugger)
    main(output_file, single_run)
