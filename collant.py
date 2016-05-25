#! /usr/bin/env python
#
# Copyright (C) 2016 University of Zurich.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# make coding more python3-ish, must be the first statement
from __future__ import (print_function, division, absolute_import)

## module doc and other metadata
"""
Ingest raw data files produced by `collectl` and make nice plots.
"""
__docformat__ = 'reStructuredText'
__author__ = ('Riccardo Murri <riccardo.murri@gmail.com>')


## imports and other dependencies
from collections import defaultdict, namedtuple, OrderedDict
from contextlib import contextmanager
import cPickle as pickle
from cStringIO import StringIO
from datetime import datetime, date, timedelta
import logging
import os
from subprocess import check_output as run
import sys
import time


from click import argument, command, group, option, echo

import dataset

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

import sqlalchemy



class const:
    """A namespace for constant and default values."""

    logfmt = "%(asctime)s [%(processName)s/%(process)d %(funcName)s:%(lineno)d] %(levelname)s: %(message)s"
    loglevel = logging.DEBUG

    db_uri = 'sqlite:///perfdata.db'

    hosts = {
        # Restricted view:
        #    'compute001': (0, 0),
        #    'tmaps':      (0, 1),
        # Full view:
         'compute001': (0, 0),
         'compute002': (0, 1),
         'compute003': (0, 2),
         'compute004': (0, 3),
         'compute005': (1, 0),
         'compute006': (1, 1),
         'compute007': (1, 2),
         'compute008': (1, 3),
         'compute009': (2, 0),
         'compute010': (2, 1),
         # -- empty slots for visual emphasis --
         'tmaps':      (2, 3)
    }

    steps = [
        ('metaextract', 'init'),
        ('metaextract', 'run'),
        ('metaconfig',  'init'),
        ('metaconfig',  'run'),
        ('metaconfig',  'collect'),
        ('imextract',   'init'),
        ('imextract',   'run'),
        ('align',       'init'),
        ('align',       'run'),
        ('align',       'collect'),
        ('corilla',     'init'),
        ('corilla',     'run'),
        ('illuminati',  'init'),
        ('illuminati',  'run'),
        ('jterator',    'init'),
        ('jterator',    'run'),
        ('jterator',    'collect'),
    ]

ONE_SECOND = timedelta(seconds=1)


#
# Plot definitions
#

def plot_cpu_utilization(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    perfdata[['cpu_user_percent', 'cpu_sys_percent', 'cpu_wait_percent']].plot(ax=ax, kind='area', stacked=True, ylim=(0, 100))


def plot_net_traffic_pkts(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    perfdata[['net_rxpkttot', 'net_txpkttot']].plot(
        ax=ax, kind='line',
    )


def plot_net_traffic_kb(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    perfdata[['net_rxkbtot', 'net_txkbtot']].plot(
        ax=ax, kind='line',
    )


def plot_nfsv4_ops(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    if hostname == 'tmaps':
        # use server-side counters
        cols = ['nfs_4sd_read', 'nfs_4sd_write', 'nfs_4sd_commit']
    else:
        cols = ['nfs_4cd_read', 'nfs_4cd_write', 'nfs_4cd_commit']
    perfdata[cols].plot(ax=ax, kind='line')


def plot_nfsv4_md_read(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    if hostname == 'tmaps':
        # use server-side ctrs
        cols = [
            'nfs_4sd_access',
            'nfs_4sd_getattr',
            'nfs_4sd_lookup',
            'nfs_4sd_readdir',
        ]
    else:
        cols = [
            'nfs_4cd_lookup',
            'nfs_4cd_access',
            'nfs_4cd_getattr',
            'nfs_4cd_setattr',
            'nfs_4cd_readdir',
            'nfs_4cd_readlink',
        ]
    perfdata[cols].plot(ax=ax, kind='line')


def plot_nfsv4_md_write(hostname, fig, ax, perfdata):
    width, height = _get_ax_size(fig, ax)
    perfdata = _resample_to_fit_width(perfdata, width, how='max')
    if hostname == 'tmaps':
        # use server-side ctrs
        cols = [
            'nfs_4sd_create',
            'nfs_4sd_remove',
            'nfs_4sd_rename',
            'nfs_4sd_setattr',
        ]
    else:
        cols = [
            'nfs_4cd_create',
            'nfs_4cd_link',
            'nfs_4cd_remove',
            'nfs_4cd_rename',
            'nfs_4cd_setattr',
            'nfs_4cd_symlink',
        ]
    perfdata[cols].plot(ax=ax, kind='line')


def _get_ax_size(fig, ax):
    """
    Return (width, height) in pixels of axis `ax` in figure `fig`.

    See: http://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def _resample_to_fit_width(perfdata, width, how='max'):
    """
    Downsample data to have no more than 2*width samples.
    If `perfdata` already contains less than that data points, then
    return it unchanged.
    """
    samples = len(perfdata.index)
    if samples > 2*width:
        # resample to match resolution
        t_start = perfdata.index.min()
        t_end = perfdata.index.max()
        duration_ms = 1 + int((t_end - t_start).total_seconds() * 1e3)
        tick_duration_ms = int(duration_ms / width)
        # see: http://stackoverflow.com/questions/17001389/pandas-resample-documentation
        return perfdata.resample(rule='{0}L'.format(tick_duration_ms), how=how)
    else:
        # no changes
        return perfdata


def plot_perfdata(outfile, db_uri, hosts=const.hosts):
    # bring all data into memory
    data = pd.read_sql('perfdata', db_uri,
                       index_col=['step', 'host', 'timestamp'],
                       parse_dates=['timestamp'])

    # matplotlib initialization
    matplotlib.style.use('ggplot')

    NROWS = 1 + max(x for x,y in hosts.values())
    NCOLS = 1 + max(y for x,y in hosts.values())

    # make plots
    pdf = PdfPages(outfile)
    for step, phase in const.steps:
        step_and_phase = '{step}.{phase}'.format(step=step, phase=phase)
        # some datasets skip `align.*`, etc. - data is, as always, not consistent
        steps_in_data = data.index.get_level_values('step')
        if step_and_phase not in steps_in_data:
            logging.warning("Skipping plotting step `%s`: no data collected!", step_and_phase)
            continue
        else:
            logging.info("Plotting data for step `%s` ...", step_and_phase)
        for plot, title in [
                (plot_cpu_utilization,  "CPU utilization"),
                (plot_net_traffic_pkts, "Net traffic (packets)"),
                (plot_net_traffic_kb,   "Net traffic (kB)"),
                (plot_nfsv4_ops,        "NFSv4 data ops"),
                (plot_nfsv4_md_read,    "NFSv4 meta-data read"),
                (plot_nfsv4_md_write,   "NFSv4 meta-data write"),
        ]:
            logging.info("* Plotting '%s' ...", title)
            fig, axes = plt.subplots(
                nrows=NROWS, ncols=NCOLS,
                sharex=True, sharey=True,
                figsize=(14*NCOLS, 10*NROWS),
            )
            fig.suptitle('{step}.{phase} - {title}'
                         .format(step=step, phase=phase, title=title))
            for host in hosts:
                hosts_in_data = data.index.get_level_values('host')
                if host not in hosts_in_data:
                    logging.warning("Skipping host `%s`: no data collected!", host)
                    continue
                perfdata = data.loc[step_and_phase, host]
                logging.debug("  - for host '%s' ...", host)
                # select subplot axis; index is different depending on
                # whether the subplots are arranged in a row, column,
                # or a 2D matrix ...
                x, y = hosts[host]
                if NROWS == 1:
                    loc = y
                elif NCOLS == 1:
                    loc = x
                else:
                    loc = x, y
                ax = axes[loc]
                plot(host, fig, ax, perfdata)
                ax.set_title(host)
            pdf.savefig()
            plt.close(fig)

    # actually output PDF file
    pdf.close()

    logging.info("All done.")


#
# Aux functions
#

def _setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(
            fmt=const.logfmt,
            level=const.loglevel
        )
    except ImportError:
        logging.basicConfig(
            format=const.logfmt,
            level=const.loglevel
        )


@contextmanager
def stream_reader(file_obj_or_name):
    """
    Ensure a "readable" object can be used in a `with` statement.

    A "readable" object is defined as any object that implements
    callable methods `.read()` and `.close()`.

    This is useful, e.g., with `StringIO` file-like objects, which do
    not implement the context manager protocol natively::

      >>> from cStringIO import StringIO
      >>> s = StringIO('a line of data')
      >>> with stream_reader(s) as stream:
      ...   for line in stream:
      ...     print(line)
      a line of data
    """
    try:
        # just do attr lookup, do not actually call `.read()` as it
        # would consume bytes from the input stream
        file_obj_or_name.read
        reader = file_obj_or_name
    except AttributeError:
        reader = open(file_obj_or_name, 'r')
    try:
        yield reader
    finally:
        # this is basically `contextlib.closing()`
        try:
            reader.close()
        except AttributeError:
            # we don't really care
            pass


def _expand_db_uri(db_uri):
    if db_uri is None:
        return const.db_uri
    elif ':' not in db_uri:
        # map filenames to SQLite URIs
        return ('sqlite:///{path}'.format(path=os.path.abspath(db_uri)))
    else:
        return db_uri


@contextmanager
def database(db_uri):
    db_uri = _expand_db_uri(db_uri)
    logging.info("Using database URI `%s`", db_uri)
    with dataset.connect(db_uri, engine_kwargs={'poolclass':sqlalchemy.pool.StaticPool}) as db:
        # ensure a minimal structure is there
        db.get_table('perfdata')
        db.get_table('mtime', 'path', 'String').create_index(['path'])
        # pass control to caller
        yield db


#
# Data ingestion
#

def _get_metadata_from_filename(pathname):
    """
    Parse a path name into components and return them in the form of a dictionary.

    Example::

      >>> md = _get_metadata_from_filename('/tmp/metaextract.run/iostat_cpu.2016-02-04.1107.compute001.log')
      >>> md['step'] == 'metaextract'
      True
      >>> md['phase'] == 'run'
      True
      >>> md['label'] == 'iostat_cpu'
      True
      >>> md['hostname'] == 'compute001'
      True
      >>> ts = md['timestamp']
      >>> ts.date() == date(2016, 2, 4)
      True
      >>> ts.hour == 11
      True
      >>> ts.minute == 7
      True

      >>> md = _get_metadata_from_filename('align.collect/collectl.2016-02-26.1004.compute001-compute001-20160226-100445.raw.gz')
      >>> md['step'] == 'align'
      True
      >>> md['phase'] == 'collect'
      True
      >>> md['label'] == 'collectl'
      True
      >>> md['hostname'] == 'compute001'
      True
      >>> ts = md['timestamp']
      >>> ts.date() == date(2016, 2, 26)
      True
      >>> ts.hour == 10
      True
      >>> ts.minute == 4
      True

    File names are assumed to have the form shown in the above
    examples.  Any deviation will result in `ValueError` exceptions
    being thrown.
    """
    metadata = {}
    parts = pathname.split('/')
    # last directory name is, e.g., `metaextract.run`
    metadata['step'], metadata['phase'] = parts[-2].split('.')
    # file name format is "${label}.${date}.${hhmm}.${hostname}.log"
    metadata['label'], date, hhmm, more = parts[-1].split('.')[:4]
    year_, month_, day_ = date.split('-')
    year = int(year_)
    month = int(month_)
    day = int(day_)
    hour = int(hhmm[:2])
    minutes = int(hhmm[2:])
    metadata['timestamp'] = datetime(year, month, day, hour, minutes)
    if metadata['label'] == 'collectl':
        metadata['hostname'] = more.split('-')[0]
    else:
        metadata['hostname'] = more
    return metadata


def _collectl_column_name(colname):
    """
    Convert a column name in `collectl` output to a valid Python identifier.

    Examples::

      >>> _collectl_column_name('[NFS:2cd]Lookup')
      'nfs_2cd_lookup'

      >>> _collectl_column_name('[CPU]GuestN%')
      'cpu_guestn_percent'

      >>> _collectl_column_name('[CPU]Intrpt/sec')
      'cpu_intrpt_per_second'

    """
    return (colname.lower()
            .replace('[', '')
            .replace(']', '_')
            .replace(':', '_')
            .replace('-', '_')
            .replace('%', '_percent')
            .replace('/sec', '_per_second'))


def _skip_file(path, db):
    """
    Return ``True`` if data in DB has been written later than `path` was modified.
    """
    stat = os.stat(path)
    row = db['mtime'].find_one(path=path)
    if row and stat.st_mtime < row['mtime']:
        logging.info(
            "File '%s' was last modified earlier than data in the DB."
            " Skipping it; use option `--force` to process it anyway.",
            path)
        return True
    return False


def _find_collectl_raw_files(rootdir, db, force=False):
    """
    Iterate over collectl "raw data" file names in the directory tree
    rooted at `rootdir`.
    """
    for dirpath, dirnames, filenames in os.walk(rootdir):
        logging.debug("Entering directory `%s` ...", dirpath)
        for filename in filenames:
            if not filename.startswith('collectl') or not filename.endswith('.raw.gz'):
                logging.debug("Ignoring file `%s`: not a `collectl` raw data file.", filename)
                continue
            path = os.path.join(dirpath, filename)
            if force or not _skip_file(path, db):
                yield path


def read_collectl_plot_data(file, format='dict'):
    """
    Parse the output of ``collectl --plot -p filename``.
    """
    # StringI objects do not implement the context protocol,
    # so we have to use a custom contextmanager
    with stream_reader(file) as lines:
        # skip until we find the column header line, which starts with `# Date Time ...`
        for line in lines:
            if line.startswith('#Date'):
                break
        # extract column names from header line
        header = line[1:]  # skip initial `#`
        colnames = [
            _collectl_column_name(col)
            for col in header.split()
        ]
        if format == 'pandas':
            return pd.read_csv(
                lines,
                delim_whitespace=True,
                header=0,
                names=colnames,
                parse_dates={'timestamp':[0,1]},
                comment='#',
                error_bad_lines=False,
                index_col=0,
            )
        elif format == 'dict':
            series = []
            for line in lines:
                values = line.split()
                # combine first two values into a timestamp; however,
                # `datetime.strptime()` will not handle fractional
                # seconds (as in ``20160223 15:16:45.402``) so, in
                # order to preserve timestamp resolution, we need to
                # split fractions of a second off and add them later
                timestamp_full = ' '.join(values[0:2])
                timestamp_int, timestamp_frac = timestamp_full.split('.')
                timestamp = datetime.strptime(timestamp_int, '%Y%m%d %H:%M:%S')
                timestamp += int(1e6 * float('.' + timestamp_frac)) * timedelta.resolution
                del values[0:2]
                # create row dictionary
                data = { 'timestamp':timestamp }
                for col, value in zip(colnames, values):
                    data[col] = float(value)
                series.append(data)
            return series
        else:
            raise ValueError(
                "Unknown `format` value `%r`:"
                " valid values are `pandas`, or `dict`.")


def read_collectl_raw_data(filename, format='dict'):
    """
    Parse a collectl raw data file.

    Spawns an external `collectl` process to convert it
    into plot format.
    """
    if filename.endswith('.raw.gz'):
        stream = StringIO(run(['collectl', '-o', 'm', '-P', '-p', filename]))
    else:
        stream = filename
    return read_collectl_plot_data(stream, format)


def _load_perfdata_file(path, db, force=False, metadata=None):
    """
    Load contents of `file` into `db`.

    The DB contains a table of file names and load time, which is
    updated by this function.  If the entry corresponding to `file` in
    this table is newer than the last modification to `file`, loading
    is skipped unless argument `force` is true.
    """
    if not force and _skip_file(path, db):
        return False
    logging.info("Processing file `%s` ...", path)
    if metadata is None:
        metadata = _get_metadata_from_filename(path)
    host = metadata['hostname']
    step = metadata['step'] + '.' + metadata['phase']
    data = read_collectl_raw_data(path, format='dict')
    # insert all data in the DB in one go
    db['perfdata'].insert_many(dict(host=host, step=step, **row) for row in data)

    # update last modification timestamp
    db['mtime'].upsert(dict(path=path, mtime=time.time()), ['path'])
    return True


def _load_all_perfdata(rootdir, db, force=False, only_hosts=None):
    """
    Load all files found in the directory tree rooted at `rootdir` into `db`.

    Each file is loaded using :func:`_load_perfdata_file` (which see).
    """
    for path in _find_collectl_raw_files(rootdir, db, force):
        md = _get_metadata_from_filename(path)
        hostname = md['hostname']
        if only_hosts is not None and hostname not in only_hosts:
            logging.warning("Skipping host `%s`: not in desired host list.", hostname)
            continue
        # "force" loading: the check has already been performed by `_find_collectl_raw_files`
        _load_perfdata_file(path, db, force=True, metadata=md)



#
# Command-line interface definition
#

@group()
def cli():
    pass


@cli.command()
@argument("rootdir")
@option("--database", "--db", 'db_uri',
        default=None, envvar='DATABASE_URL', metavar='URI',
        help="Path to DB file or connection string")
@option("--force/--no-force", default=False,
        help="Process file even if data in the DB is newer.")
def scan(rootdir, db_uri, force=False):
    """
    Load collectl raw data files from a directory into the DB.
    """
    _setup_logging()
    with database(db_uri) as db:
        _load_all_perfdata(rootdir, db, force)


@cli.command()
@argument("path")
@option("--database", "--db", 'db_uri',
        default=None, envvar='DATABASE_URL', metavar='URI',
        help="Path to DB file or connection string")
@option("--force/--no-force", default=False,
        help="Process file even if data in the DB is newer.")
def load(path, db_uri, force=False):
    """Load a `collectl` raw data file into the DB."""
    _setup_logging()
    with database(db_uri) as db:
        _load_perfdata_file(path, db, force)


@cli.command()
@argument("outfile")
@option("--database", "--db", 'db_uri',
        default=None, envvar='DATABASE_URL', metavar='URI',
        help="Path to DB file or connection string")
def plot(outfile, db_uri):
    """Plot data from the database."""
    _setup_logging()
    db_uri = _expand_db_uri(db_uri)
    logging.info("Using database URI `%s`", db_uri)
    plot_perfdata(outfile, db_uri)


@cli.command()
def selftest():
    """Run unit tests."""
    try:
        import pytest
        pytest.main(['-v', '--doctest-modules', __file__])
    except ImportError:
        # no `py.test`, but `doctest` is always available
        import doctest
        doctest.testmod(name="collant",
                        optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    cli()
