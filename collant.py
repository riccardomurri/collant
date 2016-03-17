#! /usr/bin/env python
from __future__ import (print_function, division, absolute_import)

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


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd


ONE_SECOND = timedelta(seconds=1)


# available fields for plotting
FIELDS = [
    'cpu_user%',
    'cpu_nice%',
    'cpu_sys%',
    'cpu_wait%',
    'cpu_irq%',
    'cpu_soft%',
    'cpu_steal%',
    'cpu_idle%',
    'cpu_totl%',
    'cpu_guest%',
    'cpu_guestn%',
    'cpu_intrpt/sec',
    'cpu_ctx/sec',
    'cpu_proc/sec',
    'cpu_procque',
    'cpu_procrun',
    'cpu_l-avg1',
    'cpu_l-avg5',
    'cpu_l-avg15',
    'cpu_runtot',
    'cpu_blktot',
    'net_rxpkttot',
    'net_txpkttot',
    'net_rxkbtot',
    'net_txkbtot',
    'net_rxcmptot',
    'net_rxmlttot',
    'net_txcmptot',
    'net_rxerrstot',
    'net_txerrstot',
    'nfs_3cd_read',
    'nfs_3cd_write',
    'nfs_3cd_commit',
    'nfs_3cd_lookup',
    'nfs_3cd_access',
    'nfs_3cd_getattr',
    'nfs_3cd_setattr',
    'nfs_3cd_readdir',
    'nfs_3cd_create',
    'nfs_3cd_remove',
    'nfs_3cd_rename',
    'nfs_3cd_link',
    'nfs_3cd_readlink',
    'nfs_3cd_null',
    'nfs_3cd_symlink',
    'nfs_3cd_mkdir',
    'nfs_3cd_rmdir',
    'nfs_3cd_fsstat',
    'nfs_3cd_fsinfo',
    'nfs_3cd_pathconf',
    'nfs_3cd_mknod',
    'nfs_3cd_readdirplus',
    'nfs_3sd_read',
    'nfs_3sd_write',
    'nfs_3sd_commit',
    'nfs_3sd_lookup',
    'nfs_3sd_access',
    'nfs_3sd_getattr',
    'nfs_3sd_setattr',
    'nfs_3sd_readdir',
    'nfs_3sd_create',
    'nfs_3sd_remove',
    'nfs_3sd_rename',
    'nfs_3sd_link',
    'nfs_3sd_readlink',
    'nfs_3sd_null',
    'nfs_3sd_symlink',
    'nfs_3sd_mkdir',
    'nfs_3sd_rmdir',
    'nfs_3sd_fsstat',
    'nfs_3sd_fsinfo',
    'nfs_3sd_pathconf',
    'nfs_3sd_mknod',
    'nfs_3sd_readdirplus',
    'nfs_4cd_read',
    'nfs_4cd_write',
    'nfs_4cd_commit',
    'nfs_4cd_lookup',
    'nfs_4cd_access',
    'nfs_4cd_getattr',
    'nfs_4cd_setattr',
    'nfs_4cd_readdir',
    'nfs_4cd_create',
    'nfs_4cd_remove',
    'nfs_4cd_rename',
    'nfs_4cd_link',
    'nfs_4cd_readlink',
    'nfs_4cd_null',
    'nfs_4cd_symlink',
    'nfs_4cd_fsinfo',
    'nfs_4cd_pathconf',
    'nfs_4sd_read',
    'nfs_4sd_write',
    'nfs_4sd_commit',
    'nfs_4sd_lookup',
    'nfs_4sd_access',
    'nfs_4sd_getattr',
    'nfs_4sd_setattr',
    'nfs_4sd_readdir',
    'nfs_4sd_create',
    'nfs_4sd_remove',
    'nfs_4sd_rename',
    'nfs_4sd_link',
    'nfs_4sd_readlink',
]


def plot_cpu_utilization(hostname, ax, perfdata):
    perfdata[['cpu_user%', 'cpu_sys%', 'cpu_wait%']].plot(
        ax=ax, kind='area', stacked=True,
    )


def plot_net_traffic_pkts(hostname, ax, perfdata):
    perfdata[['net_rxpkttot', 'net_txpkttot']].plot(
        ax=ax, kind='line',
    )


def plot_net_traffic_kb(hostname, ax, perfdata):
    perfdata[['net_rxkbtot', 'net_txkbtot']].plot(
        ax=ax, kind='line',
    )


def plot_nfsv4_ops(hostname, ax, perfdata):
    if hostname == 'tmaps':
        # use server-side counters
        cols = ['nfs_4sd_read', 'nfs_4sd_write', 'nfs_4sd_commit']
    else:
        cols = ['nfs_4cd_read', 'nfs_4cd_write', 'nfs_4cd_commit']
    perfdata[cols].plot(ax=ax, kind='line')


def plot_nfsv4_md_read(hostname, ax, perfdata):
    if hostname == 'tmaps':
        # use server-side ctrs
        cols = [
            'nfs_4sd_lookup',
            'nfs_4sd_access',
            'nfs_4sd_getattr',
            'nfs_4sd_readdir',
            'nfs_4sd_readlink',
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


def plot_nfsv4_md_write(hostname, ax, perfdata):
    if hostname == 'tmaps':
        # use server-side ctrs
        cols = [
            'nfs_4sd_setattr',
            'nfs_4sd_create',
            'nfs_4sd_remove',
            'nfs_4sd_rename',
            'nfs_4sd_link',
        ]
    else:
        cols = [
            'nfs_4cd_setattr',
            'nfs_4cd_create',
            'nfs_4cd_remove',
            'nfs_4cd_rename',
            'nfs_4cd_link',
            'nfs_4cd_symlink',
        ]
    perfdata[cols].plot(ax=ax, kind='line')


def get_metadata_from_filename(pathname):
    """
    Parse a path name into components and return them in the form of a dictionary.

    Example::

      >>> md = get_metadata_from_filename('/tmp/metaextract.run/iostat_cpu.2016-02-04.1107.compute001.log')
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

      >>> md = get_metadata_from_filename('align.collect/collectl.2016-02-26.1004.compute001-compute001-20160226-100445.raw.gz')
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
    md = {}
    parts = pathname.split('/')
    # last directory name is, e.g., `metaextract.run`
    md['step'], md['phase'] = parts[-2].split('.')
    # file name format is "${label}.${date}.${hhmm}.${hostname}.log"
    md['label'], date, hhmm, more = parts[-1].split('.')[:4]
    year_, month_, day_ = date.split('-')
    year = int(year_)
    month = int(month_)
    day = int(day_)
    hour = int(hhmm[:2])
    minutes = int(hhmm[2:])
    md['timestamp'] = datetime(year, month, day, hour, minutes)
    if md['label'] == 'collectl':
        md['hostname'] = more.split('-')[0]
    else:
        md['hostname'] = more
    return md


def _collectl_column_name(colname):
    """
    Convert a column name in `collectl` output to a valid Python identifier.

    Example::

      >>> _collectl_column_name('[NFS:2cd]Lookup')
      'nfs_2cd_lookup'
    """
    return (colname
            .replace('[','')
            .replace(']','_')
            .replace(':','_')
            .lower())


@contextmanager
def stream_reader(file_obj_or_name):
    """
    Ensure a "readable" object can be used in a `with` statement.

    A "readable" object is defined as any object that implements
    callable methods `.read()` and `.close()`.
    """
    try:
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


def read_collectl_plot_data(file, with_pandas=True):
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
        if with_pandas:
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
        else:
            series = []
            for line in lines:
                values = line.split()
                data = defaultdict(int)
                for col, value in zip(colnames, values):
                    data[col] = int(value)
                series.append(data)


def read_collectl(filename, with_pandas=True):
    """
    Parse the output of ``collectl --plot -p filename``.
    """
    if filename.endswith('.raw.gz'):
        stream = StringIO(run(['collectl', '-o', 'm', '-P', '-p', filename]))
    else:
        stream = filename
    return read_collectl_plot_data(stream, with_pandas)


def load_all_perfdata(rootdir, only_hosts=None, with_cache=True):
    cache_filename = rootdir + '.cache'
    dump_cache = False

    if with_cache and os.path.exists(cache_filename):
        logging.debug("Loading parsed data from cache file '%s' ...", cache_filename)
        with open(cache_filename, 'r') as cache:
            data, updated = pickle.load(cache)
    else:
        # data[step_and_phase][hostname]
        data = {}
        updated = {}

    # read all collectl files
    for dirpath, dirnames, filenames in os.walk(rootdir):
        logging.debug("Entering directory `%s` ...", dirpath)
        for filename in filenames:
            if not filename.startswith('collectl') or not filename.endswith('.raw.gz'):
                logging.debug("Ignoring file `%s`: not a `collectl` raw data file.", filename)
                continue
            path = os.path.join(dirpath, filename)
            md = get_metadata_from_filename(path)
            hostname = md['hostname']
            if hostname not in only_hosts:
                logging.warning("Skipping host `%s`: not in desired host list.", hostname)
                continue
            step_and_phase = '{step}.{phase}'.format(**md)
            if step_and_phase not in data:
                data[step_and_phase] = {}
                updated[step_and_phase] = {}
            sb = os.stat(path)
            if sb.st_mtime > updated[step_and_phase].get(hostname, 0):
                logging.info("Processing file `%s` ...", path)
                data[step_and_phase][hostname] = read_collectl(path)
                updated[step_and_phase][hostname] = time.time()
                dump_cache = True
            else:
                logging.info("File `%s` not changed since last cache update - skipping.", path)

    if with_cache and dump_cache:
        with open(cache_filename, 'w') as cache:
            logging.debug("Writing parsed data to cache file '%s' ...", cache_filename)
            pickle.dump((data, updated), cache, pickle.HIGHEST_PROTOCOL)

    return data


def main(args):

    logfmt = "%(asctime)s [%(processName)s/%(process)d %(funcName)s:%(lineno)d] %(levelname)s: %(message)s"
    try:
        import coloredlogs
        coloredlogs.install(
            fmt=logfmt,
            level=logging.DEBUG
        )
    except ImportError:
        logging.basicConfig(
            format=logfmt,
            level=logging.DEBUG,
        )

    #HOSTS = {
    #    'compute001': (0, 0),
    #    'tmaps':      (0, 1),
    #}
    HOSTS = {
         'compute001': (0, 0),
         'compute002': (0, 1),
         'compute003': (0, 2),
         'compute004': (0, 3),
         'compute005': (1, 0),
         'compute006': (1, 1),
         'compute007': (1, 2),
         'compute008': (1, 3),
         'compute009': (2, 1),
         'compute010': (2, 2),
         #
         'tmaps':      (2, 3)
    }
    NROWS = 1 + max(x for x,y in HOSTS.values())
    NCOLS = 1 + max(y for x,y in HOSTS.values())

    root = args[1]
    logging.info("Reading collected data from `%s` ...", root)

    data = load_all_perfdata(root, only_hosts=HOSTS.keys())

    # matplotlib initialization
    matplotlib.style.use('ggplot')
    #plt.ion()

    # make plots
    pdf = PdfPages(sys.argv[1] + '.pdf')
    for step, phase in [
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
    ]:
        step_and_phase = '{step}.{phase}'.format(step=step, phase=phase)
        # some datasets skip `align.*`, etc. - data is, as always, not consistent
        if step_and_phase not in data:
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
            logging.info("Plotting '%s' ...", title)
            fig, axes = plt.subplots(
                nrows=NROWS, ncols=NCOLS,
                sharex=True, sharey=True,
                figsize=(14*NCOLS, 10*NROWS),
            )
            fig.suptitle('{step}.{phase} - {title}'.format(step=step, phase=phase, title=title))
            for hostname, perfdata in data[step_and_phase].items():
                logging.debug("  - for host '%s' ...", hostname)
                # select subplot axis; index is different depending on
                # whether the subplots are arranged in a row, column,
                # or a 2D matrix ...
                x, y = HOSTS[hostname]
                if NROWS == 1:
                    loc = y
                elif NCOLS == 1:
                    loc = x
                else:
                    loc = x, y
                ax = axes[loc]
                plot(hostname, ax, perfdata)
                ax.set_title(hostname)
            pdf.savefig()
            plt.close(fig)

    # actually output PDF file
    pdf.close()

    logging.info("All done.")


if __name__ == '__main__':
    main(sys.argv)
