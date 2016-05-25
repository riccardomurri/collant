#! /bin/sh

TMAPS_USER=ubuntu

# "typical" PelkmansLab experiment parameters
nr_plates=1
nr_zplanes=1
nr_timepoints=1
nr_channels=4
nr_sites=50
nr_wells=250

NUMFILES=${1:-$(( $nr_plates * $nr_zplanes * $nr_timepoints * $nr_channels * $nr_sites * $nr_wells ))}
FILESIZE=10MB
EXPERIMENT=${2:-$PWD/tmp}
BATCH_SIZE=${3:-10}

URL="file://$EXPERIMENT"
echo "== Collecting performance data for $NUMFILES ($FILESIZE each) in '$URL' ..."

# save full path to avoid issues with $PATH across pdsh
collect_stats=$(command -v collect-stats)

run () {
    local tag="$1"; shift
    trap "pdsh -g nfs,db,compute $collect_stats abort;" EXIT INT ABRT TERM
    pdsh -g nfs,db,compute $collect_stats start
    sudo -u "$TMAPS_USER" --login "$@"
    pdsh -g nfs,db,compute $collect_stats stop
    $collect_stats save "${tag}"
    trap "" EXIT INT ABRT TERM
    sync
    sleep 2
}

set -ex  # debugging

iosim.py create $URL $NUMFILES $FILESIZE --jobs 100
sync

#                                                    ===============================  ======================  ================
#                                                    nr. jobs                         nr. images per job      payload (if any)
#                                                    ===============================  ======================  ================
run metaextract.iosim  iosim.py benchmark read $URL  $(( $NUMFILES / $BATCH_SIZE ))   $BATCH_SIZE
#run metaconfig.iosim
run imextract.iosim    iosim.py benchmark write $URL $(( $NUMFILES / $BATCH_SIZE ))   $BATCH_SIZE             $FILESIZE
run align.iosim        iosim.py benchmark read $URL  $(( $NUMFILES / $BATCH_SIZE ))   $(( 2 * $BATCH_SIZE ))
run corilla.iosim      iosim.py benchmark read $URL  $nr_channels                     $(( $NUMFILES / $nr_channels ))
run jterator.iosim     iosim.py benchmark read $URL  $(( $NUMFILES / $nr_channels ))  $nr_channels
# NOTE: the actual illuminati usage is *much* worse: around 7'000'000/$BATCH_SIZE jobs would get submitted, each processing $BATCH_SIZE images, which adds startup overhead to the (already huge) filesystem load ...
run illuminati.iosim   iosim.py benchmark write $URL 100                              70000                   16kB
