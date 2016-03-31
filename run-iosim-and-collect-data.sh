#! /bin/sh

TMAPS_USER=ubuntu

NUMFILES=${1:-10000}
FILESIZE=10MB
EXPERIMENT=${2:-/data/storage/riccardo.tmp}
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

run metaextract.iosim  iosim.py benchmark read $URL 100  $NUMFILES
#run metaconfig.iosim
#run imextract.iosim
run align.iosim        iosim.py benchmark read $URL 1000 $NUMFILES  # FIXME: should read twice the images!
run corilla.iosim      iosim.py benchmark read $URL 2    $NUMFILES
run jterator.iosim     iosim.py benchmark read $URL 5000 $NUMFILES
#run illuminati.iosim
