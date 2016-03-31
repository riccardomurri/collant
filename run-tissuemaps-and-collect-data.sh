#! /bin/sh

TMAPS_USER=tmaps

EXPERIMENT=${1:-/data_tmaps/storage/testdata/150820-Testset-CV}
BATCH_SIZE=${2:-10}
echo "== Collecting performance data for experiment '$EXPERIMENT' ..."

run () {
    local tag="$1"; shift
    trap "pdsh -w 'storage.tissuemap.com,compute[001-010]' ./collect-stats abort;" EXIT INT ABRT TERM
    pdsh -w 'storage.tissuemap.com,compute[001-010]' ./collect-stats start
    sudo -u "$TMAPS_USER" --login "$@"
    pdsh -w 'storage.tissuemap.com,compute[001-010]' ./collect-stats stop
    ./collect-stats save "${tag}"
    trap "" EXIT INT ABRT TERM
}

set -ex  # debugging

run metaextract.init    metaextract -v $EXPERIMENT init   --batch_size $BATCH_SIZE
run metaextract.run     metaextract -v $EXPERIMENT submit --phase run

run metaconfig.init    metaconfig -v $EXPERIMENT init --file_format cellvoyager
run metaconfig.run     metaconfig -v $EXPERIMENT submit --phase run
run metaconfig.collect metaconfig -v $EXPERIMENT submit --phase collect

run imextract.init imextract -v $EXPERIMENT init   --batch_size $BATCH_SIZE
run imextract.run  imextract -v $EXPERIMENT submit --phase run

run align.init    align -v $EXPERIMENT init --ref_cycle 1 --ref_channel 0 --batch_size $BATCH_SIZE
run align.run     align -v $EXPERIMENT submit --phase run
run align.collect align -v $EXPERIMENT submit --phase collect

run corilla.init corilla -v $EXPERIMENT init
run corilla.run  corilla -v $EXPERIMENT submit --phase run

run illuminati.init illuminati -v $EXPERIMENT init --illumcorr --align --clip --batch_size 10  # XXX: hard-coded!
run illuminati.run  illuminati -v $EXPERIMENT submit --phase run

run jterator.init    jterator -v $EXPERIMENT init                   --pipeline segment
run jterator.run     jterator -v $EXPERIMENT submit --phase run     --pipeline segment
