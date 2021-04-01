EXPT=$1
PYFILE=$EXPT".py"
INIT_INDICES=$EXPT
OUTFILE=$EXPT"out.txt"
TEMPFILE=$EXPT"temp.txt"
for INDICES in "initial_indices.txt"
do
    cat "data/"$INDICES > "data/"$INIT_INDICES$INDICES
    for VAR in $(seq 1 10)
        do
        python3 $PYFILE $ARGS --indices $INIT_INDICES$INDICES --out $OUTFILE
        cat "data/"$INIT_INDICES$INDICES "data/"$OUTFILE > "data/"$TEMPFILE
        cat "data/"$TEMPFILE > "data/"$INIT_INDICES$INDICES
        done
done
