EXPT=$1
DATA=$2
ARGS="--data "$DATA
PYFILE=$EXPT".py"
INIT_INDICES=$EXPT
OUTFILE=$EXPT"out.txt"
TEMPFILE=$EXPT"temp.txt"
for INDICES in "initial_50_1.txt" "initial_50_2.txt" "initial_50_3.txt"
do
    cat "data/"$DATA"/"$INDICES > "data/"$DATA"/"$INIT_INDICES$INDICES
    for VAR in $(seq 1 10)
        do
        python $PYFILE $ARGS --indices $INIT_INDICES$INDICES --out $OUTFILE
        cat "data/"$DATA"/"$INIT_INDICES$INDICES "data/"$DATA"/"$OUTFILE > "data/"$DATA"/"$TEMPFILE
        cat "data/"$DATA"/"$TEMPFILE > "data/"$DATA"/"$INIT_INDICES$INDICES
        done
done
rm "data/"$DATA"/"$TEMPFILE "data/"$DATA"/"$OUTFILE
