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
    python -u $PYFILE $ARGS --indices $INIT_INDICES$INDICES
done
rm "data/"$DATA"/"$TEMPFILE "data/"$DATA"/"$OUTFILE
