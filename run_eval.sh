EVALFILE=$1
EXPT=$2
DATA=$3
ARGS="--data "$DATA
PYFILE=$EVALFILE".py"
INIT_INDICES=$EXPT
for INDICES in "initial_50_1.txt" "initial_50_2.txt" "initial_50_3.txt"
do
    python $PYFILE $ARGS --indices $INIT_INDICES$INDICES
done
