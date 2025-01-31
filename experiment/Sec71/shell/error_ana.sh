TARGET=mnist
MODEL=dnn
INIT=0
SIMPJ=0
CORRUPTED=1
DATASIZE=200


EPOCHS=(1 2 4 8 16)
BATCH_SIZES=(20) 
LRS=(0.1)

for EPOCH in "${EPOCHS[@]}"
do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"
  do
    for LR in "${LRS[@]}"
    do  

        infl_types=(true sgd icml proposed)

        python ../train.py \
            --target $TARGET \
            --model $MODEL \
            --datasize $DATASIZE \
            --epoch $EPOCH \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --corrupted $CORRUPTED \
            --init $INIT

        for TYPE in "${infl_types[@]}"
        do  
            python ../infl.py \
                --target $TARGET \
                --model $MODEL \
                --type $TYPE \
                --datasize $DATASIZE \
                --epoch $EPOCH \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --simpj $SIMPJ \
                --corrupted $CORRUPTED \
                --init $INIT &
        done
        #等待线程结束
        wait

        python ../plot.py \
            --target $TARGET \
            --model $MODEL \
            --datasize $DATASIZE \
            --epoch $EPOCH \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --simpj $SIMPJ \
            --corrupted $CORRUPTED \
            --init $INIT 
    done
  done
done

