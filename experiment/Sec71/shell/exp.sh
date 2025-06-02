export CUDA_VISIBLE_DEVICES=0
# on noremal dataset, at the original setting, compare with SGD-Influence Estimator
SEED_LIST=(0 1 2 3 4 5)
TARGET_LIST=(mnist adult 20news)
MODEL=dnn
INIT=0
CORRUPTED=0
CORRUPTED_SIGMA=0
NOISE=0
NOISE_RATE=0
DATASIZE=400


EPOCHS=(2 6 10 14 18 22 26 30 34 38)
BATCH_SIZES=(100) 
LRS=(0.1)

for SEED in "${SEED_LIST[@]}"
  do
    for TARGET in  "${TARGET_LIST[@]}"
    do
      for EPOCH in "${EPOCHS[@]}"
      do
      (
        for BATCH_SIZE in "${BATCH_SIZES[@]}"
        do
          for LR in "${LRS[@]}"
          do  
              

              python ../train.py \
                  --target $TARGET \
                  --model $MODEL \
                  --datasize $DATASIZE \
                  --epoch $EPOCH \
                  --batch-size $BATCH_SIZE \
                  --lr $LR \
                  --corrupted $CORRUPTED \
                  --corruption_sigma $CORRUPTED_SIGMA \
                  --noise $NOISE \
                  --noise_rate $NOISE_RATE \
                  --init $INIT \
                  --seed $SEED
                  


              FULL_STEP=$((EPOCH*DATASIZE/BATCH_SIZE))
              SIMPJ_LIST=(0)
              for SIMPJ in "${SIMPJ_LIST[@]}"
              do
                  TYPE=proposed
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
                      --corruption_sigma $CORRUPTED_SIGMA \
                      --noise $NOISE \
                      --noise_rate $NOISE_RATE \
                      --init $INIT \
                      --seed $SEED
                      


                  infl_types=(true sgd)
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
                          --corruption_sigma $CORRUPTED_SIGMA \
                          --noise $NOISE \
                          --noise_rate $NOISE_RATE \
                          --init $INIT \
                          --seed $SEED
                  done

                  python ../plot.py \
                      --target $TARGET \
                      --model $MODEL \
                      --datasize $DATASIZE \
                      --epoch $EPOCH \
                      --batch-size $BATCH_SIZE \
                      --lr $LR \
                      --simpj $SIMPJ \
                      --corrupted $CORRUPTED \
                      --corruption_sigma $CORRUPTED_SIGMA \
                      --noise $NOISE \
                      --noise_rate $NOISE_RATE \
                      --init $INIT \
                      --seed $SEED
              done  
          done
        done
      ) &
      done
    done
  done