# bash run_w2v.sh
export INFER_INDEX=5

CTRL_NAME='grm_common'

# python infer.py --output_file '_en_ana_'$INFER_INDEX \
#     --model_file 'GRM_epoch_'${INFER_INDEX}'.pt' --ctrlname $CTRL_NAME \
#     --batch_size 256 --oov_file '../downstream/word_ana/word.txt' --cuda 0 \
#     --config './experiments_en.conf'
cd ../downstream/word_ana
python evalution.py --ctrlname $CTRL_NAME --input_file $INFER_INDEX --dim 400
cd ../../grm

echo --------------$INFER_INDEX----------------------------------------------------------
echo --------------$CTRL_NAME-------------------------------------------