# bash run_glyph_en_sa_1_small_gat.sh
# conda activate pytorch113
export INFER_INDEX=5

# CTRL_NAME='2022-09-26-20_55_06'
CTRL_NAME=`date "+%Y-%m-%d-%H_%M_%S"`
python train.py --epochs $INFER_INDEX --ctrlname $CTRL_NAME \
    --learning_rate 1e-3 --batch_size 256 --cuda 0 --model_type 'glyph_sa_1_small_no_decoder_bn' \
    --config './experiments_en.conf' --encoder_gnn_type 'gat' --probs 0.2 0.7 0.2

python infer.py --output_file '_en_ana_'$INFER_INDEX \
    --model_file 'GRM_epoch_'${INFER_INDEX}'.pt' --ctrlname $CTRL_NAME \
    --batch_size 256 --oov_file '../downstream/word_ana/word.txt' --cuda 0 \
    --config './experiments_en.conf'
cd ../downstream/word_ana
python evalution.py --ctrlname $CTRL_NAME --input_file $INFER_INDEX --dim 400
cd ../../grm

echo --------------$INFER_INDEX----------------------------------------------------------
echo --------------$CTRL_NAME-------------------------------------------