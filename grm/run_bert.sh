# bash run_bert.sh
export INFER_INDEX=5
lr=1e-3

CTRL_NAME=`date "+%Y-%m-%d-%H_%M_%S"`
echo --------------$CTRL_NAME-------------------------------------------
python train.py --epochs $INFER_INDEX --ctrlname $CTRL_NAME \
--learning_rate $lr --batch_size 128 --cuda 1 --model_type 'glyph_sa_1_small_bn' \
--config './experiments_bert.conf' --encoder_gnn_type 'gat' --probs 0.2 0.7 0.2

python infer.py --output_file '_CoNLL_'$INFER_INDEX \
    --model_file 'GRM_epoch_'${INFER_INDEX}'.pt' --ctrlname $CTRL_NAME \
    --batch_size 128 --oov_file '../downstream/bert/bert_ner/output_oov/words_conll_oov.txt' --cuda 0 \
    --config './experiments_bert.conf'
echo _CoNLL_------------------------------------------------------------------------

cd ../downstream/bert/bert_ner
python main.py --ctrlname $CTRL_NAME --input_file $INFER_INDEX --num_train_epochs 20\
    --input_name '_CoNLL_' --add_new True \
    --data_dir './evaluation_conll'
echo RESULT-------_CoNLL_------------------------------------------------------------------------
cd ../../../grm
