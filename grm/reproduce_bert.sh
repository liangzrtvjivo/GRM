# bash run_bert.sh
export INFER_INDEX=5
lr=1e-3

CTRL_NAME='grm_bert_128_1e-3'
echo --------------$CTRL_NAME-------------------------------------------

# python infer.py --output_file '_CoNLL_'$INFER_INDEX \
#     --model_file 'GRM_epoch_'${INFER_INDEX}'.pt' --ctrlname $CTRL_NAME \
#     --batch_size 128 --oov_file '../downstream/bert/bert_ner/output_oov/words_conll_oov.txt' --cuda 1 \
#     --config './experiments_bert.conf'
# echo _CoNLL_------------------------------------------------------------------------

cd ../downstream/bert/bert_ner
python main.py --ctrlname $CTRL_NAME --input_file $INFER_INDEX --num_train_epochs 20\
    --input_name '_CoNLL_' --add_new True \
    --data_dir './evaluation_conll'
echo RESULT-------_CoNLL_------------------------------------------------------------------------
cd ../../../grm
