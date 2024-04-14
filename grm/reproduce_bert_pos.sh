# bash run_bert_pos.sh
export INFER_INDEX=5

CTRL_NAME='grm_bert_64_5e-4'
echo --------------$CTRL_NAME-------------------------------------------

# python infer.py --output_file '_CoNLL_'$INFER_INDEX \
#     --model_file 'GRM_epoch_'${INFER_INDEX}'.pt' --ctrlname $CTRL_NAME \
#     --batch_size 128 --oov_file '../downstream/bert/bert_ner/output_oov/words_conll_oov.txt' --cuda 1 \
#     --config './experiments_bert.conf'
# echo _CoNLL_------------------------------------------------------------------------

cd ../downstream/bert/bert_pos
python main.py --ctrlname $CTRL_NAME --input_file $INFER_INDEX --num_train_epochs 20\
    --input_name '_ud_1.4_' --add_new True \
    --data_dir 'ud_1.4/ud_en_1.4'
echo RESULT-------_ud_1.4_------------------------------------------------------------------------
cd ../../../grm
