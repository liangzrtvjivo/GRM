# Graph-based Relation Mining for Context-free Out-of-Vocabulary Word Embedding Learning
Code for [Graph-based Relation Mining for Context-free Out-of-Vocabulary Word Embedding Learning]
GRM is accepted by ACL2023 main conference as a long paper.


# The usage of GRM

## Set up Environment
Set up the environment via "requirements.txt". Here we use python3.9. 

## Data Preparation
All the word embedding files for inputting require the word2vec format (binary = False).
Note that please take care of if experiments conf file is correct, you can modify it by --config parameter.
We provide "grm/experiments_en.conf" and "grm/experiments_bert.conf" for the training of Word2Vec and BERT respectively.

### Wordpiece Data (OPTIONAL)
We provide the wordpiece file, including wordpiece2id file (fixed) and wordpiece feature file, of word2vec model and BERT model, which generate followed by the setting mentioned in our paper.

If you want to replace it with your own wordpiece file, please offer the corresponding wordpiece embeddings and follow these steps:
```python
cd process
python preprocess_piece_embed.py --piece_emb_file <input_background_word_embeddings> --config <experiments conf>
```

### Word Data
Generate the corresponding preprocessing files for the input background word embeddings.

For Word2Vec model, download the Word2Vec background word embeddings provide by [Kabbach & Gulordava & Herbelot, 2019]
```bash
wget backup.3azouz.net/gensim.w2v.skipgram.model.7z
```
, and change it into the word2vec format (binary = False).
For BERT model, we use the whole words in BERT pre-trained embedding for model training.

Or you can choose your own backround word embeddings if you like.
```python
cd process
python preprocess_bg_embed.py --bg_emb_file <input_background_word_embeddings> --config <experiments conf>
```

### Synonym Data (OPTIONAL)
We also provide the word synonym file of word2vec model and BERT model.

If you replace the background word embeddings in previous step, you can regenerate the synonym file.
```python
cd process
python preprocess_syn.py --config <experiments conf>
```

## Train and Infer

For Word2Vec model,
```bash
cd grm
bash run_w2v.sh
```

For BERT model,
```bash
cd grm
bash run_bert.sh
```

## Ablation
### w/o Readout
```bash
cd grm
bash run_w2v_no_readout.sh
```
### w/o SA
```bash
cd grm
bash run_w2v_no_sa.sh
```
### w/o mask
```bash
cd grm
bash run_w2v_no_mask.sh
```
### w/o PosE
```bash
cd grm
bash run_w2v_no_posE.sh
```

# Reference
[1] Alexandre Kabbach, Kristina Gulordava, and Aurélie Herbelot. 2019. Towards incremental learning of word embeddings using context informativeness. In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28 - August 2, 2019, Volume 2: Student Research Workshop, pages 162–168. Association for Computational Linguistics.
[2] Lihu Chen, Gaël Varoquaux, and Fabian M. Suchanek. 2022. Imputing out-of-vocabulary embeddings with LOVE makes languagemodels robust with little cost. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 3488–3504. Association for Computational Linguistics.