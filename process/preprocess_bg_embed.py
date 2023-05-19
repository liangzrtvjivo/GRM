import pyhocon
import argparse
import unicodedata

# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

""" Tokenization classes (It's exactly the same code as Google BERT code """

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def tokenize(text, piece_list):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """
        unk_token = "[UNK]"
        max_input_chars_per_word = 100
        text = text.lower()
        token = _run_strip_accents(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(_run_split_on_punc(token))

        # text = convert_to_unicode(text)

        output_tokens = []
        for token in split_tokens:
            chars = list(token)
            if len(chars) > max_input_chars_per_word:
                output_tokens.append(unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in piece_list:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def word_piece(word_map, config):
    wiki_piece2id_r = open(config['file_path.wiki_piece2id'],"r",encoding='utf-8')
    wiki_wid2pieceid_w = open(config['file_path.wiki_wid2pieceid'],"w",encoding='utf-8')
    wordpiece_map = {}
    for line in wiki_piece2id_r:
        lines = line.strip().split()
        piece = lines[1]
        pieceid = lines[0]
        wordpiece_map[piece] = pieceid

    for word,word_id in word_map.items():
        pieces = tokenize(word, wordpiece_map)
        pi_len = len(pieces)
        if pi_len==1:
            pi_idx = 4
        else:
            pi_idx = 5
        for piece in pieces:
            p_id = wordpiece_map[piece] if piece in wordpiece_map else wordpiece_map['[UNK]']
            wiki_wid2pieceid_w.write(f"{word_id} {p_id} {pi_idx}\n")
            pi_idx+=1

def word_char(word_map, config):
    wiki_piece2id_r = open(config['file_path.wiki_piece2id'],"r",encoding='utf-8')
    wiki_wid2char_w = open(config['file_path.wiki_wid2char'],"w",encoding='utf-8')
    wordpiece_map = {}
    for line in wiki_piece2id_r:
        lines = line.strip().split()
        piece = lines[1]
        pieceid = lines[0]
        wordpiece_map[piece] = pieceid
    for word,word_id in word_map.items():
        char_seq = list(word)
        for char in char_seq:
            c_id = wordpiece_map[char] if char in wordpiece_map else wordpiece_map['[UNK]']
            wiki_wid2char_w.write(f"{word_id} {c_id} 2\n")

def word2id_feature(args,config):
    bg_emb_file_r = open(args.bg_emb_file,"r",encoding='utf-8')
    wiki_word2id_w = open(config['file_path.wiki_word2id'],"w",encoding='utf-8')
    wiki_wid2feature_w = open(config['file_path.wiki_wid2feature'],"w",encoding='utf-8')

    word_map = {}
    index = -1
    for line in bg_emb_file_r:
        if index<0:
            index+=1
            continue
        lines = line.strip().split(" ",1)
        word = lines[0]
        feature = lines[1]
        word_map[word] = index
        wiki_word2id_w.write(f"{index} {word}\n")
        wiki_wid2feature_w.write(f"{index} {feature}\n")
        index+=1
    return word_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The args of preprocessing background embeddings')
    parser.add_argument('--config', type=str, default='../grm/experiments_en.conf')
    parser.add_argument('--bg_emb_file', type=str, required=True)
    args = parser.parse_args()
    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)
    word_map = word2id_feature(args,config)
    word_piece(word_map, config)
    word_char(word_map, config)
    print("DONE!!!")