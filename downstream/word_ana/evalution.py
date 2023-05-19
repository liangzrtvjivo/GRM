import numpy as np
import argparse
import gensim

parser = argparse.ArgumentParser(description='Word Analogy')
parser.add_argument('--ctrlname', type=str, default='2022-05-29-16_35_29')
parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--dim', type=int, default=400)

def most_similar_3cosadd(dword1, dword2, dword3, w2id, id2w, wordVec, dim):
    wid1, wid2, wid3 = w2id[dword1], w2id[dword2], w2id[dword3]
    gamma = np.zeros([len(w2id),dim])
    for i in range (len(id2w)):
        words = id2w[i]
        v0 = wordVec[words]
        rs = [float(s) for s in v0]
        wvec1 = np.array(rs)
        gamma[i] = wvec1

    normss = np.linalg.norm(gamma, axis = 1, keepdims = True)
    #     #print(normss.shape)
    #     #print(normss)
    gamma = gamma/normss
    wvec = gamma[wid3] - gamma[wid1] + gamma[wid2]
    temp_gamma = gamma.copy()
    temp_gamma[wid1] = temp_gamma[wid2] = temp_gamma[wid3] = 0
    cos_distances = np.inner(wvec, temp_gamma)
    return id2w[np.argmax(cos_distances)]

def most_similar_3cosmul(dword1, dword2, dword3, w2id, id2w, wordVec, dim):
    wid1, wid2, wid3 = w2id[dword1], w2id[dword2], w2id[dword3]
    wid1, wid2, wid3 = w2id[dword1], w2id[dword2], w2id[dword3]
    gamma = np.zeros([len(w2id),dim])
    for i in range (len(id2w)):
        words = id2w[i]
        v0 = wordVec[words]
        rs = [float(s) for s in v0]
        wvec1 = np.array(rs)
        gamma[i] = wvec1

    normss = np.linalg.norm(gamma, axis = 1, keepdims = True)
    #     #print(normss.shape)
    #     #print(normss)
    gamma = gamma/normss
    wvec1, wvec2, wvec3 = gamma[wid1], gamma[wid2], gamma[wid3]
    temp_gamma = gamma.copy()
    temp_gamma[wid1] = temp_gamma[wid2] = temp_gamma[wid3] = 0
    cos_dst1, cos_dst2, cos_dst3 = np.inner(wvec1, temp_gamma), np.inner(wvec2, temp_gamma), np.inner(wvec3, temp_gamma)
    cos_dst1 = (cos_dst1+1)/2
    cos_dst2 = (cos_dst2+1)/2
    cos_dst3 = (cos_dst3+1)/2
    similarity = cos_dst3 * cos_dst2 / (cos_dst1 + 1e-3)
    return id2w[np.argmax(similarity)]
   
def measure_ana(wordVecfile, dim):
    test_pt = 'analogy_dataset.txt'
    wordemb = {}
    w2id = {}
    id2w = {}
    vocab_pt = wordVecfile
    inde = 0
    for l in open(vocab_pt,encoding='UTF-8'):
        #print(l)
        ws = l.strip().split()
        if len(ws)<10:
            continue
        if ws[0] not in wordemb:
            wordemb[ws[0]] = ws[1:]
            #print(len(wordemb[ws[0]]))
            w2id[ws[0]] = inde
            id2w[inde] = ws[0]
            inde +=1


    n_total = n_correct = 0
    n_total1 = n_correct1 = 0
    for line in open(test_pt,encoding='UTF-8'):
        ws = line.strip().split()
        if len(ws)!=4:
            continue
        #print(ws)
        if ws[0] in wordemb and ws[1] in wordemb and ws[2] in wordemb and ws[3] in wordemb:
            n_total = n_total + 1
            n_total1 = n_total1 + 1
            if most_similar_3cosadd(ws[0], ws[1], ws[2], w2id, id2w, wordemb, dim) == ws[3]:
                n_correct = n_correct + 1
            if most_similar_3cosmul(ws[0], ws[1], ws[2], w2id, id2w, wordemb, dim) == ws[3]:
                n_correct1 = n_correct1 + 1
            '''else:
                print(ws[0], ws[1], ws[2], ws[3])
                print(most_similar_3cosadd(ws[0], ws[1], ws[2]))'''

    #print(n_correct, n_total, n_correct/n_total)
    #return (n_correct/n_total)
    print("cosadd:",n_correct, n_total, n_correct / n_total)
    print("cosmul:",n_correct1, n_total1, n_correct1 / n_total1)

if __name__ == '__main__':
    args = parser.parse_args()
    vectors_path = '../../grm/result/resultfile_' + args.ctrlname  + '/oov_result_en_ana_'+ args.input_file +'.txt'
    # measure_ana(vectors_path, args.dim)
    model = gensim.models.KeyedVectors.load_word2vec_format(vectors_path)
    test_pt = 'analogy_dataset.txt'
    result = model.evaluate_word_analogies(test_pt)
    print("result",result[0])