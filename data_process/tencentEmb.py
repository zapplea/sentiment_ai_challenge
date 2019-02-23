import numpy as np
import pickle

def read(fname):
    with open(fname,'r') as f:
        count = 0
        id2word_dic = {}
        word2id_dic = {}
        wordsVec_ls = []
        for line in f:
            if count ==0:
                print(line)
                continue
            count+=1
            line = line.replace('\n','')
            line_ls = line.split(' ')
            word = line_ls[0]
            vec = np.array(line_ls[1:]).astype('float32')
            assert word not in word2id_dic
            word2id_dic[word]=count-1
            id2word_dic[count-1]=word
            wordsVec_ls.append(vec)
        print('wordsVec shape: ',np.shape(np.array(wordsVec_ls).astype('float32')))
        print(word2id_dic)
    return {'wordsVec':np.array(wordsVec_ls).astype('float32'),
            'id2word':id2word_dic,
            'word2id':word2id_dic}

def write(fname,data):
    with open(fname,'wb') as f:
        pickle.dump(data,f,protocol=4)

if __name__ == "__main__":
    fname = "/datastore/liu121/wordEmb/tencent_cn/Tencent_AILab_ChineseEmbedding.txt"
    data = read(fname)
    fname="/datastore/liu121/wordEmb/tencent_cn/tencent_wordsVec.pkl"
    write(fname,data)