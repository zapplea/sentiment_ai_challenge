import pickle

with open('/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl', 'rb') as f:
    merged_train_review, merged_train_char, train_attr_label,\
    train_senti_label, attribute_dic, new_word_dic, new_wordsVec, char_vecs = pickle.load(f)
    print(char_vecs.shape)