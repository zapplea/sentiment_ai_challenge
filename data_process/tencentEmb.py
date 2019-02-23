def read(fname):
    with open(fname,'r') as f:
        for line in f:
            print(line)
            exit()

if __name__ == "__main__":
    fname = "/datastore/liu121/wordEmb/tencent_cn/Tencent_AILab_ChineseEmbedding.txt"
    read(fname)