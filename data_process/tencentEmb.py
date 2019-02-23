def read(fname):
    with open(fname,'r') as f:
        count = 0
        for line in f:
            count+=1
            print(line)
            if count>=2:
                exit()

if __name__ == "__main__":
    fname = "/datastore/liu121/wordEmb/tencent_cn/Tencent_AILab_ChineseEmbedding.txt"
    read(fname)