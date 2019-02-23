def read(fname):
    with open(fname,'r') as f:
        count = 0
        for line in f:
            if count ==0:
                print(line)
            count+=1
            line = line.replace('\n','')
            print(repr(line))
            line_ls = line.split(' ')
            print(line_ls)
            if count>=2:
                exit()

if __name__ == "__main__":
    fname = "/datastore/liu121/wordEmb/tencent_cn/Tencent_AILab_ChineseEmbedding.txt"
    read(fname)