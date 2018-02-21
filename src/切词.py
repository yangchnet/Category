import os
import jieba
path = os.path.abspath('..')

allFile = []
def gci(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
          gci(fi_d)
        else:
          # print(os.path.join(filepath,fi_d))
            print(fi_d)
            allFile.append(fi_d)

gci('..\data')


for i in range(0, len(allFile)-1):
    temp = ''
    with open(allFile[i], 'r+', encoding = 'utf-8') as f:
        for line in f.readlines():
            writeback = jieba.cut(line)
            temp += " ".join(writeback)
        f.close()
    with open(allFile[i], 'w', encoding = 'utf-8') as f:
        f.write(temp.replace("\n", ' '))
        f.close()

