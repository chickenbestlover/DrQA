import glob
import errno
import csv
import matplotlib.pyplot as plt
import matplotlib.text as text
import os




netType='selfattn2'
path = './Log_files/'+netType+'/*.log'

txtFiles= glob.glob(path)
#csvFile = open('results.csv', 'w', encoding='utf-8',newline='')
#csvWriter = csv.writer(csvFile)
EMs = list()
F1s = list()
fileNames = list()
for fileName in txtFiles:
    try:
#        csvFileName = fileName[12:]
#        csvFile = open('csv_files/'+netType+'/'+csvFileName+'.csv', 'w', encoding='utf-8', newline='')
#        csvWriter = csv.writer(csvFile)
#        csvWriter.writerow([fileName])
#        csvWriter.writerow(['EM','F1'])
        EM = list()
        F1 = list()
        print(fileName)
        with open(fileName) as f:

            for line in f:
                words  = line.split()
                if len(words)>2:
                    if words[2] =='dev':
#                       csvWriter.writerow([words[4],words[6]])
                        EM.append(float(words[4]))
                        F1.append(float(words[6]))
                        #print(words)
            #input()
            fileNames.append(fileName)
        EMs.append(EM)
        F1s.append(F1)

#        csvFile.close()
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
MAX_EPOCH=150
EMs = [(EM + [0] * MAX_EPOCH)[:MAX_EPOCH] for EM in EMs]
F1s = [(F1 + [0] * MAX_EPOCH)[:MAX_EPOCH] for F1 in F1s]


plt.figure(figsize=(10,6))
plt.rc('grid', linestyle=":", color='black')
plt.grid()



plt.grid()
#plt.ylim([60,82])

j=0
for i,(F1,EM) in enumerate(zip(F1s,EMs)):
    fileName = fileNames[i].split('output_')[1].split('.log')[0]

    plt.subplot(2, 1, 1)
    plt.plot(range(1,MAX_EPOCH+1),F1,label=fileName)
    plt.ylim([79.0, 82.0])
    plt.ylim([77.0, 82.0])
    plt.ylim([75.0, 82.0])

    #plt.ylim([50.0, 82.0])

    plt.ylabel('F1',fontsize=20)
    plt.legend()
    F1max = max(F1)
    xmax= F1.index(F1max)+1
    #plt.annotate('{:.5}'.format(str(F1[F1.index(F1max)])), fontsize=11,xy=(xmax, F1max), xytext=(xmax, F1max + 0.5 + 0.1 * (j % 8)),arrowprops=dict(facecolor='black', shrink=0.05,width=0.1))
    plt.annotate('{:.5}'.format(str(F1[F1.index(F1max)])), fontsize=11,xy=(xmax, F1max), xytext=(xmax, F1max + 0.5 + 0.2 * (j % 8)),
                 arrowprops=dict(facecolor='black', shrink=0.05,width=0.1))

    plt.subplot(2, 1, 2)
    plt.plot(range(1,MAX_EPOCH+1),EM,label=fileName)
    #plt.xlabel('epoch')
    plt.ylabel('EM',fontsize=20)
    plt.legend()

    #plt.annotate('{:.5}'.format(str(EM[xmax-1])),fontsize=11, xy=(xmax, EM[xmax-1]), xytext=(xmax, EM[xmax-1]  + 0.5+ 0.1*(j%8)),arrowprops=dict(facecolor='black', shrink=0.05,width=0.1))
    plt.annotate('{:.5}'.format(str(EM[xmax-1])),fontsize=11, xy=(xmax, EM[xmax-1]), xytext=(xmax, EM[xmax-1]  + 0.5+ 0.5*(j%6)),
               arrowprops=dict(facecolor='black', shrink=0.05,width=0.1))

    plt.ylim([69.0, 72.0])
    plt.ylim([67.0, 72.0])
    plt.ylim([65.0, 72.0])

    #plt.ylim([50.0, 72.0])

    plt.xlabel('epoch',fontsize=20)
    j+=1
    if xmax < 50:
        print(fileName)
#plt.ylim([50,100])


plt.subplots_adjust(hspace=0.5)
savePath='./results/'+netType
if not os.path.exists(savePath):
    os.makedirs(savePath)

plt.savefig(savePath+'/'+netType+'.png')
plt.show()
