import glob
import errno
import csv
path = './Log_files/*.log'

txtFiles= glob.glob(path)
#csvFile = open('results.csv', 'w', encoding='utf-8',newline='')
#csvWriter = csv.writer(csvFile)

for fileName in txtFiles:
    try:
        csvFileName = fileName[12:]
        csvFile = open('csv_files/'+csvFileName+'.csv', 'w', encoding='utf-8', newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow([fileName])
        csvWriter.writerow(['EM','F1'])
        with open(fileName) as f:
            for line in f:
                words  = line.split()
                if len(words)>2:
                    if words[2] =='dev':
                        csvWriter.writerow([words[4],words[6]])
        csvFile.close()
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise