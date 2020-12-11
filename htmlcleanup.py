import csv

import html2text

dsafilename = "C:\\acptdataextract\\DSAACTCMAPPING.csv"

with open(dsafilename, newline="", encoding='utf-8') as file:
    readData = [row for row in csv.DictReader(file)]
    i = 0
    for item in readData:
        readData[i]['AcptCrit'] = html2text.html2text(readData[i]['AcptCrit'])
        i += 1

def writer(header, data, filename, option):
    with open(filename, "w", newline="") as csvfile:
        if option == "write":

            movies = csv.writer(csvfile)
            movies.writerow(header)
            for x in data:
                movies.writerow(x)
        elif option == "update":
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)
        else:
            print("Option is not known")


readHeader = readData[0].keys()
writer(readHeader, readData, dsafilename, "update")