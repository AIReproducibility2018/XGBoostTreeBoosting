filePath = "C:\\Users\Axim-\Downloads\Yahoo\Webscope_C14\Webscope_C14\Learning to Rank Challenge\ltrc_yahoo.tar\ltrc_yahoo\set2.train"
file = open(filePath+".txt", 'r')

data = []

for line in file:

    values = line.split(" ")
    label = values[0]
    data.append([label])
    nOfValues = len(values)
    index = 2

    for i in range(1, 701):
        if index < nOfValues and values[index].split(":")[0] == str(i):
            data[-1].append(float(values[index].split(":")[1]))
            index += 1
        else:
            data[-1].append(0)

parsedFile = open(filePath+"Parsed.txt", 'w')

for d in data:
    parsedFile.write(','.join(map(str, d)) + "\n")