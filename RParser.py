filePath = "D:\Development\MasterProject\DataSets\Higgs\HIGGS.csv\HIGGSTenth.csv"
file = open(filePath, 'r')
train = open("D:\Development\MasterProject\DataSets\Higgs\HIGGS.csv\\train.csv", 'w')

for line in file:
    lineArray = line.split(",")
    data = list(reversed(lineArray))
    data = map(lambda s: s.strip(), data)
    train.write(','.join(map(str, data)) + "\n")
