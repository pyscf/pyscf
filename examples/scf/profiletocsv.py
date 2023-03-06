openfile = 'lih-profile.txt'
savefile = 'lih-profile.csv'

writelines = []

with open(openfile) as of:
    content = of.readlines()

    for line in content:

        line = line.replace(".", ",")
        string = ""
        sepcont = line.split()

        print(sepcont) # 1, 4, 9

        string += sepcont[1] + ";" + sepcont[4] + ";" + sepcont[9] + ";"
        writelines.append(string)

with open(savefile, 'w') as sf:
    for l in writelines:
        sf.write(l + "\n")

