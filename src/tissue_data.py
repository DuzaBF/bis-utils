import csv
import numpy
import argparse


class TissueData(object):

    def __init__(self, filename) -> None:
        self.name = filename
        self.freq = []
        self.sigma = []
        self.eps = []
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                self.freq.append(row[0])
                self.sigma.append(row[1])
                self.eps.append(row[2])
        self.freq = numpy.array(self.freq)
        self.sigma = numpy.array(self.sigma)
        self.eps = numpy.array(self.eps)

    def __str__(self) -> str:
        width = 11
        rstr = self.name + "\n" + "{:>{width}} | {:>{width}} | {:>{width}}\n".format("freq [Hz]","sigma [S/m]","eps",width=width) 
        rstr = rstr + "{:->{width}} | {:->{width}} | {:->{width}}\n".format("","","",width=width)
        for i,_ in enumerate(self.freq):
            rstr = rstr + "{:>{width}} | {:>{width}} | {:>{width}}\n".format(self.freq[i],self.sigma[i],self.eps[i],width=width)
        return rstr
    
    def __repr__(self) -> str:
        return str(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Tissue Data')
    parser.add_argument('filename') 
    args = parser.parse_args()

    td = TissueData(args.filename)

    print(td)