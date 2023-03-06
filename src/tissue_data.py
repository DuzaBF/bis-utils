import csv
import numpy
import scipy.constants


class TissueData(object):

    def __init__(self, filename) -> None:
        self.name = filename
        self.freq = []
        self.sigma = []
        self.eps = []
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                self.freq.append(float(row[0]))
                self.sigma.append(float(row[1]))
                self.eps.append(float(row[2]))
        self.freq = numpy.array(self.freq)
        self.sigma = numpy.array(self.sigma)
        self.eps = numpy.array(self.eps)

    def __str__(self) -> str:
        wd = 11
        rstr = self.name + "\n" + "{:>{width}} | {:>{width}} | {:>{width}}\n".format("freq [Hz]","sigma [S/m]","eps",width=wd) 
        rstr = rstr + "{:->{width}} | {:->{width}} | {:->{width}}\n".format("","","",width=wd)
        for i,_ in enumerate(self.freq):
            rstr = rstr + "{:>{width}} | {:>{width}} | {:>{width}}\n".format(self.freq[i],self.sigma[i],self.eps[i],width=wd)
        return rstr
    
    def __repr__(self) -> str:
        return str(self)

 
class TissueDataComplex(TissueData):

    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.complex_sigma = self.sigma + numpy.multiply(self.freq, self.eps * scipy.constants.epsilon_0) * 1j * 2*scipy.constants.pi 
    
    def __str__(self) -> str:
        wd = 11
        rstr = self.name + "\n{:>{width}} | {:>{width}}\n".format("freq [Hz]","sigma [S/m]",width=wd)
        for i, f in enumerate(self.freq):
            rstr = rstr + "{:>{width}} | {:>{width}}\n".format(f,"{:.3f}".format(self.complex_sigma[i]),width=wd)
        return rstr

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog = 'Tissue Data')
    parser.add_argument('filename') 
    args = parser.parse_args()

    td = TissueData(args.filename)
    print(td)

    tdc = TissueDataComplex(args.filename)
    print(tdc)
    print(super(TissueDataComplex, tdc).__str__())