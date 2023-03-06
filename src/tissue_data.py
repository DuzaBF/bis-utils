import csv
import typing
import numpy
import scipy.constants



class TabularData(object):

    def __init__(self, columns_data) -> None:
        self.columns = columns_data

    def __str__(self) -> str:
        wd = 10
        return "{:{width}} | {:{width}} | {:{width}}".format(self.columns[0][0], self.columns[0][1], self.columns[0][2], width=wd)

    def __repr__(self) -> str:
        return self.__str__()


class TabularDataComplex(TabularData):

    def __init__(self, columns_data) -> None:
        super().__init__(columns_data)

    def __str__(self) -> str:
        wd = 10
        return "{:{width}} | {:{width}}".format(self.columns[0][0], self.columns[0][1], width=wd)

    def __repr__(self) -> str:
        return self.__str__()


class TabularDataFile(object):

    def __init__(self, filename) -> None:
        self.name = filename
        _columns = [[], [], []]
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                _columns[0].append(float(row[0]))
                _columns[1].append(float(row[1]))
                _columns[2].append(float(row[2]))
        
        _columns[0] = numpy.array(_columns[0])
        _columns[1] = numpy.array(_columns[1])
        _columns[2] = numpy.array(_columns[2])

        self.data = TabularData(_columns)
        
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.__str__()


class TissueData(object):

    def __init__(self, freq, sigma, eps, name=None) -> None:
        self.name = name
        self.freq = freq
        self.sigma = sigma
        self.eps = eps

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

    def __init__(self, freq, complex_sigma=None, sigma_real=None, eps=None, name=None) -> None:
        if complex_sigma is not None:
            self.complex_sigma = complex_sigma
            sigma_real = numpy.fromiter(map(lambda x: self.get_sigma_real(x[0], x[1]), zip(freq, complex_sigma)), dtype=numpy.single)
            eps = numpy.fromiter(map(lambda x: self.get_eps(x[0], x[1]), zip(freq, complex_sigma)), dtype=numpy.single)
            super().__init__(freq=freq, sigma=sigma_real, eps=eps, name=name)
        elif (sigma_real is not None) and (eps is not None):
            super().__init__(freq=freq, sigma=sigma_real, eps=eps, name=name)
            self.complex_sigma = numpy.fromiter(map(lambda x: self.to_complex(x[0], x[1], x[2]), zip(freq, sigma_real, eps)), dtype=numpy.csingle)
        else:
            raise ValueError("Must specify complex_sigma or both sigma_real and eps")

    def __str__(self) -> str:
        wd = 11
        rstr = self.name + "\n{:>{width}} | {:>{width}}\n".format("freq [Hz]","sigma [S/m]",width=wd)
        for i, f in enumerate(self.freq):
            rstr = rstr + "{:>{width}} | {:>{width}}\n".format(f,"{:.3f}".format(self.complex_sigma[i]),width=wd)
        return rstr

    @staticmethod
    def to_complex(freq, sigma, eps) -> complex:
        return sigma + 1j * 2*scipy.constants.pi * freq * eps * scipy.constants.epsilon_0
    
    @staticmethod
    def get_sigma_real(freq, sigma) -> float:
        return sigma.real
    
    @staticmethod
    def get_eps(freq, sigma) -> float:
        return sigma.imag / (2*scipy.constants.pi * freq * scipy.constants.epsilon_0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog = 'Tissue Data')
    parser.add_argument('filename') 
    args = parser.parse_args()

    tdf = TabularDataFile(args.filename)

    td = TissueData(freq=tdf.data.columns[0], sigma=tdf.data.columns[1], eps=tdf.data.columns[2], name=args.filename)
    print(td)

    tdc = TissueDataComplex(freq=tdf.data.columns[0], sigma_real=tdf.data.columns[1], eps=tdf.data.columns[2], name=args.filename)
    print(tdc)
    print(super(TissueDataComplex, tdc).__str__())

    tdcc = TissueDataComplex(freq=tdf.data.columns[0], complex_sigma=tdc.complex_sigma, name=args.filename)
    print(tdcc)