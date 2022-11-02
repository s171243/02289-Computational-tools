import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRWordFrequencyCount(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer)
        ]

    def mapper(self, _, line):
        data_row = line.split(",")
        if data_row[7].isdigit():
            yield data_row[1], (int(data_row[7]))

    def reducer(self, key, values):
        countt = 0
        val = []
        for v in values:
            countt += 1
            val.append(v)
        yield (key, sum(val) / countt)


if __name__ == '__main__':
    MRWordFrequencyCount.run()


"""class MRWordFrequencyCount(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]

    def mapper(self, _, line):
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()"""
