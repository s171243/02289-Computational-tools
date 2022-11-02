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
        is_correct = data_row[6] == "True"
        if data_row[7].isdigit() and data_row[13].isdigit():
            yield data_row[1], (int(data_row[7]), int(data_row[13]), is_correct, data_row[3])

    def reducer(self, key, values):
        countt = 0
        val = []
        levels = []
        correct = []
        problems = set()
        for v, level, is_correct, upid in values:
            countt += 1
            val.append(v)
            levels.append(level)
            correct.append(is_correct)
            problems.add(upid)
        yield key, (sum(val) / countt, sum(levels) / countt, sum(correct) / countt, countt, len(problems), max(levels))


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
