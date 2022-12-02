import os
import sys

import pandas as pd
from mrjob.job import MRJob
from mrjob.step import MRStep
from mr3px.csvprotocol import CsvProtocol
from data_preprocessing import preprocess_df
from feature_categorization import Ex_features


class MRWordFrequencyCount(MRJob):
    OUTPUT_PROTOCOL = CsvProtocol

    def __init__(self, args=None):
        super().__init__(args)
        self.content = None

    def with_numpy(self, df, col, col2, val=0):
        return df[col].to_numpy()[df[col2].to_numpy() == val].item()

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init, mapper=self.mapper, reducer=self.reducer)
        ]

    def mapper_init(self):
        global current_cwd
        self.content = pd.read_csv(current_cwd + "/data/csv_files/Info_Content.csv")
        features = Ex_features()
        self.content = preprocess_df(self.content, features)
        # self.content[["learning_stage", "difficulty"]] = ordinal_encode(self.content, ["learning_stage", "difficulty"])
        sys.stderr.write(current_cwd.encode())

    def mapper(self, _, line):
        data_row = line.split(",")
        uuid = data_row[1]
        ucid = data_row[2]
        sys.stderr.write((",".join(data_row)).encode())

        is_correct = data_row[6] == "True"
        if data_row[7].isdigit() and data_row[13].isdigit():
            learning_stage = self.with_numpy(self.content, "learning_stage", "ucid", ucid)
            difficulty = self.with_numpy(self.content, "difficulty", "ucid", ucid)
            yield uuid, (int(data_row[7]), int(data_row[13]), is_correct, data_row[3],
                                learning_stage, difficulty, int(data_row[13]))

    def reducer(self, key, values):
        countt = 0
        val = []
        levels = []
        difficulty = []
        stages = []
        correct = []
        hints = []
        problems = set()
        for v, level, is_correct, upid, stage, diff, hint in values:
            countt += 1
            val.append(v)
            levels.append(level)
            correct.append(is_correct)
            problems.add(upid)
            difficulty.append(diff)
            stages.append(stage)
            hints.append(hint)
        yield key, (key, countt, sum(val) / countt, sum(levels) / countt, sum(correct) / countt, len(problems) / countt,
                    max(levels), sum(stages) / countt, sum(difficulty) / countt, sum(hints)/countt)


current_cwd = os.getcwd()

if __name__ == '__main__':
    users = pd.read_csv(current_cwd + "/data/csv_files/Info_UserData.csv")
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
