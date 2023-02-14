import os, sys
from statistics import stdev
from math import floor

if __name__ == '__main__':
    result = {}
    with open('./data/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', 'r', encoding='cp1252') as f:\
        # A glimpse of the data
        # head = [next(f) for x in range(30)]
        lines = f.readlines()
    for l in lines:
        l = l.split('|')
        result[l[0]] = result.get(l[0], {l[1]: 0})
        result[l[0]][l[1]] = result[l[0]].get(l[1], 0) + 1
    print("Total number of abbreviations: %s\n" %len(result))
    result_greater_than_one = {}
    for abbr, values in result.items():
        if len(values) >1:
            result_greater_than_one[abbr] = values
    sorted_by_stdev = dict(sorted(result_greater_than_one.items(), key=lambda x: (-floor(stdev(x[1].values())), len(x[1].values())) if len(x[1]) > 1 else 0, reverse=True))
    # sorted_by_stdev = dict(sorted(result.items(), key=lambda x: (floor(stdev(x[1].values()))) if len(x[1]) > 1 else 0, reverse=True))
    with open ('./data/stats.txt', 'w') as stats:
        for abbr, values in sorted_by_stdev.items():
            sense_std = stdev(values.values()) if len(values) > 1 else 0
            stats.write("Abbreviation: %s\n" %abbr)
            stats.write("Standard deviation of senses: %s\n" %sense_std)
            stats.write("Number of expansions: %s\n" %len(values))
            stats.write("Expansions: %s\n\n" %values)
        


        