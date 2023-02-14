import os
from re import match

# Change path to where the script is located.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def fetch_lines(abbr):
    """
    Fetches all lines that start with the abbreviation.
    :param abbr: abbreviation as a string
    :return: list of lines
    """
    pattern = abbr + '|'
    with open('./data/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', 'r', encoding='cp1252') as f:
        lines = f.readlines()
    return [l.strip('\n') for l in lines if l.startswith(pattern)]

def get_n_gram(abbr, n = 2):
    """
    Returns a list of n-grams of the abbreviation.
    :param abbr: abbreviation as a string
    :param n: n in n-gram
    :return: list of n-grams
    """
    lines = fetch_lines(abbr)
    # field[0]:= abbr, field[1]:= expansion, field[6]:= [sentences]
    lines_fields = [l.split('|') for l in lines]
    range = n - 1
    # initialize result
    # result := {sense1: [n-gram1, n-gram2, ...]}
    result =  dict()
    for line in lines_fields:
        result[line[1]] = []
    # Remove empty sentences and sentences that don't contain the abbreviation.
    # Contains often more than 1 sentence. Each sentence split by ' '.
    #///!FIXME: REMOVE DECOS ARROUND KEYWORDS like (IT)
    for line in lines_fields:
        line[6] = [sentence.split() for sentence in line[6].split('.') if sentence != '']
        sentences_with_abbr = []
        for sentences in line[6]:
            for i, word in enumerate(sentences):
                sentences[i] = word.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(':', '').replace(';', '').replace(',', '')
            if abbr in sentences:
                sentences_with_abbr.append(sentences)
        line[6] = sentences_with_abbr
        for sentence in line[6]:
            abbr_index = None
            try:
                abbr_index = sentence.index(abbr)
            except ValueError:
                print("The abbreviation %s is not in the sentence %s, that is wield ;( \n" %(abbr, sentence))
                continue
            lower_bound = abbr_index - range if abbr_index - range >= 0 else 0
            upper_bound = abbr_index + range if abbr_index + range < len(sentence) else len(sentence) - 1
            n_gram = sentence[lower_bound:upper_bound+1]
            #///!TESTME Unify anonymized dates and Zip codes
            # HACK r'\d{4}' as "_%#DATE#%_"
            #? MAYBE?
            for i, word in enumerate(n_gram):
                #? Could use some better regex
                if match("_%#[0-9]{4}#%_|_%#DDMM[0-9]{4}?#%_|_%#MMDD#%_|_%#MM[0-9]{4}#%_|_%#DD#%_|_%#MMDD[0-9]{4}?#%_|_%#MM#%_|^[0-9]{4}", word):
                    n_gram[i] = "_%#DATE#%_"
                elif match("_%#[0-9]{5}#%_|_%#ZIP#%_", word):
                    n_gram[i] = "_%#ZIP#%_"
            result[line[1]].append(n_gram)
    return result

if __name__ == '__main__':
    #? TESTING
    sample = get_n_gram("CVA", 5)
    print(sample)
    # print(re.match("_%#[0-9]{4}#%_|_%#DDMM[0-9]{4}#%_|_%#MMDD#%_|_%#MM#%_|_%#DD#%_", "_%#MMDD#%_"))
