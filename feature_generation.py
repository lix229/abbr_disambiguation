from preprocessing import *
import pandas as pd

def generate_feature(abbr, n_grams, OPTION = 0):
    """
    Generate feature from n_grams
    :param n_grams: list of n_grams
    :param OPTION: {
        0: basic n_gram feature [word1, word2,...,wordn]
        1: n_gram with direction [l-word1, l-word2,..., r-wordn]
        2: n_gram with direction and position [l-m-word1, l-m-1-word2 ,..., r-n-wordn]
    }
    :return: feature
    """
    #TODO NEEDS TESTING
    feature_list = []
    if OPTION == 0:
        total_words = []
        for sense in n_grams.keys():
            total_words+=sum(n_grams[sense], [])
        total_words_without_abbr = [word for word in total_words if word != abbr]
        unique_words = set(total_words_without_abbr)
        #///!FIXME: The abbreviation is also included in the feature
        for i, sense in enumerate(n_grams.keys()):
            for sentence in n_grams[sense]:
                word_occurrence_in_sentence = [0 for i in range(len(unique_words))]
                for word in sentence:
                    if word != abbr:
                        word_occurrence_in_sentence[list(unique_words).index(word)] += 1
                word_occurrence_in_sentence.append(i)
                feature_list.append(word_occurrence_in_sentence)
        # Convert to pd.DataFrame
        #///!FIXME: This is not working as intended
        colomns = list(unique_words)
        colomns.append("sense")
        feature = pd.DataFrame(feature_list, columns=colomns)
    elif OPTION == 1:
        total_words_without_abbr = []
        for sense in n_grams.keys():
            for i, sentence in enumerate(n_grams[sense]):
                abbr_index = sentence.index(abbr)
                left = ['l-'+ word for word in sentence[:abbr_index]]
                right = ['r-'+ word for word in sentence[abbr_index+1:]]
                n_grams[sense][i] = left + right
        for sense in n_grams.keys():
            total_words_without_abbr+= sum(n_grams[sense], [])
        unique_words = set(total_words_without_abbr)
        for i, sense in enumerate(n_grams.keys()):
            for sentence in n_grams[sense]:
                word_occurrence_in_sentence = [0 for i in range(len(unique_words))]
                for word in sentence:
                    if word != abbr:
                        word_occurrence_in_sentence[list(unique_words).index(word)] += 1
                word_occurrence_in_sentence.append(i)
                feature_list.append(word_occurrence_in_sentence)
        colomns = list(unique_words)
        colomns.append("sense")
        feature = pd.DataFrame(feature_list, columns=colomns)

    elif OPTION == 2:
        total_words_without_abbr = []
        for sense in n_grams.keys():
            for i, sentence in enumerate(n_grams[sense]):
                new_sentence = []
                for j, word in enumerate(sentence):
                    abbr_index = sentence.index(abbr)
                    distance = abs(abbr_index - j)
                    if abbr_index - j < 0:
                        new_sentence.append('l-%s-%s'%(distance, word))
                    elif abbr_index - j > 0:
                        new_sentence.append('r-%s-%s'%(distance, word))
                    else:
                        pass
                n_grams[sense][i] = new_sentence
                    
        for sense in n_grams.keys():
            total_words_without_abbr+= sum(n_grams[sense], [])
        unique_words = set(total_words_without_abbr)
        for i, sense in enumerate(n_grams.keys()):
            for sentence in n_grams[sense]:
                word_occurrence_in_sentence = [0 for i in range(len(unique_words))]
                for word in sentence:
                    if word != abbr:
                        word_occurrence_in_sentence[list(unique_words).index(word)] += 1
                word_occurrence_in_sentence.append(i)
                feature_list.append(word_occurrence_in_sentence)
        colomns = list(unique_words)
        colomns.append("sense")
        feature = pd.DataFrame(feature_list, columns=colomns)
    else:
        raise ValueError("Invalid option")
    return feature


if __name__ == '__main__':
    '''
    abbr: abbreviation as a string
    n: n in n-gram
    OPTION: {
        0: basic n_gram feature [word1, word2,...,wordn]
        1: n_gram with direction [l-word1, l-word2,..., r-wordn]
        2: n_gram with direction and position [l-m-word1, l-m-1-word2 ,..., r-n-wordn]
    }
    '''

    #* TESTING
    #! OPTIONS HERE
    abbr = "CVA"
    for n in range(2, 6):
        for OPTION in range(3):
            sample = get_n_gram(abbr, n)
            feature_df = generate_feature(abbr, sample, OPTION)
            label = feature_df["sense"]
            label.to_csv("./features/label_%s_%s.csv"%(n, OPTION), index=False)
            feature_df.drop("sense", axis=1, inplace=True)
            feature_df.to_csv("./features/feature_%s_%s.csv"%(n, OPTION), index=False)

    # sample = get_n_gram(abbr, n)
    # feature_df = generate_feature(abbr, sample, OPTION)
    # label = feature_df["sense"]
    # label.to_csv("./features/label_option_%s.csv"%OPTION, index=False)
    # feature_df.drop("sense", axis=1, inplace=True)
    # feature_df.to_csv("./features/feature_option_%s.csv"%OPTION, index=False)
