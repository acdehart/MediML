import re
import string


def clean_string(text):
    # Part 1
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('!', '', text)
    text = re.sub(',', '', text)


    # Part 2
    text = re.sub('[`".]', '', text)
    text = re.sub('\n\t', '', text)

    # All text lower,
    # remove punctuation
    # remove numerical values
    # remove non speech (eg '\n')
    # tokenize text
    # remove stop words

    # stemming/ lemmazation
    # parts of speech tagging
    return text