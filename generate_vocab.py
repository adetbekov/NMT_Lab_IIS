import collections
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def generate(input_file, output_file):
    fin = open(input_file, "r")
    dictionary, reverse_dictionary = build_dataset(fin.read().split())
    fout = open(output_file, "w")
    fout.write("\n".join(dictionary.keys()))
    fin.close()
    fout.close()

generate("train.en", "vocab.en")
generate("train.kz", "vocab.kz")
