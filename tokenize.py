from transformers import AutoTokenizer
import argparse
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description="Tokenize data")
parser.add_argument('--tokenizer', default='roberta-base',
                    help="path to tokenizer")
parser.add_argument('--input', default='para/en-pt.pt',
                    help="path to text file")
parser.add_argument('--output', default='align/en-pt.pt',
                    help="path to tokenized output")

params = parser.parse_args()



def main():
    tokenizer = AutoTokenizer.from_pretrained(
        params.tokenizer)

    with open(params.input, 'r') as reader, open(params.output, 'w') as writer:
        sentences = reader.readlines()
        for sentence in tqdm(sentences):
            encoded = tokenizer.tokenize(sentence)
            writer.write(' '.join(encoded) + '\n')


if __name__ == '__main__':
    main()


