import csv
import pandas as pd
from simplet5 import SimpleT5
import nltk

if __name__ == '__main__':
    train_dataset = pd.read_csv("unmodified/train_web_nlg.csv", names=['source_text', 'target_text'], header=None)
    dev_dataset = pd.read_csv("unmodified/dev_web_nlg.csv", names=['source_text', 'target_text'], header=None)
    test_dataset = pd.read_csv("unmodified/test_web_nlg.csv", names=['source_text', 'target_text'], header=None)

    model = SimpleT5()
    model.load_model("t5", "epoch0_unmodified_t5-small", use_gpu=False)
    total = 0
    ctr = 0

    dataforsrik = open('resultsdata.csv', 'w', encoding="utf-8", newline='')
    writer = csv.writer(dataforsrik)
    writer.writerow(['original text', 'original tuple', 'model predicted tuple', 'BLEU score'])
    for source, target in zip(test_dataset['source_text'], test_dataset['target_text']):
        model_prediction = model.predict(source)
        BLEU = nltk.translate.bleu_score.sentence_bleu(model_prediction, target)
        total += BLEU
        ctr += 1
        to_write = [source, target, model_prediction, BLEU]
        writer.writerow(to_write)
        print(ctr)
    dataforsrik.close()
    print(total/ctr)

