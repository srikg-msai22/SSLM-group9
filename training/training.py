import pandas as pd
from simplet5 import SimpleT5

if __name__ == '__main__':
    train_dataset = pd.read_csv("unmodified/train_web_nlg.csv", names=['source_text', 'target_text'], header=None)
    dev_dataset = pd.read_csv("unmodified/dev_web_nlg.csv", names=['source_text', 'target_text'], header=None)
    test_dataset = pd.read_csv("unmodified/test_web_nlg.csv", names=['source_text', 'target_text'], header=None)
    print(train_dataset.head())

    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")
    model.train(train_df=train_dataset, eval_df=dev_dataset, use_gpu=False)

