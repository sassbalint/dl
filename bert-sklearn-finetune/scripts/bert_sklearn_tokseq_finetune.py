"""
Token sequence classification using BERT in scikit-learn.
Based on this notebook:
https://colab.research.google.com/drive/1-wTNA-qYmOBdSYG7sRhIdOrxcgPpcl6L
Thanks to Charles Nainan for: https://github.com/charles9n/bert-sklearn
which allows a very convenient API,
and is a great piece of code in itself, indeed.
"""


import argparse
import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
import torch

from bert_sklearn import BertTokenClassifier


def main():
    """Do the thing."""
    args = get_args()


    # --- check
    print('\n--- check')

    print()
    print('check pytorch version:', torch.__version__)
    print('check GPU:',torch.cuda.get_device_name(0))


    # --- utils

    def flatten(l):
        return [item for sublist in l for item in sublist]

    def read_CoNLL2003_format(filename, idx=3):
        """Read file in CoNLL-2003 shared task format"""
        
        # read file
        lines =  open(filename).read().strip()   
        
        # find sentence-like boundaries
        lines = lines.split("\n\n")  
        
         # split on newlines
        lines = [line.split("\n") for line in lines]
        
        # get tokens
        tokens = [[l.split('\t')[0] for l in line] for line in lines]
        
        # get labels/tags
        labels = [[l.split('\t')[idx] for l in line] for line in lines]
        
        #convert to df
        data= {'tokens': tokens, 'labels': labels}
        df=pd.DataFrame(data=data)
        
        return df


    # --- load data
    print('\n--- load data')

    PATH_PREFIX = args.path_prefix
    LABEL_COLUMN_INDEX = 1

    # XXX refaktorálható sorminta :)
    trainfile = PATH_PREFIX + "train.txt"
    devfile = PATH_PREFIX + "dev.txt"
    testfile = PATH_PREFIX + "test.txt"

    train = read_CoNLL2003_format(trainfile, idx=LABEL_COLUMN_INDEX)
    dev = read_CoNLL2003_format(devfile, idx=LABEL_COLUMN_INDEX)
    test = read_CoNLL2003_format(testfile, idx=LABEL_COLUMN_INDEX)

    print()
    print(f'train data: {len(train)} sentences, {len(flatten(train.tokens))} tokens')
    print(f'dev data: {len(dev)} sentences, {len(flatten(dev.tokens))} tokens')
    print(f'test data: {len(test)} sentences, {len(flatten(test.tokens))} tokens')
        
    print()
    print(train.head())

    # --- subsample data
    print('\n--- subsample data')

    SUBSAMPLE = args.subsample_size != 0
    SUBSAMPLE_SIZE = args.subsample_size
    # 1000 default
    # 5000 ez már keményebb! :)

    X_train, y_train = train.tokens, train.labels
    X_dev, y_dev = dev.tokens, dev.labels
    X_test, y_test = test.tokens, test.labels

    label_list = np.unique(flatten(y_train))
    label_list = list(label_list)

    if SUBSAMPLE:
        n = SUBSAMPLE_SIZE
        X_train, y_train = X_train[:n], y_train[:n]
        X_test, y_test = X_test[:n], y_test[:n]

    # leghosszabb mondat hossza, gondolom
    MAX_MAX_SEQ_LENGTH = 256 # asszem ne legyen túl sok, mert sok a padding akkor (???)
    max_seq_length = min(
        MAX_MAX_SEQ_LENGTH,
        max(len(x) for x in list(X_train) + list(X_dev) + list(X_test)))

    print()
    print(f'#train={len(X_train)} #test={len(X_test)}')
    print(f'max_seq_length={max_seq_length}')
    print(f'#tags={len(label_list)} -> {label_list[:5]} ...')
    print()


    # --- define BERT token sequence classification model
    print('\n--- define BERT token sequence classification model')

    BERT_MODEL = 'bert-base-multilingual-cased'
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 64 # XXX alapból 16 volt!
    GR_ACC_STEPS = 2 # kb. hogy ennyivel szorozza a BATCH_SIZE-t trükkösen

    model = BertTokenClassifier(
        bert_model=BERT_MODEL,
        epochs=EPOCHS,
        max_seq_length=max_seq_length,
        learning_rate=LEARNING_RATE,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACC_STEPS,
        #ignore_label=['O'] # ha kell... IOB esetén az 'O' lehet ignore
    )

    print(model)


    # --- finetune model
    print('\n--- finetune model <- this step takes time :)')

    print()
    model.fit(X_train, y_train)


    # --- evaluate model
    print('\n--- evaluate model')

    # score model
    print()
    f1_test = model.score(X_test, y_test)
    print()
    print(f'>>> Test F1 = {f1_test:.02f}')
    print()

    # make predictions
    y_preds = model.predict(X_test)

    # calculate the probability of each class
    y_probs = model.predict_proba(X_test)

    # classification report
    cr = classification_report(flatten(y_test), flatten(y_preds), output_dict=True)
    df = pd.DataFrame(data=cr).transpose().sort_values('f1-score')

    print()
    print('classification report')
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None):
        print(df)


    ## --- optional span level stats (implemented in perl)
    ## write out predictions to file for conlleval.pl
    #iter_zip = zip(flatten(X_test),flatten(y_test),flatten(y_preds))
    #preds = [" ".join([token, y, y_pred]) for token, y, y_pred in iter_zip]
    #with open("preds.txt",'w') as f:
    #    for x in preds:
    #        f.write(str(x)+'\n') 
    #
    ## run conlleval perl script 
    # $ perl ./other_examples/conlleval.pl < preds.txt
    # $ rm preds.txt


    # --- check results on test data
    print('\n--- check results on test data')

    TEST_DATA_SIZE = 6

    for i in range(TEST_DATA_SIZE):
        tokens = X_test[i]
        labels = y_test[i]
        preds = y_preds[i]
        changed = ['' if l == p else '<<<' for l, p in zip(labels, preds)]

        data = {
            "token": tokens,
            "label": labels,
            "predict": preds,
            "changed": changed
        }

        df = pd.DataFrame(data=data)
        print()
        print(df)

    # print out probabilities for this observation
    prob = y_probs[i]
    tokens_prob = model.tokens_proba(tokens, prob)
    print()
    print(tokens_prob)


    # --- check result on some new text
    print('\n--- check result on some new text')

#    # XXX for 'frag'
#    TEXTS = [
#        "Ez egy vastag könyv.",
#        "Jól működik ez a program.",
#        "A gyerekeim nagyon okosak és szépek.",
#        "Bevertem a kezemet a szekrénybe.",
#        "Igen, csak az a baj, hogy igaz.",
#        "12 óra alatt 5 cm-t megh aladó f riss hó hul lhat.",
#        "Jó, most már ké rem a tablet et.",
#        "Maradjat ok csönd ben, mert Anya als zik.",
#    ]
    # XXX for 'accent'
    TEXTS = [
        "Ez egy vastag könyv.",
        "Jól mukodik ez a program.",
        "A gyerekeim nagyon okosak és szépek.",
        "Bevertem a kezemet a szekrenybe.",
        "Igen, csak az a baj, hogy igaz.",
        "12 óra alatt 5 cm-t meghaladó friss ho hullhat.",
        "Jó, most már kerem a tabletet.",
        "Maradjatok csöndben, mert Anya alszik.",
    ]

    for text in TEXTS:
        print()
        tag_predicts  = model.tag_text(text)       
        #prob_predicts = model.tag_text_proba(text)


#    # --- save model
#    print('\n--- save model')
#     
#    model.save('bert_sklearn_tokseqcls_model.bin')


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--path-prefix',
        help='datafiles = PATH_PREFIX + train/dev/test + ".txt"',
        type=str,
        default='data/data_'
    )
    parser.add_argument(
        '-s', '--subsample-size',
        help='use SUBSAMPLE_SIZE items from data; 0 means no subsampling, use entire data',
        type=int,
        default=0
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()

