#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:21:32 2018

@author: angli
"""
import re, numpy as np, pandas as pd


URL_PATTERN=re.compile(r'''(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))''')
SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)
NUMBERS_PATTERN = re.compile(r"(^|\s)(\-?\d+(?:\.\d)*|\d+)")
IP_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


if __name__ == "__main__":
    #loading the dictionary
    positive_list = pd.read_table("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/dictionary/positive-words.txt", header=None)
    positive_list = list(positive_list[0])
    negative_list = pd.read_table("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/dictionary/negative-words.txt", header=None, encoding = "ISO-8859-1")
    negative_list = list(negative_list[0])
    sem_score = pd.read_table("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/dictionary/SemEval2015-English-Twitter-Lexicon.txt", header=None)
    sem_score = sem_score.rename(index=str, columns={0: "score", 1: "word"}) 
    sem_score = pd.Series(sem_score.score.values,index=sem_score.word).to_dict()
    
    #get the basic features
    train = pd.read_csv("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/train.csv")
    train['total_length'] = train['comment_text'].apply(len)
    train['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    train['caps_vs_length'] = train.apply(lambda row: float(row['capitals'])/float(row['total_length'] + 0.001), axis=1)
    train['num_exclamation_marks'] = train['comment_text'].apply(lambda comment: comment.count('!'))
    train['num_question_marks'] = train['comment_text'].apply(lambda comment: comment.count('?'))
    train['num_punctuation'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    train['num_symbols'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    train['num_we'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in ['we', 'We', 'WE']))
    train['num_words'] = train['comment_text'].apply(lambda comment: len(comment.split(" ")))
    train['num_unique_words'] = train['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    train['words_vs_unique'] = train['num_unique_words'] / (train['num_words']+0.0001)
    train['num_smilies'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    train['num_IP'] = train['comment_text'].apply(lambda comment: len(re.findall(IP_PATTERN, comment)))
    train['num_URL'] = train['comment_text'].apply(lambda comment: len(re.findall(URL_PATTERN, comment)))
    #features from dictionary
    train['comment_text_lower'] = train['comment_text'].str.lower()
    train['num_positive'] = train['comment_text_lower'].apply(lambda comment: sum(comment.count(w) for w in positive_list))
    train['num_negtive'] = train['comment_text_lower'].apply(lambda comment: sum(comment.count(w) for w in negative_list))
    train['score'] = train['comment_text_lower'].apply(lambda comment: sum(sem_score.get(c.strip(),0.00) for c in comment.split(" ")))
    train['score'] = train.apply(lambda row: float(row['score'])/float(row['num_words'] + 0.001), axis=1)
    train = train[['id', 'total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 'num_question_marks', 'num_punctuation', 'num_symbols', 'num_we', 'num_words', 'num_unique_words', 'words_vs_unique', 'num_smilies', 'num_IP', 'num_URL', 'comment_text_lower', 'num_positive', 'num_negtive', 'score', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    train.to_csv("train_features.csv", index=False)

    test = pd.read_csv("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/test.csv")
    #get the basic features
    test['total_length'] = test['comment_text'].apply(len)
    test['capitals'] = test['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    test['caps_vs_length'] = test.apply(lambda row: float(row['capitals'])/float(row['total_length'] + 0.001), axis=1)
    test['num_exclamation_marks'] = test['comment_text'].apply(lambda comment: comment.count('!'))
    test['num_question_marks'] = test['comment_text'].apply(lambda comment: comment.count('?'))
    test['num_punctuation'] = test['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    test['num_symbols'] = test['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    test['num_we'] = test['comment_text'].apply(lambda comment: sum(comment.count(w) for w in ['we', 'We', 'WE']))
    test['num_words'] = test['comment_text'].apply(lambda comment: len(comment.split(" ")))
    test['num_unique_words'] = test['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    test['words_vs_unique'] = test['num_unique_words'] / (train['num_words']+0.0001)
    test['num_smilies'] = test['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    test['num_IP'] = test['comment_text'].apply(lambda comment: len(re.findall(IP_PATTERN, comment)))
    test['num_URL'] = test['comment_text'].apply(lambda comment: len(re.findall(URL_PATTERN, comment)))
    #features from dictionary
    test['comment_text_lower'] = test['comment_text'].str.lower()
    test['num_positive'] = test['comment_text_lower'].apply(lambda comment: sum(comment.count(w) for w in positive_list))
    test['num_negtive'] = test['comment_text_lower'].apply(lambda comment: sum(comment.count(w) for w in negative_list))
    test['score'] = test['comment_text_lower'].apply(lambda comment: sum(sem_score.get(c.strip(),0.00) for c in comment.split(" ")))
    test['score'] = test.apply(lambda row: float(row['score'])/float(row['num_words'] + 0.001), axis=1)
    test = test[['id', 'total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 'num_question_marks', 'num_punctuation', 'num_symbols', 'num_we', 'num_words', 'num_unique_words', 'words_vs_unique', 'num_smilies', 'num_IP', 'num_URL', 'comment_text_lower', 'num_positive', 'num_negtive', 'score']]
    test.to_csv("test_features.csv", index=False)
    
