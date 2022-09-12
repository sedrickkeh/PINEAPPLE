import pandas as pd 
import nltk
nltk.download('wordnet')
from copy import deepcopy

from topic_attribute_extractor import Topic_Attribute_Extractor
t = Topic_Attribute_Extractor()

from comet_relations import COMET_Relations
c = COMET_Relations()

from bart_infiller import BART_Infiller
b = BART_Infiller()


df = pd.read_csv("/data/personifications_corpus.csv")
df = df[df.type_of_pers != "none"].reset_index(drop=True)

pers = df.Sentence.tolist()

import numpy as np
from bert_score import score 

class Scorer:
    def __init__(self, pair_extractor, comet_scorer, a=1, b=1, c=1):
        self.pair_extractor = pair_extractor
        self.comet_scorer = comet_scorer
        self.a = a
        self.b = b
        self.c = c

    def get_score(self, new_sent, bart_mask_score, bert_score):
        # The lower the better for all scores

        # BERTScore
        score_a = 1-bert_score
        if score_a < 1e-2:
            score_a = 100   # Don't accept if sentence is exactly the same

        # BART Mask Score
        score_b = -bart_mask_score

        # COMET Score
        topic_attribute_pairs = self.pair_extractor.extract(new_sent)
        comet_scores = []
        for t in topic_attribute_pairs:
            topic, action = t[0].text, t[1].text
            if self.comet_scorer.is_a_person(topic)[1] < 7.0:
                continue
            human_score = self.comet_scorer.is_a_person_action(action)
            nonhuman_score = self.comet_scorer.capable_of(topic, action)
            curr_comet_score = (nonhuman_score-human_score[1])+10.0
            comet_scores.append(curr_comet_score)
        if len(comet_scores)==0:
            comet_scores.append(10.0)
        score_c = max(comet_scores)

        print(score_a, score_b, np.log(score_c))
        return self.a*score_a + self.b*score_b + self.c*np.log(score_c)

sc = Scorer(t, c)

arr, origs = [], []
for idx, sent in enumerate(pers):
    origsent = deepcopy(sent)
    print(idx, sent)

    # Extract attribute pairs
    topic_attribute_pairs = t.extract(sent)
    print("Topic attribute pairs: \t", topic_attribute_pairs)

    # Identify which ones are non-humans with COMET
    nonhumans = []
    for p in topic_attribute_pairs:
        person_score = c.is_a_person(p[0].text)
        if person_score[1] >= 7.0:
            nonhumans.append(p)
    print("Non-human pairs: \t", nonhumans)
    if len(nonhumans)==0: 
        print("="*50, '\n')
        continue

    # Mask sentence:
    for p in nonhumans:
        sent = sent.replace(p[1].text, '<mask>')
    print("Masked sent: \t", sent)
    
    # Generated sentences:
    print("Replacements:")
    reps = b.generate_replacements(sent)
    # print(reps)
    origs.append(origsent)
    
    candidates, bart_scores = reps[0], reps[1]
    
    # BERT score
    P,R,F = score(candidates, [origsent for _ in range(len(candidates))], lang="en", rescale_with_baseline=True)
    F = F.cpu().detach().numpy()
    best_score, best_cand = 100000, None
    for (cand, bs, F1) in zip(candidates, bart_scores, F):
        curr_score = sc.get_score(cand, bs, F1)
        if curr_score < best_score:
            best_score = curr_score
            best_cand = cand
        print(curr_score, cand, origsent)

    arr.append(best_cand)
    origs.append(origsent)
    print(best_cand)
    print(origsent)
    print("="*50, '\n')


with open("data/depersonification_literal.txt", "w") as f:
    for line in arr:
        f.write(line+'\n')
with open("data/depersonification_pers.txt", "w") as f:
    for line in origs:
        f.write(line+'\n')