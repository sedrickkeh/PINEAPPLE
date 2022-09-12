import spacy

class Topic_Attribute_Extractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, sent):
        """ Extracts the nouns and verbs/adjectives in the sentence
        :param sent: the current sentence to extract from

        :return: a list of pairs (topic, attribute)
        """

        doc = self.nlp(sent)
        while True:
            to_merge = None
            for token in doc:
                if token.dep_ == "compound":
                    if token.head.pos_ == "NOUN" or token.head.dep_ == "nsubj":
                        to_merge = doc[token.i : token.head.i+1]
                        break
                    elif token.head.pos_ == "VERB":
                        if (token.head.i > token.i+1):
                            to_merge = doc[token.i : token.head.i]
                            break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "neg":
                        if token.head.i < token.i:
                            to_merge = doc[token.head.i : token.i+1]
                        else:
                            to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "nmod" or token.dep_ == "adjmod":
                        to_merge = doc[token.i : token.head.i+1]
                        break
                    # if token.dep_ == "advmod":
                    #     if token.head.i < token.i:
                    #         to_merge = doc[token.head.i : token.i+1]
                    #     else:
                    #         to_merge = doc[token.i : token.head.i+1]
                    #     break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "poss" or token.dep_ == "det":
                        to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "pobj" or token.dep_ == "oprd":
                        if token.head.i < token.i:
                            to_merge = doc[token.head.i : token.i+1]
                        else:
                            to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "xcomp":
                        if token.head.i < token.i:
                            to_merge = doc[token.head.i : token.i+1]
                        else:
                            to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "acomp":
                        to_merge = doc[token.head.i : token.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "prt":
                        if token.head.i < token.i:
                            to_merge = doc[token.head.i : token.i+1]
                        else:
                            to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "amod":
                        if token.head.i < token.i:
                            to_merge = doc[token.head.i : token.i+1]
                        else:
                            to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "dobj":
                        if token.head.text in ["has", "have", "had"]:
                            to_merge = doc[token.head.i : token.i+1]
                            break
            if to_merge is None:
                for token in doc:
                    if token.pos_ == "NOUN" and token.dep_ == "attr":
                        if token.head.text in ["is", "are"]:
                            to_merge = doc[token.head.i : token.i+1]
                            break
            if to_merge is None:
                for token in doc:
                    if token.text == "can" and token.dep_ == "aux":
                        to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "dobj":
                        if token.head.text in ["gives", "give", "gave", "get", "gets", "got"]:
                            if token.head.i < token.i:
                                to_merge = doc[token.head.i : token.i+1]
                            else:
                                to_merge = doc[token.i : token.head.i+1]


            if to_merge is None: break
            if len(to_merge)==0: break
            with doc.retokenize() as retokenizer:
                retokenizer.merge(to_merge)

        pairs = []
        for possible_subject in doc:            
            if possible_subject.text.lower() in ["that", "which"]:
                if possible_subject.dep_ == "nsubj":
                    if possible_subject.head.dep_ == "relcl":
                        pairs.append((possible_subject.head.head, possible_subject.head, possible_subject.head.pos_))
            elif possible_subject.dep_ == "nsubj" or possible_subject.dep_ == "compound" or possible_subject.dep_ == "npadvmod":
                if possible_subject.head.pos_ == "VERB" or possible_subject.head.pos_ == "AUX":
                    pairs.append((possible_subject, possible_subject.head, possible_subject.head.pos_))
        return pairs

    def extract_old(self, sent):
        doc = self.nlp(sent)
        while True:
            to_merge = None
            for token in doc:
                if token.dep_ == "compound":
                    if token.head.pos_ == "NOUN" or token.head.dep_ == "nsubj":
                        to_merge = doc[token.i : token.head.i+1]
                        break
                    elif token.head.pos_ == "VERB":
                        if (token.head.i > token.i+1):
                            to_merge = doc[token.i : token.head.i]
                            break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "nmod" or token.dep_ == "adjmod":
                        to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "poss" or token.dep_ == "aux" or token.dep_ == "det":
                        to_merge = doc[token.i : token.head.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "pobj":
                        to_merge = doc[token.head.i : token.i+1]
                        break
            if to_merge is None:
                for token in doc:
                    if token.dep_ == "acomp":
                        to_merge = doc[token.head.i : token.i+1]
                        break
            if to_merge is None: break
            if len(to_merge)==0: break
            with doc.retokenize() as retokenizer:
                retokenizer.merge(to_merge)

        pairs = []
        for possible_subject in doc:
            if possible_subject.dep_ == "nsubj" or possible_subject.dep_ == "compound":
                if possible_subject.head.pos_ == "VERB":
                    pairs.append((possible_subject, possible_subject.head))
        return pairs