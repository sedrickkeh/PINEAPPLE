import torch
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import spacy
from nltk.stem import WordNetLemmatizer

rel_formatting = rel_formatting = {
    'AtLocation': "\t\t",
    'CapableOf': "\t\t",
    'Causes': "\t\t",
    'CausesDesire': "\t\t",
    'CreatedBy': "\t\t",
    'DefinedAs': "\t\t",
    'DesireOf': "\t\t",
    'Desires': "\t\t",
    'HasA': "\t\t\t",
    'HasFirstSubevent': "\t",
    'HasLastSubevent': "\t",
    'HasPainCharacter': "\t",
    'HasPainIntensity': "\t",
    'HasPrerequisite': "\t",
    'HasProperty': "\t\t",
    'HasSubevent': "\t\t",
    'InheritsFrom': "\t\t",
    'InstanceOf': "\t\t",
    'IsA': "\t\t\t",
    'LocatedNear': "\t\t",
    'LocationOfAction': "\t",
    'MadeOf': "\t\t",
    'MotivatedByGoal': "\t",
    'NotCapableOf': "\t\t",
    'NotDesires': "\t\t",
    'NotHasA': "\t\t",
    'NotHasProperty': "\t",
    'NotIsA': "\t\t",
    'NotMadeOf': "\t\t",
    'PartOf': "\t\t",
    'ReceivesAction': "\t",
    'RelatedTo': "\t\t",
    'SymbolOf': "\t\t",
    'UsedFor': "\t\t"
}

class COMET_Relations:
    def __init__(self):
        self.rel_formatting = rel_formatting
        self.device = 0
        self.model_file = "pretrained_models/conceptnet_pretrained_model.pickle"

        opt, state_dict = interactive.load_model_file(self.model_file)
        self.data_loader, self.text_encoder = interactive.load_data("conceptnet", opt)
        n_ctx = self.data_loader.max_e1 + self.data_loader.max_e2 + self.data_loader.max_r
        n_vocab = len(self.text_encoder.encoder) + n_ctx

        self.model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

        if self.device != "cpu":
            try:
                cfg.device = int(self.device)
                cfg.do_gpu = True
                torch.cuda.set_device(cfg.device)
                model.cuda(cfg.device)
            except:
                cfg.device = "cpu"
        else:
            cfg.device = "cpu"

        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")

    def is_a_person(self, word):
        if word.lower().strip()=="it":
            return word, 10.0

        input_e1 = word
        relation = "IsA"
        input_e2 = ["person", "human", "man", "woman", "human being", "boy", "girl"]
        
        scores = []
        for e in input_e2:
            outputs = interactive.evaluate_conceptnet_sequence(
                input_e1, self.model, self.data_loader, self.text_encoder, relation, e)
            scores.append(outputs["IsA"]["normalized_loss"])

        return word, sum(scores)/len(scores)


    def get_verb_root(self, action):
        doc = self.nlp(action)
        verb = None
        for word in doc:
            if word.pos_=="VERB":
                verb = word.text
                break
        if verb is None:
            return action
        else:
            return self.lemmatizer.lemmatize(verb, 'v')


    def is_a_person_action(self, action):
        input_e1 = ["person", "human", "man", "woman", "human being", "boy", "girl"]
        relation = "CapableOf"
        input_e2 = self.get_verb_root(action)
        
        scores = []
        for e in input_e1:
            outputs = interactive.evaluate_conceptnet_sequence(
                e, self.model, self.data_loader, self.text_encoder, relation, input_e2)
            scores.append(outputs["CapableOf"]["normalized_loss"])

        return input_e2, sum(scores)/len(scores)

    def is_a_person_property(self, property):
        input_e1 = ["person", "human", "man", "woman", "human being", "boy", "girl"]
        relation = "HasProperty"
        input_e2 = self.get_verb_root(property)
        
        scores = []
        for e in input_e1:
            outputs = interactive.evaluate_conceptnet_sequence(
                e, self.model, self.data_loader, self.text_encoder, relation, input_e2)
            scores.append(outputs["HasProperty"]["normalized_loss"])

        return input_e2, sum(scores)/len(scores)

    def capable_of(self, word1, word2):
        outputs = interactive.evaluate_conceptnet_sequence(
                word1, self.model, self.data_loader, self.text_encoder, "CapableOf", self.get_verb_root(word2))
        return (outputs["CapableOf"]["normalized_loss"]) 

    def has_property(self, word1, word2):
        outputs = interactive.evaluate_conceptnet_sequence(
                word1, self.model, self.data_loader, self.text_encoder, "HasProperty", self.get_verb_root(word2))
        return (outputs["HasProperty"]["normalized_loss"]) 