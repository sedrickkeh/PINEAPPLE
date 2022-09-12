from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

class BART_Infiller:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.config = BartConfig(force_bos_token_to_be_generated=True)
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", config=self.config)

    def generate_replacements(self, sent, num_beams=10, num_return_sentences=10):
        """Replaces <mask> appropriately using BART

        :param sent: a sentence with at least one '<mask>' token
        :param num_beams: number of beams for beam search
        :param num_return_sentences: number of sentences to return

        :return: a list of sentences with '<mask>' replaced
        """
        batch = self.tokenizer(sent, return_tensors='pt')
        generated_ids = self.model.generate(batch['input_ids'], num_beams=num_beams, num_return_sequences=num_return_sentences, output_scores=True, return_dict_in_generate=True, max_length=128)
        generated_sents = []
        for x in self.tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True):
            generated_sents.append(x)
        scores = generated_ids[1].cpu().detach().numpy()
        return generated_sents, scores