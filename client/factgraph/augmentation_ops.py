import spacy
spacy.prefer_gpu()
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import json
import unidecode
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}

class Transformation():
    # Base class for all data transformations

    def __init__(self):
        # Spacy toolkit used for all NLP-related substeps
        self.spacy = spacy.load("en_core_web_sm",exclude=["tagger","entity_linker","entity_ruler","textcat",
                                                          "textcat_multilabel","attribute_ruler", "lemmatizer",
                                                          "trainable_lemmatizer","morphologizer","attribute_ruler",
                                                          "transformer","ner","senter","sentencizer"])

    def transform(self, example):
        # Function applies transformation on passed example
        pass

#select sentences from a document that are most similar to a summary of the document.
class SelectSentencesScore(Transformation):
    # Embed document as Spacy object and sample one sentence as claim
    # the model is for generating sentence embeddings
    def __init__(self, article, min_sent_len=8, model_type='paraphrase-mpnet-base-v2'):
        super().__init__()
        self.min_sent_len = min_sent_len
        self.count_at_random = 0
        self.model_sentence_transf = SentenceTransformer(model_type)
        self.page_text = article.replace("\n", "")
        self.sents_text = [sent.text for sent in self.spacy(self.page_text).sents]
        self.embs = self.model_sentence_transf.encode(self.sents_text, convert_to_tensor=True)


    def transform(self, example, number_sents):
        # assert example["article"] is not None, "Text must be available"

        # self.model_sentence_transf.encode(['Test'], convert_to_tensor=True)

        # example["article"] = unidecode.unidecode(example["article"])
        # example["summary"] = unidecode.unidecode(example["summary"])

        # page_text = example["article"].replace("\n", "")
        summaries = example["summary"].replace("\n", "")

        # page_doc = self.spacy(page_text)
        # #sents = [sent for sent in page_doc.sents if len(sent) >= self.min_sent_len]
        # sents_text = [sent.text for sent in page_doc.sents]

        page_sum = self.spacy(summaries)
        sents_text_sum = [sent.text for sent in page_sum.sents]

        # sents_text = [sent.text for sent in sents]
        # sents_text_sum = [sent.text for sent in sents_sum]

        # assert len(sents) == len(sents_text)

        try:
            # generate the sentence embedding
            # embs = self.model_sentence_transf.encode(sents_text, convert_to_tensor=True)
            embs_sum = self.model_sentence_transf.encode(sents_text_sum, convert_to_tensor=True)

            # Compute the pair-wise cosine similarities
            cos_scores = util.pytorch_cos_sim(self.embs, embs_sum).cpu().detach().numpy()

            centrality_scores = np.mean(cos_scores, axis=1)

            # We argsort so that the first element is the sentence with the highest score
            # sent index
            most_central_sentence_indices = np.argsort(-centrality_scores)

            #assert len(most_central_sentence_indices) == len(sents)

            best_sents = []
            for idx in most_central_sentence_indices[:number_sents]:
                # sents - document
                tmp_claim = self.sents_text[idx]
                # claim_text = tmp_claim.text
                best_sents.append((tmp_claim, int(idx), float(centrality_scores[idx])))
        except:
            self.count_at_random += 1
            print("return none error", self.count_at_random)
            return None

        example['sentences'] = json.dumps(best_sents)
        return example