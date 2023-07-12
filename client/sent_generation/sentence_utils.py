import re
from typing import List, Tuple
import urllib.request
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm",exclude=["tagger","entity_linker","entity_ruler","textcat",
                                                          "textcat_multilabel","attribute_ruler", "lemmatizer",
                                                          "trainable_lemmatizer","morphologizer","attribute_ruler",
                                                          "transformer","ner","senter","sentencizer"])

import os

PORT = os.getenv('PORT')

class ConverterSentenceToAMR():

    def __call__(self, sentences: List[str], output_path=None)->str:
        raise NotImplementedError("ConverterSentenceToAMR needs to be subclassed")

class SpringConverter(ConverterSentenceToAMR):

    def __init__(self, data_dir: str):       
        self.data_dir = data_dir

    def __call__(self, sentences: List[str], output_path=None)->str:

        assert isinstance(sentences, list), 'sentences param must be list type'

        with open(f'{self.data_dir}/predict.txt', 'w') as file:
            sentence = '\n'.join([sent for sent in sentences])
            file.write(sentence)

        # convert sentence to amr
        input_path = urllib.parse.quote_plus(f'{self.data_dir}/predict.txt')
        if output_path is not None:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}&only_ok=1&output_file_name={output_path}")
        else:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}&only_ok=1")
        encodedContent = fp.read()
        amr_file_path = encodedContent.decode("utf8")
        fp.close()

        return amr_file_path
    
class AMRBartConverter(ConverterSentenceToAMR):
    def __init__(self, data_dir: str):       
        self.data_dir = data_dir

    def __call__(self, sentences: List[str], output_path=None) -> str:
        input_path = f'{self.data_dir}/predict.jsonl'

        assert isinstance(sentences, list), 'sentences param must be list type'


        with open(input_path, 'w') as file:
            file.write('\n'.join(["{\"sent\": \""+ sent +"\", \"amr\":\"\"}" for sent in sentences]))

        # convert sentence to amr
        input_path = urllib.parse.quote_plus(input_path)
        if output_path is not None:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}&output_file_name={output_path}")
        else:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}")
        encodedContent = fp.read()
        amr_file_path = encodedContent.decode("utf8")
        fp.close()
        
        return amr_file_path
    
class AMRLibServerConverter(ConverterSentenceToAMR):
    def __init__(self, data_dir: str):       
        self.data_dir = data_dir

    def __call__(self, sentences: List[str], output_path=None) -> str:
        input_path = f'{self.data_dir}/predict.jsonl'

        assert isinstance(sentences, list), 'sentences param must be list type'

        with open(input_path, 'w') as file:
            file.write('\n'.join(["{\"summary\": \""+ sent +"\", \"article\":\"\"}" for sent in sentences]))

        # convert sentence to amr
        input_path = urllib.parse.quote_plus(input_path)
        if output_path is not None:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}&output_file_name={output_path}")
        else:
            fp = urllib.request.urlopen(f"http://localhost:{PORT}/predict_amrs_from_plaintext?input_path={input_path}")
        encodedContent = fp.read()
        amr_file_path = encodedContent.decode("utf8")
        fp.close()
        
        return amr_file_path

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class AMRLibConverter(ConverterSentenceToAMR):
    def __init__(self, data_dir: str, output_dir: str, model_dir: str) -> None:
        import amrlib

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.stog = amrlib.load_stog_model(model_dir=model_dir)

    def __call__(self, sentences: List[str], output_path=None) -> str:

        assert isinstance(sentences, list), 'sentences param must be list type'

        basename = output_path if output_path else 'predict'
        output_path = ''.join([self.output_dir, '/' , basename, '.amr.txt'])

        try:
            sentences = list(set(sentences))
            for sents in chunks(sentences, 20):
                with open(output_path, 'w') as f:
                    graphs = self.stog.parse_sents(sents, add_metadata=True)
                    for g in graphs:
                        f.write(g + "\n\n")
        except Exception as e:
            raise e

        return output_path


def is_sentence_done(text: str) -> bool:
    exception_indicators = ["(\s\"?s|\s?\"?S)ec\.", "(\su|\s?U)\.\s?(s|S)\.", "\se\.?\s?g\.", "\s?E\.?\s?g\.", "\set al\.\)?", \
        "\si\.?\s?e\.", "\sw\.?r\.?t.", "\sw\.?\s?r\.", "\sa\.?k\.?a\.", "\sa\.?\s?k\.", \
        "\setc\.,", "\s(v|V)\.?(s|S)\.", "(\sp|\s?P)\.?(s|S)\.", "\s[A-Z][a-z]+\s[A-Z]\.", \
        "https?:[0-9\/a-zA-Z\.\-^\s]+\.", "\s?[0-9]+\.[0-9]+", "(\(|\[)[^\)|\]]+(\.|\;)",\
        "\s[a-zA-Z]\.", "\s[a-z]\.\s?[^A-Z]", "M(?:rs?|s|iss)\.", "\sm(?:rs?|s|iss)\.", "(C|\sc)orp\.", "\sdept\.", "(\s?D|\sd)r\.", "\sed\.", "\sest\.", "\sinc\.", "\s?Jr\.", "\s?Sr\.", "\s?Ltd\.", "(\s?N|\sn)o\.", "(\s?S|\ss)t\.", "\sibid\."]

    for match in re.finditer("|".join(exception_indicators), text):
        if match.end() == len(text):
            return False
        
    end_indicators = ["\;","\.\"?\s?\)?",'\?"?', '\!"?']
    for match in re.finditer("|".join(end_indicators), text):
        if match.end() == len(text):
            return True

    return False

def get_num_sent(text: str):
    assert isinstance(text, str), 'text param must be str type'
    return len(list(nlp(text).sents))

def is_num(text: str):
    return bool(re.search("\d\.\d", text))