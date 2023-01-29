import re
from typing import List, Tuple
import subprocess
import urllib.request

class ConverterSentenceToAMR():
    def __call__(self, sentences:str)->List[str]:
        raise NotImplementedError("ConverterSentenceToAMR needs to be subclassed")

class SpringConverter(ConverterSentenceToAMR):
    def __call__(self, sentences:str)->List[str]:
        with open('../documents/predict.txt', 'w') as file:
            file.write(sentences)

        # convert sentences to amr
        input_path = urllib.parse.quote_plus('../documents/predict.txt')
        fp = urllib.request.urlopen(f"http://localhost:1234/predict_amrs_from_plaintext?input_path={input_path}&only_ok=1")
        encodedContent = fp.read()
        amr_file_path = encodedContent.decode("utf8")
        fp.close()
        # amr_file_path = subprocess.check_output('conda run -n spring-env /home/grandi/project/seq2seq/sentence_to_amr.sh documents/predict.txt', shell=True).decode("utf-8").replace('\n', '')

        return amr_file_path

def is_sentence_done(text: str) -> Tuple[bool, bool]:
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