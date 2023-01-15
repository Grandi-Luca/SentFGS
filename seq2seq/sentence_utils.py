from typing import List, Tuple
import subprocess
import re

class ConverterSentenceToAMR():
    def __call__(self, sentences:str)->List[str]:
        raise NotImplementedError("ConverterSentenceToAMR needs to be subclassed")

class SpringConverter(ConverterSentenceToAMR):
    def __call__(self, sentences:str)->List[str]:
        with open('documents/predict.txt', 'w') as file:
            file.write(sentences)

        # convert sentences to amr
        amr_graphs_path = subprocess.check_output('conda run -n spring-env /home/grandi/project/seq2seq/sentence_to_amr.sh documents/predict.txt', shell=True).decode("utf-8").replace('\n', '')

        return amr_graphs_path + "predict.amr.txt"

def is_sentence_done(text: str) -> Tuple[bool, int]:
    exception_indicators = ["(\s\"?s|\s?\"?S)ec\.", "(\su|\s?U)\.\s?(s|S)\.", "\se\.?\s?g\.", "\s?E\.?\s?g\.", "\set al\.\)?", \
        "\si\.?\s?e\.", "\sw\.?r\.?t.", "\sw\.?\s?r\.", "\sa\.?k\.?a\.", "\sa\.?\s?k\.", \
        "\setc\.,", "\s(v|V)\.?(s|S)\.", "(\sp|\s?P)\.?(s|S)\.", "\s[A-Z][a-z]+\s[A-Z]\.", \
        "https?:[0-9\/a-zA-Z\.\-^\s]+\.", "\s?[0-9]+\.[0-9]+", "(\(|\[)[^\)|\]]+(\.|\;)",\
        "\s[a-zA-Z]\.", "\s[a-z]\.\s?[^A-Z]", "M(?:rs?|s|iss)\.", "\sm(?:rs?|s|iss)\.", "(C|\sc)orp\.", "\sdept\.", "(\s?D|\sd)r\.", "\sed\.", "\sest\.", "\sinc\.", "\s?Jr\.", "\s?Sr\.", "\s?Ltd\.", "(\s?N|\sn)o\.", "(\s?S|\ss)t\.", "\sibid\."]

    for match in re.finditer("|".join(exception_indicators), text):
        mask = '_'*(len(text[match.start():match.end()]))
        text = text.replace(text[match.start():match.end()], mask)
        
    end_indicators = ["\;","\.\"?\s?\)?",'\?"?', '\!"?']
    end_sentences = []
    for match in re.finditer("|".join(end_indicators), text):
        end_sentences.append(match)
    not_empty = len(end_sentences) > 0

    return (not_empty, end_sentences[-1].end() if not_empty else -1)