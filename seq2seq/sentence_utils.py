from typing import List
import subprocess

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