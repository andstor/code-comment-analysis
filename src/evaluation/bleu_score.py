import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

smoothie = SmoothingFunction().method1

def calc_bleu_score(target, generated):
    #reference = target.strip(" ").splitlines()
    reference = target.strip().splitlines()
    #reference = list(filter(None, reference))
    #reference = list(map(str.strip, reference))
    #hypothesis = generated.strip(" ").splitlines()
    hypothesis = generated.strip().splitlines()
    #hypothesis = list(filter(None, hypothesis))
    #hypothesis = list(map(str.strip, hypothesis))
    list_of_references = [[line.split()] for line in reference]
    hypotheses = [line.split() for line in hypothesis]
    if len(list_of_references) < len(hypotheses):
        diff = len(hypotheses) - len(list_of_references)
        list_of_references.extend([[[]]] * diff)
    elif len(list_of_references) > len(hypotheses):
        diff =  len(list_of_references) - len(hypotheses)
        hypotheses.extend([[]] * diff)
    try:
        return corpus_bleu(list_of_references, hypotheses)#, smoothing_function=smoothie,)
    except Exception as e:
        return -1#pd.NA