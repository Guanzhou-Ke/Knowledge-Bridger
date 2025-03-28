from torchmetrics.text import BLEUScore, BERTScore, ROUGEScore
import pandas as pd
import torch
import json

class NLPEvaluator:
    
    def __init__(self, bert_path='google-bert/bert-large-cased') -> None:
        self.bertscore = BERTScore(model_name_or_path=bert_path)
        self.bleu1 = BLEUScore(n_gram=1)
        self.bleu4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys='rougeL')
    
    
    def compute(self, predictions, targets):
        bert_score = self.bertscore(predictions, targets)
        bleu1 = self.bleu1(predictions, targets)
        bleu4 = self.bleu4(predictions, targets)
        rougeL = self.rougeL(predictions, targets)
        
        self.result = {
            'bert_f1': torch.mean(bert_score['f1']).item(),
            'bert_precision': torch.mean(bert_score['precision']).item(),
            'bert_recall': torch.mean(bert_score['recall']).item(),
            'BLEU-1': bleu1.item(),
            'BLEU-4': bleu4.item(),
            'rougeL': rougeL['rougeL_fmeasure'].item(),
        }
        
        return self.result
    
    
    def save(self, save_path):
        # df = pd.DataFrame([self.result]).T
        # df.to_csv(save_path)
        json.dump(self.result, open(save_path, 'w'), indent=4)
        
        
if __name__ == '__main__':
    predictions = ['Red helmeted rider on a dirt path, motorcycle in hand, against a backdrop of green mountains and clouds.',
                   "Motorcycle and rider on a rugged path, surrounded by nature's serenity and mountainous vistas.",
                   'Adventurous rider in red, navigating a rocky path beside a lush mountain, motorcycle gleaming in the sun.',
                   'A person in a red shirt and helmet riding a motorcycle along a winding mountain path.',
                   'Rider on a red motorcycle, path lined with rocks and trees, mountain peaks in the distance.',]
    targets = ['A man with a red helmet on a small moped on a dirt road. ',
  'Man riding a motor bike on a dirt road on the countryside.',
  'A man riding on the back of a motorcycle.',
  'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ',
  'A man in a red shirt and a red hat is on a motorcycle on a hill side.']
    
    nlp_eval = NLPEvaluator()
    result = nlp_eval.compute(predictions, targets)
    print(result)
    nlp_eval.save('./src/evaluation_results/test.json')