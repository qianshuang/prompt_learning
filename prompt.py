# -*- coding: utf-8 -*-

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch


class Prompting(object):

    def __init__(self, **kwargs):
        model_path = kwargs['model']
        tokenizer_path = kwargs['model']
        if "tokenizer" in kwargs.keys():
            tokenizer_path = kwargs['tokenizer']
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def prompt_pred(self, text):
        """
        输入带有[MASK]的序列，输出LM模型Vocab中的词语列表及其概率
        """
        indexed_tokens = self.tokenizer(text, return_tensors="pt").input_ids
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens[0])
        mask_pos = tokenized_text.index(self.tokenizer.mask_token)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(indexed_tokens)
            predictions = outputs[0]
        values, indices = torch.sort(predictions[0, mask_pos], descending=True)
        result = list(zip(self.tokenizer.convert_ids_to_tokens(indices), values))
        self.scores_dict = {a: b for a, b in result}
        return result

    def compute_tokens_prob(self, text, token_list1, token_list2):
        """
        1. token_list1表示正面情感positive的词（如good, great）；token_list2表示负面情感positive的词（如good, great，bad, terrible）
        2. 计算概率时，正负向类别词得分加和，并softmax归一化，作为最终类别概率
        """
        _ = self.prompt_pred(text)
        score1 = [self.scores_dict[token1] if token1 in self.scores_dict.keys() else 0 \
                  for token1 in token_list1]
        score1 = sum(score1)
        score2 = [self.scores_dict[token2] if token2 in self.scores_dict.keys() else 0 \
                  for token2 in token_list2]
        score2 = sum(score2)
        softmax_rt = torch.nn.functional.softmax(torch.Tensor([score1, score2]), dim=0)
        return softmax_rt

    def fine_tune(self, sentences, labels, prompt=" Since it was [MASK].", goodToken="good", badToken="bad"):
        """
        对已有标注数据进行Fine tune训练。
        """
        good = self.tokenizer.convert_tokens_to_ids(goodToken)
        bad = self.tokenizer.convert_tokens_to_ids(badToken)
        from transformers import AdamW
        optimizer = AdamW(self.model.parameters(), lr=1e-3)

        for sen, label in zip(sentences, labels):
            tokenized_text = self.tokenizer.tokenize(sen + prompt)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            mask_pos = tokenized_text.index(self.tokenizer.mask_token)
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            pred = predictions[0, mask_pos][[good, bad]]
            prob = torch.nn.functional.softmax(pred, dim=0)
            lossFunc = torch.nn.CrossEntropyLoss()
            loss = lossFunc(prob.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            optimizer.step()
