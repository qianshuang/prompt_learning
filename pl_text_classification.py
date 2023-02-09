# -*- coding: utf-8 -*-

from prompt import Prompting

model_path = "bert-base-uncased"
prompting = Prompting(model=model_path)

template = "Because it was [MASK]."
vervlize_pos = ["great", "amazin", "good"]
vervlize_neg = ["bad", "awfull", "terrible"]

text1 = "I really like the film a lot. "
prompt1 = text1 + template
pred1 = prompting.prompt_pred(prompt1)
pred_score1 = prompting.compute_tokens_prob(prompt1, token_list1=vervlize_pos, token_list2=vervlize_neg)
print(pred1[:3])
print(pred_score1)

text2 = "I did not like the film. "
prompt2 = text2 + template
pred2 = prompting.prompt_pred(prompt2)
pred_score2 = prompting.compute_tokens_prob(prompt2, token_list1=vervlize_pos, token_list2=vervlize_neg)
print(pred2[:3])
print(pred_score2)
