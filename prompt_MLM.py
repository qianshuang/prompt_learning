# -*- coding: utf-8 -*-

from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
template = "Because it was [MASK]."

text1 = "I really like the film a lot. "
prompt1 = text1 + template
pred1 = unmasker(prompt1)
print(pred1)

[{'score': 0.14730945229530334, 'token': 2307, 'token_str': 'great',
  'sequence': 'i really like the film a lot. because it was great.'},
 {'score': 0.10884211957454681, 'token': 6429, 'token_str': 'amazing',
  'sequence': 'i really like the film a lot. because it was amazing.'},
 {'score': 0.09781624376773834, 'token': 2204, 'token_str': 'good',
  'sequence': 'i really like the film a lot. because it was good.'},
 {'score': 0.046277400106191635, 'token': 4569, 'token_str': 'fun',
  'sequence': 'i really like the film a lot. because it was fun.'},
 {'score': 0.04313807934522629, 'token': 10392, 'token_str': 'fantastic',
  'sequence': 'i really like the film a lot. because it was fantastic.'}]

text2 = "this movie makes me very disgusting. "
prompt2 = text2 + template
pred2 = unmasker(prompt2)
print(pred2)

[{'score': 0.054643284529447556, 'token': 9643, 'token_str': 'awful',
  'sequence': 'this movie makes me very disgusting. because it was awful.'},
 {'score': 0.0503225214779377, 'token': 2204, 'token_str': 'good',
  'sequence': 'this movie makes me very disgusting. because it was good.'},
 {'score': 0.04008961096405983, 'token': 9202, 'token_str': 'horrible',
  'sequence': 'this movie makes me very disgusting. because it was horrible.'},
 {'score': 0.035693779587745667, 'token': 3308, 'token_str': 'wrong',
  'sequence': 'this movie makes me very disgusting. because it was wrong.'},
 {'score': 0.03335856646299362, 'token': 2613, 'token_str': 'real',
  'sequence': 'this movie makes me very disgusting. because it was real.'}]
