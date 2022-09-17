# MIDX_Journal
This is a repo for MIDX sampler applied in other domains like NLP, Knowledge Graph, Sequential Recommendation and Extreme Classification.


### RUN

```bash
python run.py -m=LSTM -d=penntreebank --sampler=midx-uni
```

### Supported

| Task | Model | Dataset |
| --- | --- | --- |
| LanguageModeling | [LSTM, Transformer] | [penntreebank, wikitext-2, wikitext-103] |
| KnowledgeGraph | [HouseE] | |
| ExtremeClassification | | [AmazonCat-13k, Delicious-200k, WikiLSHTC] |

Optional Sampler: [None, 'midx-uni', 'midx-pop'] (Default: None)

when sampler is None, FullSoftmax is used as loss function, otherwise sampled softmax is used.