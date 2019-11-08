# papago_test
* [X] git config 바꿨으니까 이제 이름이 제대로 올라가겠지...?

## 0. 너무 low resource 라서... 근데 그냥 아는 알고리즘들로 해보자
- Revisiting Low-Resource Neural Machine Translation:A Case Study
- 에서는 hyper parameter tuning의 효과랑 label smoothing 효과를 봤다더라
    (... 처음에 아는 알고리즘들에 이것저것 붙여서 결과뽑자고 너무 쉽게 생각했다)
    
## 1. seq2seq, rnnsearch, ~~transformer 정도~~ 만 만들어서 실험돌려보자
- platform 만들고 메트릭 만들고 beamsearch 만들면 내일 끝날듯?
- 그러면 내일 저녁부터는 실험걸어놓는거지...
- 그 중에 제일 좋아보이는 걸로 최종본 내면 되겠다

## 2. 최대한 서치 범위를 줄여야하므로
- learning rate 부터 건드려봄: 혹시몰라서 transformer 사용된 스케쥴러도 rnn에 써보았다
- layer는 하나로만(rnns)
- embedding, hidden 사이즈 256 통일, 데이터 스케일 비해 이것도 큰 게 아닐까하는 생각...

## 3. to-do
* [X] seq2seq
* [ ] debug rnnsearch
* [ ] metrics.py:
    - * [ ] length stats of original dataset: done
    - * [X] stats of trg, src -> data/len.py 에 주석으로 적혀있다
* [X] beamsearch: @MaximumEntropy 에게서 가져와서 고쳐씀
* [X] word_drop:
* [ ] label_smoothing
* [ ] Transformer 일단 가져다 써보자...
