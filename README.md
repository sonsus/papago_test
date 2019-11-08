# papago_test
0. git config 바꿨으니까 이제 이름이 제대로 올라가겠지...?

1. seq2seq, rnnsearch, ~~transformer 정도(시간나면 추가)~~ 만 만들어서 실험돌려보자
- platform 만들고 메트릭 만들고 beamsearch 만들면 내일 끝날듯?
- 그러면 내일 저녁부터는 실험걸어놓는거지...
- 그 중에 제일 좋아보이는 걸로 최종본 내면 되겠다

2. 난관
- ~~pycocoevalcap을 원래 쓰던대로 쓰려고 했는데 보니까 이게 string이 아니라서 어떻게 될지 모르겠다 조금 고쳐서 쓸 수 있을까?~~
- relative positional embedding 논문 NAACL 2018

3. to-do
* [ ] debug and run rnnsearch and seq2seq with various settings
* [ ] main.py: eval routine
* [ ] metrics.py: Length_measure
    - * [ ] length stats of original dataset
