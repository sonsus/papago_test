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
* [ ] **debug rnnsearch**
* [ ] metrics.py:
    - * [ ] **length measure**
    - * [X] pycocoevalcap 붙임
    - * [X] stats of trg, src -> data/len.py 에 주석 + 그리고 png파일로 그려놓음
* [X] beamsearch: @MaximumEntropy 에게서 가져와서 고쳐씀
* [X] word_drop:
* [X] label_smoothing
* [X] decoding heuristic
    - * [X] no EOS, PAD, SOS until likely end length (trglen ~= 0.552srclen - 0.406)
* [ ] Transformer **일단 쓰던 코드로** 써보자...
    - * [ ] beam search with heuristic for Transformer  
    - * [ ] greedy with heuristic **어차피 이건 나도 써야할 코드라 작성해야함**

* [ ] main.py: eval routine
* [ ] set-up guide!


## 4. 진척
### seq2seq는 어차피 baseline 으로 뒀으니 괜찮다고 생각했으나
    - 너무 안되는거 아닌가... 심한데
    - eval routine을 마련하는 건 제출을 위한 마무리단계이긴하다  
    - 하지만 작성하지 못하면 완료를 못하는건 맞음




## 5. write2u 환경에

```bash
nvidia-smi
##CUDA 10.1, Driver 435.21, TitanXp
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
| 23%   31C    P8    12W / 250W |    518MiB / 12192MiB |      7%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN Xp            Off  | 00000000:07:00.0 Off |                  N/A |
| 23%   23C    P8     7W / 250W |      2MiB / 12196MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      2098      G   /usr/lib/xorg/Xorg                            30MiB |
|    0      2224      G   /usr/bin/gnome-shell                          96MiB |
|    0      8247      G   /usr/lib/xorg/Xorg                           269MiB |
|    0      8374      G   /usr/bin/gnome-shell                          98MiB |
+-----------------------------------------------------------------------------+


# 1.write2u
write2u 환경 설치 #pytorch 1.2, torchtext 0.4.0

# 2.conda forge
conda config --add channels conda-forge
conda config --set channel_priority strict

# 3.if need to run data/len.py
conda install matplotlib seaborn pandas

```
