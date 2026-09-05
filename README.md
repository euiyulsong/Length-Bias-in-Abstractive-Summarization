이 결과만 보면 **length inflation은 발생하지 않았고, 오히려 요약 품질은 좋아지면서 길이는 조금 짧아졌어.**

### 핵심 결과

| Metric       | PRETRAINED |        SFT |                       변화 |
| ------------ | ---------: | ---------: | -----------------------: |
| ROUGE-1      |     0.4004 | **0.4138** |                  +0.0134 |
| ROUGE-2      |     0.1212 | **0.1615** |              **+0.0403** |
| ROUGE-L      |     0.2533 | **0.2782** |                  +0.0249 |
| ROUGE-Lsum   |     0.2822 | **0.2969** |                  +0.0147 |
| 평균 생성 길이     | **145.92** | **138.60** | **-7.32 tokens (-5.0%)** |
| Reference 길이 |      92.64 |      92.64 |                        - |

특히 ROUGE-2가 `0.1212 → 0.1615`, 상대적으로 약 **33% 증가**해서 단순히 출력 형식만 배운 것보다는 핵심 phrase/content overlap이 꽤 좋아진 것으로 보인다.

그런데 동시에 길이는:

$$
145.92 \rightarrow 138.60
$$

으로 약 **5% 감소**했어.

따라서 지금 결과는:

> **좋은 summarization SFT → quality ↑, length ↓**

야.

즉 네가 확인하려던 **“좋은 요약을 학습하면 모델이 quality를 올리면서 그냥 더 길게 쓰는 length inflation이 생기는가?”**라는 가설은 현재 결과에서는 지지되지 않아.

### 다만 원래 모델부터 상당히 길다

Reference 대비 길이를 계산하면:

$$
\frac{145.92}{92.64}=1.575
$$

즉 pretrained는 human reference보다 **57.5% 길고**,

SFT는

$$
\frac{138.60}{92.64}=1.496
$$

즉 human reference보다 여전히 **49.6% 길어**.

그래서 결과를 더 정확히 표현하면:

> Qwen3 pretrained에는 이미 상당한 **verbosity/length bias**가 존재했지만, summarization SFT가 이를 악화시키지는 않았으며 오히려 약간 완화했다.

이게 꽤 흥미로운 결과야.

---

### 지금 단계에서 결론을 세게 내리면 안 되는 이유

`n=25`라서 아직 표본이 작아.

특히 평균 길이는 몇 개의 매우 긴 output에 영향을 받을 수 있어. 다음에는 반드시:

* `median_pred_tokens`
* sample별 `SFT_len - BASE_len`
* `P(SFT_len > BASE_len)`
* 95% bootstrap CI
* length histogram

까지 보는 게 좋아.

예를 들어:

```text
BASE → SFT

평균 length       145.9 → 138.6
median length     ???
SFT가 더 긴 sample ?? / 25
SFT가 더 짧은 sample ?? / 25
```

까지 나오면 length inflation 여부를 훨씬 확실히 말할 수 있어.

### 그리고 `OpenOrca summaries: 0/109`

이건 quality evaluation 결과와는 별개야. 앞에서 만든 코드가 **추가 summarization 데이터 109개를 OpenOrca에서 찾으려고 streaming scan하는 단계**로 보인다.

`0/109` 자체만으로 오류라고 볼 수는 없지만, 한동안 계속 0이면 우리가 만든 summarization keyword filter가 OpenOrca에서 너무 sparse할 가능성이 있어.

사실 네 목적에는 이 부분도 별로 마음에 안 들어. **이번 결과처럼 human-summary 데이터만 가지고 실험하는 편이 훨씬 깨끗해.** OpenOrca synthetic response를 섞으면 "좋은 human summary SFT"라는 해석이 흐려지니까.

그리고 네 목적에 가장 중요한 다음 실험은 **학습량을 증가시키는 것**이야:

```text
0 examples     → BASE
100 examples
250 examples
500 examples
1000 examples
```

각 checkpoint에서

```text
ROUGE
mean output length
reference 대비 length ratio
P(output > reference)
```

를 재면,

> **summary quality를 계속 학습할수록 length가 systematic하게 증가하는가?**

를 직접 볼 수 있어.

현재 첫 결과는 오히려 **“quality improvement ≠ length inflation”** 쪽으로 나왔다.
