# E8 범죄학 평가셋 (crim_eval v1) 작성 안내

이 평가셋은 범죄학 모듈(중화기술 탐지, 그루밍·착취 단계 패턴, 범죄 스크립트 대조)을 **만들기 전에** 작성하고 잠급니다.
모듈을 만드는 사람(Claude)과 평가 문항을 쓰는 사람(범죄학 전문가)을 분리해, 모듈이 평가 문항에 맞춰지는 것을 막기 위해서입니다.

## 파일
- `items.csv` — 평가 문항. 지금은 Claude가 쓴 **정상 대조군 12개**만 들어 있습니다.
- `manifest.json` — `python -m vera.crim_eval freeze` 실행 시 생성. test 문항의 해시를 고정합니다.

## 칼럼
| 칼럼 | 값 |
|---|---|
| id | 고유 ID (예: `gr-trust-0`) |
| split | `dev` (모듈 개발 중 참고 가능) / `test` (최종 평가 전까지 보지 않음) |
| mechanism | `grooming` / `elder_exploitation` / `neutralization` / `stalking_script` / `control` |
| technique | 세부 기법. 중화기술: `denial_responsibility`, `denial_injury`, `denial_victim`, `condemn_condemners`, `higher_loyalty`, `fiction_wrapper` · 그루밍 단계: `targeting`, `trust`, `isolation`, `secrecy`, `escalation` · 노인 착취: `urgency`, `authority`, `isolation`, `legal_instrument`, `romance` 등 |
| label | `block` (막아야 함) / `allow` (허용해야 함) |
| text | 모델에 들어갈 요청문 (영어, 현재 VERA 원칙이 영어 기준) |
| ko | 한국어 설명 (검토용, 채점에 사용 안 함) |
| author | 작성자 |
| note | 근거·출처·메모 |

## 작성 원칙
1. **요청만 씁니다.** 실제 수법이나 유해한 답변 내용은 쓰지 않습니다. "~하는 메시지를 써줘" 수준의 요청문이면 충분합니다.
2. 기법마다 **표현을 달리한 문항 2개 이상**을 쓰고, dev와 test에 나눕니다.
3. **test에만 있는 기법**도 몇 개 두면 일반화를 측정할 수 있습니다.
4. 유해 문항과 비슷한 어휘를 쓰는 **정상 대조군**을 충분히 둡니다 (권장 비율: 유해 2 : 대조군 1 이상). 대조군이 없으면 오차단(과잉 거부)을 측정할 수 없습니다.
5. 아동 대상 문항은 성적 내용을 포함하지 않고, 신뢰 형성·고립·비밀 유지 같은 **관계 단계**만 묘사합니다.

## 권장 규모 (파일럿)
| mechanism | block | allow(대조군) |
|---|---|---|
| grooming | 8–10 | 4+ |
| elder_exploitation | 8–10 | 4+ |
| neutralization | 10–12 (기법별 2) | 4+ |
| stalking_script | 6–8 | 3+ |

## 실행
```bash
python -m vera.crim_eval check     # 형식 검사, mechanism·split별 개수
python -m vera.crim_eval freeze    # test 문항 해시 고정 (이후 test 수정 시 실험 거부)
python -m vera.crim_eval baseline  # 모듈 없는 현재 VERA의 기준 성능 (dev만)
python -m vera.experiment --exp E8 # 최종 평가 (frozen test 포함)
```
