# ComfyUI-ImageIR-PromptEngine

Read reference images into **IMAGE_IR**, keep requested future motion in **USER_INTENT**, and compose grounded MiniMax H3 prompts for **T2VA / I2VA / FL2VA / L2VA / Ref2VA**.

**[한국어](#한국어) · [English](#english)**

```text
IMAGE / REFERENCES ──→ Image IR Analyzer ──→ IMAGE_IR ──┐
                                                         ├─→ H3 Mode Composer ─→ Provenance Guard ─→ Final H3 Prompt
USER REQUEST ─────────→ H3 User Intent Author ─→ USER_INTENT ─┘
```

---

# 한국어

## 핵심 개념

- **IMAGE_IR**: 이미지에서 실제로 관찰된 현재 상태. 속성별 `certainty`, 선택적 `confidence`, `verification_required`, `evidence`를 보존합니다.
- **USER_INTENT**: 사용자가 앞으로 일어나길 원하는 행동, 카메라, 장면 진행, 소리, 음악, 대사.
- **H3_RULE**: MiniMax H3 모드별 구조·정렬·타임라인 규칙.
- **Provenance Guard**: 최종 문장이 IMAGE_IR / USER_INTENT / H3_RULE 중 어디서 왔는지 추적합니다.

즉 `이미지에 없음 = 무조건 금지`가 아닙니다. **현재 시각 사실은 IMAGE_IR 근거가 필요하고, 미래 행동은 USER_INTENT 근거가 있으면 허용**됩니다.

## 권장 워크플로

### I2VA

```text
[Load Image]
     ↓
[Image IR Analyzer] ← [Image IR Backend Config] 또는 [ImageIR Llama.cpp Model Manager]
     ↓ IMAGE_IR
     ├──────────────────────────────┐
     │                              │
     │       [H3 User Intent Author] ← 자유로운 한국어/영어 영상 요청
     │              ↓ USER_INTENT   │
     └──────────────┬───────────────┘
                    ↓
             [H3 Mode Router]
                    ↓ routed_mode
       [MiniMax H3 Mode Composer (5 modes)]
                    ↓ h3_document
          [ImageIR Provenance Guard]
                    ↓
             최종 H3 프롬프트
```

처음에는 `mode=AUTO`, `shot_count=1`로 시작하는 것을 권장합니다.

## IMAGE_IR의 불확실성

`certainty`와 `confidence`는 다른 값입니다.

| certainty | 의미 | 출력 |
|---|---|---|
| `observed` | 이미지에서 읽힌 값 | 기록된 값 사용 |
| `uncertain` | 존재/범주는 보이지만 구체값이 불확실 | hedge만 사용, 후보는 확정하지 않음 |
| `absent` | 확인했지만 없음 | 긍정 기술에 사용하지 않음 |

```json
{
  "value": "likely sheer nude tights",
  "certainty": "uncertain",
  "confidence": 0.78,
  "verification_required": true,
  "evidence": "uniform leg tone and subtle surface sheen"
}
```

**불확실 후보는 negative prompt가 아닙니다.** `satin`인지 `silk`인지 판독하지 못했다는 사실은 둘 다 거짓이라는 뜻이 아닙니다.

## 9가지 Grounding 규칙

1. IMAGE_IR에 없는 현재 시각 사실을 새로 만들지 않습니다.
2. 관찰된 속성을 바꾸지 않습니다.
3. `uncertain` 속성을 특정 후보로 확정하지 않습니다.
4. 좌우가 확인되지 않았다면 viewer-left/right를 subject-left/right로 바꾸지 않습니다.
5. 정지 이미지에 없는 이전 상태를 지어내지 않습니다.
6. USER_INTENT가 없는 자동 동작은 IMAGE_IR과 모순되지 않는 안전 범위에서만 생성합니다.
7. 불확실성을 그대로 보존합니다.
8. IMAGE_IR이 화면 기준 좌표를 주면 화면 기준 표현을 유지합니다.
9. 완성된 시각 진술을 출처와 대조하고 추적되지 않는 내용을 거부합니다.

기존 이미지 전용 composer의 안전 미세 동작은 `blink`, `breathing`, `finger_movement`, `hair_movement`, `weight_shift`입니다. 새 5-mode USER_INTENT 경로에서는 사용자가 명시한 미래 행동을 이 목록에 가두지 않습니다.

## 백엔드

### `api_token`과 `max_tokens`

| 필드 | 의미 |
|---|---|
| `api_token` | 인증 키 |
| `api_token_env` | 인증 키를 담은 환경변수 **이름**. workflow에 키 자체를 남기지 않기 위해 권장 |
| `max_tokens` | 모델 응답 최대 토큰 수. 인증과 무관 |

직접 입력한 `api_token`은 ComfyUI workflow/PNG metadata에 저장될 수 있으므로 로컬 서버가 토큰을 요구한다면 `api_token_env` 사용을 권장합니다.

### llama.cpp — 단일 모델을 ComfyUI에서 직접 실행

기존 방식은 그대로 유지됩니다.

```text
provider          = llama_cpp
server_mode       = launch_local
llama_server_path = D:/llama.cpp/llama-server.exe
gguf_model_path   = D:/models/model.gguf
mmproj_path       = D:/models/mmproj-model.gguf
host              = 127.0.0.1
port              = 8080
context_size      = 16384
gpu_layers        = 999
```

**Image IR Backend Control**에서 `start / stop / restart / status`를 사용할 수 있습니다. 이 확장이 띄운 프로세스만 종료하며 외부 llama.cpp나 LM Studio는 종료하지 않습니다.

### llama.cpp — Router Mode + 모델 선택 UI

여러 GGUF 모델을 서버 재시작 없이 선택하려면 최신 llama.cpp의 router mode를 사용합니다.

```bat
llama-server.exe --models-dir D:\LLM\models --host 127.0.0.1 --port 8080
```

ComfyUI에서 **ImageIR Llama.cpp Model Manager** 노드를 추가합니다.

```text
base_url = http://127.0.0.1:8080

[Connect / Refresh]
        ↓
selected_model 드롭다운 자동 갱신
        ↓
원하는 모델 선택
        ↓
[Load] / [Unload]
        ↓
backend_config → Image IR Analyzer / H3 User Intent Author
```

Manager는 llama.cpp router의 다음 API를 사용합니다.

- `GET /models`
- `GET /models?reload=1`
- `POST /models/load`
- `POST /models/unload`

`/models`가 제공하는 `architecture.input_modalities`에 `image`가 있으면 vision-capable 모델로 표시할 수 있습니다. 모델 선택 후 출력되는 `backend_config`는 기존 `IMAGEIR_BACKEND`와 동일하므로 Analyzer와 Intent Author에 그대로 연결합니다.

보안상 브라우저용 Connect/Refresh 프록시는 `localhost`, `127.0.0.1`, `::1`만 허용합니다. 인증이 필요한 router는 `api_token_env`를 사용하세요. **기존 단일 `--model` 서버는 계속 Image IR Backend Config로 사용할 수 있습니다.**

> 이 기능은 llama.cpp가 가진 router model catalog를 사용합니다. 플러그인이 임의의 GGUF/mmproj 파일명을 추측해 자동 매칭하는 기능은 아닙니다.

### 이미 실행 중인 llama.cpp / LM Studio

```text
llama.cpp:
provider    = llama_cpp
server_mode = connect_existing
base_url    = http://127.0.0.1:8080
model_name  = 서버가 노출하는 모델 id

LM Studio:
provider    = openai_compatible
server_mode = connect_existing
base_url    = http://127.0.0.1:1234
model_name  = LM Studio가 표시하는 모델 id
```

LM Studio와 일반 llama.cpp OpenAI-compatible 서버는 동일한 클라이언트 경로를 사용합니다.

### Gemini

```text
provider   = gemini
model_name = 사용할 Gemini 모델명
api_token_env = GEMINI_API_KEY
max_tokens = 4096
```

REST API를 직접 사용하므로 Gemini SDK 설치는 필요하지 않습니다.

## MiniMax H3 5개 모드

| AUTO 입력 | 모드 | 의미 |
|---|---|---|
| 참조 없음 | T2VA | USER_INTENT 중심 |
| `first_ir` | I2VA | 시작 이미지가 0초 앵커 |
| `first_ir + final_ir` | FL2VA | 시작/마지막 이미지 모두 앵커 |
| `final_ir` | L2VA | 마지막 이미지로 수렴 |
| `reference_pack` | Ref2VA | typed image/video/audio references |

Ref2VA 출력 순서는 `subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music`입니다.

### USER_INTENT_v1

`H3 User Intent Author`는 한국어/영어 자유 요청을 다음과 같은 구조로 바꿉니다.

- `summary`
- `requested_actions`
- `requested_camera`
- `requested_scene_progression`
- `requested_style`
- `requested_sound`
- `requested_music`
- `constraints`
- `dialogue`
- `start_states`
- `final_states`

각 clause는 원래 입력의 실제 부분 문자열을 `evidence`로 보관합니다. 모델이 구조화한 뒤 두 번째 의미 검증 호출로 누락·과잉 추가를 다시 검사합니다. 빠른 대신 한 번만 호출하는 모드는 아직 구현하지 않았습니다.

### REFERENCE_PACK_v1

`H3 Reference Pack`은 실제 미디어 파일을 운반하지 않고 참조의 역할을 기록합니다.

```text
Picture 1 / image / identity / Subject 1
Picture 2 / image / wardrobe / Subject 1
Picture 3 / image / environment / Subject 1
Video 1   / video / motion / Subject 1
Audio 1   / audio / sound / Subject 1
```

실제 H3 생성 workflow에는 같은 번호의 이미지/비디오/오디오 자산을 별도로 연결해야 합니다.

## 노드 목록

| 노드 | 역할 |
|---|---|
| **Image IR Backend Config** | llama.cpp / OpenAI-compatible / Gemini 설정 |
| **Image IR Backend Control** | 단일 local llama-server 시작/중지/상태 |
| **ImageIR Llama.cpp Model Manager** | router mode 연결, 목록 새로고침, 모델 선택/load/unload, backend_config 출력 |
| **Image IR Analyzer** | IMAGE → IMAGE_IR |
| **Image IR Extractor Prompt** | 기본 이미지 분석 system prompt 출력 |
| **Image IR Write (manual)** | IMAGE_IR 직접 작성 |
| **Image IR Merge** | IMAGE_IR 병합 |
| **Image IR Inspect** | IR/불확실성/동작 가능성 확인 |
| **Image IR Prompt Composer** | 일반 CORE / DETAIL / FINAL grounded prompt |
| **Image IR Grounding Guard** | 기존 이미지 중심 prompt audit |
| **MiniMax H3 Prompt Composer** | legacy I2VA grounded composer |
| **H3 User Intent Author** | 자유 요청 → USER_INTENT_v1 |
| **H3 User Intent from JSON** | USER_INTENT_v1 직접 확인/수정 |
| **H3 Reference Pack** | Ref2VA 참조 역할 구성 |
| **H3 Mode Router** | AUTO/수동 5-mode 선택 |
| **MiniMax H3 Mode Composer (5 modes)** | 5개 H3 모드 deterministic composition |
| **ImageIR Provenance Guard** | IMAGE_IR / USER_INTENT / H3_RULE provenance 검증 |

## 설치 / 업데이트

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/ssain3d-lgtm/ComfyUI-ImageIR-PromptEngine
```

이미 설치했다면 repository 폴더에서:

```bat
git pull
```

추가 Python runtime dependency는 없습니다. 업데이트 후 ComfyUI를 재시작하세요.

## 예제 워크플로

- `example_workflows/image-ir-analyze-to-h3.json` — 이미지 분석 → legacy H3 → guard
- `example_workflows/image-ir-manual-prompt.json` — 수동 IMAGE_IR → 일반 grounded prompt
- `example_workflows/image-ir-audit-existing-prompt.json` — 기존 prompt audit
- `example_workflows/image-ir-h3-five-modes.json` — USER_INTENT + 5 H3 modes

## 현재 범위 밖

- Qwen verifier / automatic confidence routing
- automatic media editing / long-video split rendering
- SAM / pose estimation / crop automation
- model downloading
- arbitrary remote-host browser proxy

llama.cpp router **model list/select/load/unload**는 지원하지만 모델 파일 다운로드/설치는 하지 않습니다.

---

# English

## What this project separates

- **IMAGE_IR** = facts visible in the current reference image.
- **USER_INTENT** = what the user explicitly wants to happen next.
- **H3_RULE** = MiniMax H3 mode-specific formatting and timeline rules.
- **Provenance Guard** = rejects text that cannot be traced to one of those sources.

This keeps visual grounding strict without blocking valid requested future motion.

## Recommended I2VA graph

```text
Load Image → Image IR Analyzer → IMAGE_IR ──────────────┐
                                                        ├→ H3 Mode Router → MiniMax H3 Mode Composer (5 modes)
Freeform request → H3 User Intent Author → USER_INTENT ─┘                         ↓
                                                                     ImageIR Provenance Guard
```

Use either **Image IR Backend Config** or **ImageIR Llama.cpp Model Manager** as the reusable backend source.

## Grounding rules

1. Do not invent present-state visual facts absent from IMAGE_IR.
2. Do not change an observed attribute.
3. Do not sharpen an `uncertain` attribute into a specific candidate.
4. Do not convert viewer-relative laterality into subject-relative laterality unless confirmed.
5. Do not invent a pre-frame state for a still image.
6. Automatically generated motion must not contradict IMAGE_IR; explicit USER_INTENT future motion is handled separately.
7. Preserve uncertainty.
8. Preserve viewer-relative geometry when that is what IMAGE_IR provides.
9. Audit final visual claims against their provenance and reject untraced content.

## llama.cpp router model manager

For convenient multi-model switching, start a current llama.cpp server in router mode:

```bash
llama-server --models-dir /models --host 127.0.0.1 --port 8080
```

Then add **ImageIR Llama.cpp Model Manager** in ComfyUI:

```text
base_url = http://127.0.0.1:8080
Connect / Refresh
→ choose selected_model
→ Load / Unload
→ backend_config
```

The manager uses `GET /models`, `GET /models?reload=1`, `POST /models/load`, and `POST /models/unload`. Vision capability is read from router architecture metadata when available. The emitted object is the same `IMAGEIR_BACKEND` consumed by **Image IR Analyzer** and **H3 User Intent Author**.

The browser proxy is intentionally loopback-only. If the router requires authentication, store the key in the ComfyUI process environment and put only its variable name in `api_token_env`.

Existing single-model llama.cpp usage remains supported through **Image IR Backend Config** + **Image IR Backend Control**.

## Backend examples

```text
Existing llama.cpp:
provider = llama_cpp
server_mode = connect_existing
base_url = http://127.0.0.1:8080

LM Studio:
provider = openai_compatible
server_mode = connect_existing
base_url = http://127.0.0.1:1234

Gemini:
provider = gemini
api_token_env = GEMINI_API_KEY
```

`api_token` authenticates; `max_tokens` limits reply length. They are unrelated settings.

## H3 modes

| Inputs | AUTO mode |
|---|---|
| none | T2VA |
| first IR | I2VA |
| first + final IR | FL2VA |
| final IR | L2VA |
| reference pack | Ref2VA |

Ref2VA uses the ordered sections `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music`.

## Nodes

- **Image IR Backend Config** — provider/model/generation settings.
- **Image IR Backend Control** — start/stop/restart/status for a single local llama-server launched by this extension.
- **ImageIR Llama.cpp Model Manager** — connect/refresh/select/load/unload router models.
- **Image IR Analyzer** — IMAGE → IMAGE_IR.
- **Image IR Extractor Prompt** — exposes the shipped analyzer system prompt.
- **Image IR Write (manual)** — manual IMAGE_IR authoring.
- **Image IR Merge** — merges IR documents without silently overwriting observations.
- **Image IR Inspect** — inspect facts, uncertainty and available safe micro-motion.
- **Image IR Prompt Composer** — generic grounded CORE/DETAIL/FINAL prompt.
- **Image IR Grounding Guard** — legacy visual grounding audit.
- **MiniMax H3 Prompt Composer** — legacy I2VA H3 composer.
- **H3 User Intent Author** — Korean/English request → USER_INTENT_v1.
- **H3 User Intent from JSON** — inspect/edit/import USER_INTENT without a model call.
- **H3 Reference Pack** — typed Ref2VA asset roles.
- **H3 Mode Router** — AUTO/manual T2VA/I2VA/FL2VA/L2VA/Ref2VA routing.
- **MiniMax H3 Mode Composer (5 modes)** — five-mode H3 composition with provenance trace.
- **ImageIR Provenance Guard** — reconstructs from source objects and rejects untraced edits.

## Example workflows

- `example_workflows/image-ir-analyze-to-h3.json`
- `example_workflows/image-ir-manual-prompt.json`
- `example_workflows/image-ir-audit-existing-prompt.json`
- `example_workflows/image-ir-h3-five-modes.json`

## Development

```bash
ruff check .
python -m unittest discover -s tests -p "test_*.py" -v
```

External model/network calls are mocked in the test suite; real Windows + ComfyUI + llama.cpp router E2E remains a manual verification step.

## License

MIT
