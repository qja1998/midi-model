import streamlit as st
import argparse  # Streamlit에서는 직접 사용하지 않지만, 원본 구조 유지
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

# import tqdm # Streamlit에서는 st.progress 사용
from huggingface_hub import hf_hub_download
from transformers import DynamicCache
from safetensors.torch import load_file as safe_load_file

# --- 필요한 사용자 정의 모듈 임포트 ---
# (실제 파일들이 존재해야 합니다)
try:
    import MIDI
    from midi_model import MIDIModel, config_name_list, MIDIModelConfig
    from midi_synthesizer import MidiSynthesizer
    from midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2

    _modules_loaded = True
except ImportError as e:
    st.error(f"필수 모듈 로딩 실패: {e}. 해당 모듈이 설치되어 있는지 확인하세요.")
    _modules_loaded = False
# ------------------------------------

# --- 상수 및 초기 설정 ---
MAX_SEED = np.iinfo(np.int32).max
# OUTPUT_BATCH_SIZE = opt.batch # Streamlit에서는 opt 대신 고정값 또는 입력 사용
DEFAULT_BATCH_SIZE = 1  # Streamlit에서는 배치 처리를 단순화하기 위해 1로 시작하거나, 사용자 입력 받도록 수정 가능
number2drum_kits = {
    -1: "None",
    0: "Standard",
    8: "Room",
    16: "Power",
    24: "Electric",
    25: "TR-808",
    32: "Jazz",
    40: "Blush",
    48: "Orchestra",
}
patch2number = {v: k for k, v in MIDI.Number2patch.items()} if _modules_loaded else {}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = [
    "C♭",
    "A♭m",
    "G♭",
    "E♭m",
    "D♭",
    "B♭m",
    "A♭",
    "Fm",
    "E♭",
    "Cm",
    "B♭",
    "Gm",
    "F",
    "Dm",
    "C",
    "Am",
    "G",
    "Em",
    "D",
    "Bm",
    "A",
    "F♯m",
    "E",
    "C♯m",
    "B",
    "G♯m",
    "F♯",
    "D♯m",
    "C♯",
    "A♯m",
]
key_sig_map_to_index = {name: i + 1 for i, name in enumerate(key_signatures)}
key_sig_map_to_index["auto"] = 0

# --- 모델 및 토크나이저 전역 변수 (Streamlit에서는 session_state 사용 권장) ---
# model: Optional[MIDIModel] = None
# tokenizer: Union[MIDITokenizerV1, MIDITokenizerV2, None] = None

# --- 세션 상태 초기화 ---
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "output_midi_seq" not in st.session_state:
    st.session_state.output_midi_seq = None  # 생성된 MIDI 시퀀스 저장 (배치 고려 필요)
if "continuation_state" not in st.session_state:
    st.session_state.continuation_state = [0]  # [int] or [list]
if "last_seed" not in st.session_state:
    st.session_state.last_seed = 0
if "output_files" not in st.session_state:
    st.session_state.output_files = []  # 생성된 파일 경로 저장
if "output_audio_data" not in st.session_state:
    st.session_state.output_audio_data = []  # 생성된 오디오 데이터 저장

# --- Helper Functions (Gradio 코드에서 가져옴, 일부 수정) ---


@st.cache_resource  # 사운드폰트 및 신디사이저는 한 번만 로드
def get_synthesizer():
    try:
        soundfont_path = hf_hub_download(
            repo_id="skytnt/midi-model", filename="soundfont.sf2"
        )
        return MidiSynthesizer(soundfont_path)
    except Exception as e:
        st.error(f"사운드폰트 로딩 또는 신디사이저 초기화 실패: {e}")
        return None


@st.cache_resource  # 스레드 풀도 한 번만 생성
def get_thread_pool(max_workers=DEFAULT_BATCH_SIZE):
    return ThreadPoolExecutor(max_workers=max_workers)


synthesizer = get_synthesizer()
thread_pool = get_thread_pool()


@st.cache_data  # 파일 목록은 캐싱 가능
def get_model_paths():
    ckpt_files = glob.glob("**/*.ckpt", recursive=True)
    bin_files = glob.glob("**/*.bin", recursive=True)
    safetensors_files = glob.glob("**/*.safetensors", recursive=True)
    print(ckpt_files, bin_files, safetensors_files)
    model_paths = sorted(ckpt_files + bin_files + safetensors_files)
    model_paths = [
        model_path for model_path in model_paths if "adapter_model" not in model_path
    ]
    return model_paths


@st.cache_data
def get_lora_paths():
    lora_paths = sorted(glob.glob("**/adapter_config.json", recursive=True))
    lora_paths = [str(Path(lora_path).parent) for lora_path in lora_paths]
    return [""] + lora_paths  # 로라 미사용 옵션 추가


def load_model(path, model_config, lora_path, device):
    if not _modules_loaded:
        return "필수 모듈 로딩 실패"
    try:
        if model_config == "auto":
            config_path = Path(path).parent / "config.json"
            if config_path.exists():
                config = MIDIModelConfig.from_json_file(config_path)
            else:
                return "config.json 파일을 찾을 수 없습니다. 설정을 직접 지정해주세요."
        else:
            config = MIDIModelConfig.from_name(model_config)

        model_instance = MIDIModel(config=config)
        tokenizer_instance = model_instance.tokenizer

        st.write(f"'{path}' 로딩 중...")
        if path.endswith(".safetensors"):
            state_dict = safe_load_file(path, device="cpu")  # CPU로 먼저 로드
        else:
            ckpt = torch.load(path, map_location="cpu")  # CPU로 먼저 로드
            state_dict = ckpt.get("state_dict", ckpt)

        model_instance.load_state_dict(state_dict, strict=False)
        st.write("모델 가중치 로드 완료.")

        if lora_path:
            st.write(f"LoRA '{lora_path}' 병합 중...")
            model_instance = model_instance.load_merge_lora(
                lora_path
            )  # load_merge_lora가 해당 경로를 처리한다고 가정
            st.write("LoRA 병합 완료.")

        # 모델을 지정된 장치로 이동하고 평가 모드로 설정
        model_dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        model_instance.to(device, dtype=model_dtype).eval()
        st.write(f"모델을 {device} ({model_dtype})로 이동 및 평가 모드 설정 완료.")

        # 세션 상태에 저장
        st.session_state.model = model_instance
        st.session_state.tokenizer = tokenizer_instance
        st.session_state.model_loaded = True
        return "모델 로딩 성공!"

    except Exception as e:
        st.session_state.model = None
        st.session_state.tokenizer = None
        st.session_state.model_loaded = False
        return f"모델 로딩 실패: {e}"


@torch.inference_mode()
def generate(
    prompt=None,
    batch_size=1,
    max_len=512,
    temp=1.0,
    top_p=0.98,
    top_k=20,
    disable_patch_change=False,
    disable_control_change=False,
    disable_channels=None,
    generator=None,
    progress_bar=None,
):  # progress_bar 추가

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    device = next(model.parameters()).device  # 모델에서 현재 device 가져오기

    if tokenizer is None or model is None:
        st.error("모델 또는 토크나이저가 로드되지 않았습니다.")
        return iter([])  # 빈 iterator 반환

    if disable_channels is not None:
        # disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels] # 원본 코드: c는 채널 번호 (0-15)
        # tokenizer.parameter_ids["channel"] 의 구조 확인 필요. list 또는 dict 일 수 있음.
        # 만약 list이고 인덱스가 채널 번호라면:
        valid_channel_ids = tokenizer.parameter_ids.get("channel", [])
        if valid_channel_ids:
            disable_channel_ids = [
                valid_channel_ids[c]
                for c in disable_channels
                if c < len(valid_channel_ids)
            ]
        else:
            disable_channel_ids = []

    else:
        disable_channel_ids = []

    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        # 기본 프롬프트 생성 (BOS 토큰)
        input_tensor = torch.full(
            (batch_size, 1, max_token_seq),
            tokenizer.pad_id,
            dtype=torch.long,
            device=device,
        )
        input_tensor[:, 0, 0] = tokenizer.bos_id
    else:
        # 제공된 프롬프트 사용
        if isinstance(prompt, list):  # 토크나이즈된 리스트 형태일 경우
            prompt = np.asarray(prompt, dtype=np.int64)

        if (
            len(prompt.shape) == 2
        ):  # (seq_len, token_dim) 형태 -> (batch, seq_len, token_dim)
            prompt = prompt[None, ...]
            if prompt.shape[0] != batch_size:
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif len(prompt.shape) == 3:  # (batch, seq_len, token_dim)
            if prompt.shape[0] != batch_size:
                # 배치 크기 조정 (예: 첫번째 샘플 반복)
                prompt = np.repeat(prompt[:1], repeats=batch_size, axis=0)
        else:
            st.error(f"잘못된 프롬프트 형태: {prompt.shape}")
            return iter([])

        # 시퀀스 길이 제한 및 패딩
        prompt = prompt[:, :, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(
                prompt,
                ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                mode="constant",
                constant_values=tokenizer.pad_id,
            )

        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=device)

    # 모델 입력 길이 제한 (예: 4096) - 모델 설정에 따라 조절 필요
    # input_tensor = input_tensor[:, -4096:] # 이 부분은 모델 구조에 따라 다를 수 있음
    input_tensor = input_tensor[
        :, -model.config.n_positions // max_token_seq :
    ]  # 예시: 모델의 최대 position 기반

    cur_len = input_tensor.shape[1]  # 현재 이벤트(토큰 시퀀스) 수
    total_gen_len = max_len  # 목표 총 이벤트 수

    # --- 생성 루프 ---
    cache1 = DynamicCache()
    past_len = 0
    generated_events_count = 0

    # Streamlit 진행률 표시기 업데이트
    if progress_bar:
        progress_bar.progress(0.0)

    while cur_len < total_gen_len:
        end = [False] * batch_size
        # 다음 context window 계산 (슬라이딩 윈도우 방식 고려)
        current_input = input_tensor[:, past_len:]
        hidden = model.forward(current_input, cache=cache1)[
            :, -1
        ]  # 마지막 hidden state 사용

        next_token_seq = None
        event_names = [""] * batch_size
        cache2 = DynamicCache()  # 토큰 내부 생성용 캐시

        # 각 이벤트 내 토큰 생성 (최대 max_token_seq 만큼)
        for i in range(max_token_seq):
            mask = torch.zeros(
                (batch_size, tokenizer.vocab_size), dtype=torch.bool, device=device
            )  # bool 마스크 사용 권장

            # 마스킹 로직 (어떤 토큰을 생성할 수 있는지 제한)
            for b in range(batch_size):
                if end[b]:  # 이미 EOS가 나온 배치는 PAD만 가능하게
                    mask[b, tokenizer.pad_id] = True
                    continue

                if i == 0:  # 첫 토큰: 이벤트 타입 또는 EOS
                    mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                    if disable_patch_change:
                        if tokenizer.event_ids.get("patch_change") in mask_ids:
                            mask_ids.remove(tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        if tokenizer.event_ids.get("control_change") in mask_ids:
                            mask_ids.remove(tokenizer.event_ids["control_change"])
                    mask[b, mask_ids] = True
                else:  # 이벤트 파라미터 토큰
                    if not event_names[
                        b
                    ]:  # 이전 스텝에서 이벤트가 결정되지 않은 경우 (오류 상황?)
                        mask[b, tokenizer.pad_id] = True
                        continue

                    param_names = tokenizer.events.get(event_names[b], [])
                    if i > len(param_names):  # 현재 이벤트의 파라미터 개수 초과 시 PAD
                        mask[b, tokenizer.pad_id] = True
                        continue

                    param_name = param_names[i - 1]
                    param_mask_ids = tokenizer.parameter_ids.get(param_name, [])

                    if param_name == "channel":
                        param_mask_ids = [
                            pid
                            for pid in param_mask_ids
                            if pid not in disable_channel_ids
                        ]

                    if param_mask_ids:
                        mask[b, param_mask_ids] = True
                    else:  # 가능한 파라미터 값이 없으면 PAD
                        mask[b, tokenizer.pad_id] = True

            # 로짓 계산 및 샘플링
            logits = model.forward_token(
                hidden, next_token_seq[:, -1:] if i > 0 else None, cache=cache2
            )[:, -1]
            # 마스크 적용 (확률 0으로 만들기) - 마스크가 True인 곳만 살림
            logits[~mask] = -float("inf")

            # 온도 적용 및 확률 계산
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)

            # Top-k / Top-p 샘플링
            samples = model.sample_top_p_k(
                probs, top_p, top_k, generator=generator
            )  # sample_top_p_k 구현 필요

            # 생성된 토큰 처리
            if i == 0:
                next_token_seq = samples
                for b in range(batch_size):
                    if end[b]:
                        continue
                    eid = samples[b].item()
                    if eid == tokenizer.eos_id:
                        end[b] = True
                        event_names[b] = "EOS"  # EOS 표시
                    elif eid in tokenizer.id_events:
                        event_names[b] = tokenizer.id_events[eid]
                    else:  # 유효하지 않은 이벤트 ID가 샘플링된 경우 (오류 처리)
                        end[b] = True
                        event_names[b] = "Error"
                        next_token_seq[b] = tokenizer.pad_id  # PAD로 강제 변경
            else:
                next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                # 모든 활성 배치가 현재 이벤트의 파라미터 생성을 완료했는지 확인
                all_params_done = True
                for b in range(batch_size):
                    if not end[b]:
                        event_name = event_names[b]
                        if event_name and event_name != "EOS" and event_name != "Error":
                            expected_len = len(tokenizer.events.get(event_name, []))
                            if i <= expected_len:  # 아직 파라미터 생성 중
                                all_params_done = False
                                break
                        elif not event_name:  # 이벤트 결정 안됨 (오류)
                            pass
                if all_params_done:
                    break  # 현재 이벤트 생성 완료

        # 생성된 토큰 시퀀스 패딩
        if next_token_seq.shape[1] < max_token_seq:
            next_token_seq = F.pad(
                next_token_seq,
                (0, max_token_seq - next_token_seq.shape[1]),
                "constant",
                value=tokenizer.pad_id,
            )

        # (batch_size, max_token_seq) -> (batch_size, 1, max_token_seq)
        next_token_seq_unsqueeze = next_token_seq.unsqueeze(1)

        # 입력 텐서에 추가
        input_tensor = torch.cat([input_tensor, next_token_seq_unsqueeze], dim=1)

        past_len = cur_len  # 이전 길이 업데이트 (주의: 슬라이딩 윈도우 방식이면 다름)
        cur_len += 1  # 생성된 이벤트 수 증가
        generated_events_count += 1

        # 진행률 업데이트
        progress_percentage = min(
            1.0,
            generated_events_count
            / (total_gen_len - (input_tensor.shape[1] - generated_events_count)),
        )
        if progress_bar:
            progress_bar.progress(progress_percentage)

        # 생성된 토큰 시퀀스 반환 (CPU로 이동)
        yield next_token_seq.cpu().numpy()

        if all(end):  # 모든 배치가 EOS 생성 완료
            st.write("모든 배치에서 EOS 생성 완료.")
            break

    # 최종 진행률 100%
    if progress_bar:
        progress_bar.progress(1.0)


def synthesis_task(mid_score):
    if synthesizer:
        # MIDI.score2opus가 정의되어 있어야 함
        try:
            opus_data = MIDI.score2opus(mid_score)
            return synthesizer.synthesis(opus_data)
        except Exception as e:
            st.error(f"오디오 합성 중 오류: {e}")
            return None
    return None


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎵 MIDI 생성 모델 (Streamlit 버전)")

if not _modules_loaded:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 모델 설정")
    model_paths = get_model_paths()
    selected_model_path = st.selectbox(
        "모델 파일 선택", model_paths, index=0 if model_paths else -1
    )
    model_configs = ["auto"] + config_name_list
    selected_model_config = st.selectbox("모델 설정 선택", model_configs, index=0)
    lora_paths = get_lora_paths()
    selected_lora_path = st.selectbox("LoRA 경로 선택 (선택 사항)", lora_paths, index=0)
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.selectbox("실행 장치", device_options, index=0)

    if st.button("모델 로드"):
        with st.spinner("모델을 로드하는 중입니다..."):
            load_msg = load_model(
                selected_model_path,
                selected_model_config,
                selected_lora_path,
                selected_device,
            )
            st.info(load_msg)

    if st.session_state.model_loaded:
        st.success(
            f"모델 '{Path(selected_model_path).name}' 로드됨 ({selected_device})"
        )
    else:
        st.warning("모델이 로드되지 않았습니다. 모델을 로드해주세요.")

    st.header("🎹 오디오 합성")
    render_audio_checkbox = st.checkbox("생성 후 오디오 렌더링", value=True)


# --- Main Area ---
if not st.session_state.model_loaded:
    st.warning("모델을 로드한 후 생성 옵션을 사용할 수 있습니다.")
    st.stop()

# --- Input Tabs ---
tab1, tab2, tab3 = st.tabs(["Custom Prompt", "MIDI Prompt", "Last Output Prompt"])

with tab1:
    st.subheader("Custom Prompt (처음부터 생성)")
    input_instruments = st.multiselect(
        "🪗 악기 선택 (없으면 자동)",
        options=list(patch2number.keys()),
        max_selections=15,
    )
    input_drum_kit = st.selectbox(
        "🥁 드럼 킷 선택", options=list(drum_kits2number.keys()), index=0
    )  # Default "None"
    input_bpm = st.slider("BPM (0이면 자동)", 0, 255, 120)
    input_time_sig = st.radio(
        "박자표 (v2 모델용)",
        options=[
            "auto",
            "4/4",
            "2/4",
            "3/4",
            "6/4",
            "7/4",
            "2/2",
            "3/2",
            "4/2",
            "3/8",
            "5/8",
            "6/8",
            "7/8",
            "9/8",
            "12/8",
        ],
        index=0,
        horizontal=True,
    )
    input_key_sig = st.radio(
        "조성 (v2 모델용)", options=["auto"] + key_signatures, index=0, horizontal=True
    )
    selected_tab = 0

with tab2:
    st.subheader("MIDI Prompt (입력 MIDI 파일 기반)")
    input_midi_file = st.file_uploader(
        "MIDI 파일 업로드 (.mid, .midi)", type=["mid", "midi"]
    )
    input_midi_events = st.slider(
        "프롬프트로 사용할 초기 MIDI 이벤트 수 (최대 4096)", 1, 4097, 128
    )
    # MIDI 처리 옵션
    col1, col2 = st.columns(2)
    with col1:
        input_reduce_cc_st = st.checkbox(
            "Control Change/Set Tempo 이벤트 줄이기", value=True
        )
        input_add_default_instr = st.checkbox(
            "악기 없는 채널에 기본 악기 추가", value=True
        )
    with col2:
        input_remap_track_channel = st.checkbox("트랙/채널 재매핑", value=True)
        input_remove_empty_channels = st.checkbox("빈 채널 제거", value=False)
    selected_tab = 1


with tab3:
    st.subheader("Last Output Prompt (이전 생성 결과 이어하기)")
    st.write("마지막 생성 결과를 이어서 생성합니다.")
    # Gradio의 Radio와 달리, Streamlit에서는 버튼 등으로 트리거 필요
    # continuation_options = ["새로 시작"] + [f"결과 {i+1} 이어하기" for i in range(DEFAULT_BATCH_SIZE)] # 배치 지원 시 필요
    # input_continuation_select = st.radio("이어할 출력 선택", options=continuation_options, index=0) # 여기서는 단순화

    can_undo = len(st.session_state.continuation_state) >= 2
    if st.button("마지막 이어하기 취소", disabled=not can_undo):
        if can_undo:
            # 상태 복원 로직
            if isinstance(
                st.session_state.continuation_state[-1], list
            ):  # 이전 시퀀스 저장됨
                st.session_state.output_midi_seq = st.session_state.continuation_state[
                    -1
                ]
            else:  # 길이만 저장됨
                prev_len = st.session_state.continuation_state[-1]
                if st.session_state.output_midi_seq:
                    # 주의: output_midi_seq가 배치 형태일 수 있음 [[seq1], [seq2]]
                    st.session_state.output_midi_seq = [
                        ms[:prev_len] for ms in st.session_state.output_midi_seq
                    ]

            st.session_state.continuation_state = st.session_state.continuation_state[
                :-1
            ]
            st.success("마지막 이어하기를 취소했습니다.")
            # UI 갱신을 위해 rerun 필요
            st.rerun()
        else:
            st.warning("취소할 이전 상태가 없습니다.")

    selected_tab = 2


# --- Generation Parameters ---
st.divider()
st.subheader("🎲 생성 파라미터")
col1, col2 = st.columns(2)
with col1:
    input_gen_events = st.slider("생성할 최대 MIDI 이벤트 수", 1, 4096, 512)
    input_seed = int(
        st.number_input(
            "Seed (무작위는 -1)",
            min_value=-1,
            max_value=MAX_SEED,
            value=st.session_state.last_seed if st.session_state.last_seed != 0 else -1,
        )
    )  # Use -1 for random
    use_random_seed = input_seed == -1
with col2:
    input_temp = st.slider("Temperature", 0.1, 1.2, 1.0, 0.01)
    input_top_p = st.slider("Top-p", 0.1, 1.0, 0.94, 0.01)
    input_top_k = st.slider("Top-k", 1, 128, 20, 1)
    input_allow_cc = st.checkbox("MIDI CC 이벤트 허용", value=True)

# --- Generate Button ---
st.divider()
if st.button(
    "🚀 MIDI 생성 시작!", type="primary", disabled=not st.session_state.model_loaded
):

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    if not model or not tokenizer:
        st.error("모델이 로드되지 않았습니다!")
        st.stop()

    # --- 입력 준비 ---
    current_seed = np.random.randint(0, MAX_SEED) if use_random_seed else input_seed
    st.session_state.last_seed = current_seed  # 다음 실행을 위해 저장
    generator = torch.Generator(next(model.parameters()).device).manual_seed(
        current_seed
    )
    st.write(f"Seed: {current_seed}")

    batch_size = DEFAULT_BATCH_SIZE  # 배치 크기
    disable_patch_change = False
    disable_channels = None
    prompt_np = None  # 모델에 전달될 최종 numpy 프롬프트

    # 선택된 탭에 따라 프롬프트 준비
    if selected_tab == 0:  # Custom Prompt
        st.write("Custom Prompt 선택됨")
        prompt_list = [
            [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)
        ]  # BOS 이벤트
        initial_events_str = ["[BOS]"]

        if tokenizer.version == "v2":
            time_sig = input_time_sig
            if time_sig != "auto":
                time_sig_nn, time_sig_dd = map(int, time_sig.split("/"))
                time_sig_dd_val = {2: 1, 4: 2, 8: 3}.get(
                    time_sig_dd, 2
                )  # 기본값 4/4의 dd=2
                event = [
                    "time_signature",
                    0,
                    0,
                    0,
                    time_sig_nn - 1,
                    time_sig_dd_val - 1,
                ]
                prompt_list.append(tokenizer.event2tokens(event))
                initial_events_str.append(f"TimeSig({time_sig})")

            key_sig_name = input_key_sig
            if key_sig_name != "auto":
                key_sig_index = key_sig_map_to_index.get(key_sig_name, 0)
                if key_sig_index != 0:
                    key_sig = key_sig_index - 1
                    key_sig_sf = key_sig // 2 - 7
                    key_sig_mi = key_sig % 2
                    event = ["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]
                    prompt_list.append(tokenizer.event2tokens(event))
                    initial_events_str.append(f"KeySig({key_sig_name})")

        if input_bpm > 0:
            event = ["set_tempo", 0, 0, 0, input_bpm]
            prompt_list.append(tokenizer.event2tokens(event))
            initial_events_str.append(f"Tempo({input_bpm})")

        patches = {}
        instr_channel_idx = 0
        for instr_name in input_instruments:
            if instr_name in patch2number:
                if instr_channel_idx == 9:
                    instr_channel_idx = 10  # 드럼 채널 건너뛰기
                if instr_channel_idx < 16:
                    patches[instr_channel_idx] = patch2number[instr_name]
                    instr_channel_idx += 1
                else:
                    st.warning("악기는 최대 15개까지 설정 가능합니다 (드럼 제외).")
                    break

        if input_drum_kit != "None":
            if input_drum_kit in drum_kits2number:
                patches[9] = drum_kits2number[input_drum_kit]  # 드럼은 채널 9 고정

        if patches:
            disable_patch_change = (
                True  # 사용자가 악기 설정 시, 이후 생성에서 patch_change 막기
            )
            disable_channels = [
                c for c in range(16) if c not in patches
            ]  # 설정되지 않은 채널 비활성화
            for ch, patch_num in patches.items():
                event = ["patch_change", 0, 0, 0, ch, patch_num]
                prompt_list.append(tokenizer.event2tokens(event))
                instr_name = MIDI.Number2patch.get(patch_num, f"P{patch_num}")
                initial_events_str.append(f"Patch(Ch={ch}, P={instr_name})")

        prompt_np = np.asarray(
            [prompt_list] * batch_size, dtype=np.int64
        )  # (batch, n_events, max_token_seq)
        st.session_state.output_midi_seq = prompt_np.tolist()  # 초기 프롬프트를 저장
        st.session_state.continuation_state = [0]  # 상태 초기화
        st.write(f"초기 프롬프트 이벤트: {' '.join(initial_events_str)}")

    elif selected_tab == 1 and input_midi_file is not None:  # MIDI Prompt
        st.write("MIDI Prompt 선택됨")
        midi_bytes = input_midi_file.getvalue()
        try:
            # MIDI.midi2score 구현 필요
            score = MIDI.midi2score(midi_bytes)
            # 토크나이즈 옵션 적용
            eps = 4 if input_reduce_cc_st else 0
            tokenized_events = tokenizer.tokenize(
                score,
                cc_eps=eps,
                tempo_eps=eps,
                remap_track_channel=input_remap_track_channel,
                add_default_instr=input_add_default_instr,
                remove_empty_channels=input_remove_empty_channels,
            )

            # 사용할 이벤트 수 제한
            max_prompt_events = (
                input_midi_events
                if input_midi_events <= 4096
                else len(tokenized_events)
            )
            prompt_events = tokenized_events[:max_prompt_events]

            if not prompt_events:
                st.error("MIDI 파일에서 유효한 이벤트를 토큰화하지 못했습니다.")
                st.stop()

            # BOS 추가 및 패딩 처리 필요? -> 토크나이저가 BOS 포함하는지 확인
            # Gradio 코드는 BOS를 직접 추가하지 않음. tokenize 결과를 그대로 사용.
            # 단, 각 이벤트는 max_token_seq 길이로 변환되어야 함. tokenize 함수 확인 필요.
            # 만약 tokenize가 (n_events, max_token_seq) 형태 반환 시:
            prompt_list = prompt_events  # 이미 적절한 형태라고 가정
            prompt_np = np.asarray([prompt_list] * batch_size, dtype=np.int64)

            # 만약 tokenize가 [[tok1, tok2..], [tokA, tokB..]] 형태 반환 시 패딩 필요:
            # padded_prompt_list = []
            # for event_tokens in prompt_events:
            #     padded = event_tokens[:tokenizer.max_token_seq]
            #     if len(padded) < tokenizer.max_token_seq:
            #         padded += [tokenizer.pad_id] * (tokenizer.max_token_seq - len(padded))
            #     padded_prompt_list.append(padded)
            # prompt_np = np.asarray([padded_prompt_list] * batch_size, dtype=np.int64)

            st.session_state.output_midi_seq = prompt_np.tolist()
            st.session_state.continuation_state = [0]  # 상태 초기화
            st.write(
                f"입력 MIDI 파일에서 {prompt_np.shape[1]}개의 이벤트를 프롬프트로 사용합니다."
            )

        except Exception as e:
            st.error(f"MIDI 파일 처리 중 오류 발생: {e}")
            st.stop()

    elif selected_tab == 2:  # Last Output Prompt
        st.write("Last Output Prompt 선택됨")
        if st.session_state.output_midi_seq:
            # 이전 결과 사용
            prompt_np = np.asarray(st.session_state.output_midi_seq, dtype=np.int64)
            # Gradio의 continuation_select 로직 단순화: 무조건 마지막 결과 전체를 사용
            # 이전 상태 저장 (undo용)
            if isinstance(
                st.session_state.continuation_state[-1], list
            ):  # 이미 undo한 경우
                # 상태 업데이트 없이 진행
                pass
            else:  # 길이 저장
                st.session_state.continuation_state.append(
                    prompt_np.shape[1]
                )  # 현재 길이를 저장
            st.write(
                f"이전 생성 결과 ({prompt_np.shape[1]} 이벤트)를 이어받아 생성합니다."
            )
        else:
            st.warning("이어할 이전 생성 결과가 없습니다. Custom Prompt로 생성합니다.")
            selected_tab = 0  # 강제로 Custom Prompt로 전환 (또는 오류 처리)
            # Custom Prompt 로직 재실행 또는 기본값 사용
            prompt_list = [
                [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)
            ]
            prompt_np = np.asarray([prompt_list] * batch_size, dtype=np.int64)
            st.session_state.output_midi_seq = prompt_np.tolist()
            st.session_state.continuation_state = [0]

    else:  # 잘못된 탭 선택 또는 MIDI 파일 미업로드 시
        st.error("유효한 프롬프트가 준비되지 않았습니다.")
        st.stop()

    # --- 생성 실행 ---
    if prompt_np is not None:
        st.write("MIDI 생성을 시작합니다...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        max_len = prompt_np.shape[1] + input_gen_events  # 목표 총 이벤트 수

        generated_sequences = [
            list(seq) for seq in prompt_np.tolist()
        ]  # 생성 결과 누적 (배치 처리)

        try:
            midi_generator = generate(
                prompt=prompt_np,
                batch_size=batch_size,
                max_len=max_len,
                temp=input_temp,
                top_p=input_top_p,
                top_k=input_top_k,
                disable_patch_change=disable_patch_change,
                disable_control_change=not input_allow_cc,
                disable_channels=disable_channels,
                generator=generator,
                progress_bar=progress_bar,
            )

            gen_start_time = time.time()
            generated_event_count = 0
            for i, next_token_batch in enumerate(midi_generator):
                # next_token_batch shape: (batch_size, max_token_seq)
                for b in range(batch_size):
                    generated_sequences[b].append(next_token_batch[b].tolist())
                generated_event_count += 1
                status_text.text(
                    f"이벤트 생성 중... ({generated_event_count}/{input_gen_events})"
                )

            gen_end_time = time.time()
            st.success(
                f"MIDI 생성 완료! (소요 시간: {gen_end_time - gen_start_time:.2f}초)"
            )
            st.session_state.output_midi_seq = generated_sequences  # 최종 결과 업데이트

            # --- 결과 처리 및 출력 ---
            st.divider()
            st.subheader("🎶 생성 결과")

            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            st.session_state.output_files = []
            st.session_state.output_audio_data = []

            audio_futures = []

            cols = st.columns(batch_size)  # 배치 수 만큼 컬럼 생성

            for i in range(batch_size):
                with cols[i]:
                    st.markdown(f"**결과 {i+1}**")
                    final_event_sequence = st.session_state.output_midi_seq[i]

                    # 1. MIDI 파일 저장 및 다운로드 버튼
                    try:
                        # tokenizer.detokenize 구현 필요
                        mid_score = tokenizer.detokenize(final_event_sequence)
                        # MIDI.score2midi 구현 필요
                        midi_bytes = MIDI.score2midi(mid_score)

                        output_filename = f"output_{i+1}_{int(time.time())}.mid"
                        output_path = os.path.join(output_dir, output_filename)
                        with open(output_path, "wb") as f:
                            f.write(midi_bytes)
                        st.session_state.output_files.append(output_path)

                        st.download_button(
                            label=f"결과 {i+1} MIDI 다운로드",
                            data=midi_bytes,
                            file_name=output_filename,
                            mime="audio/midi",
                        )

                        # 2. 오디오 렌더링 (선택 사항)
                        if render_audio_checkbox and synthesizer and thread_pool:
                            future = thread_pool.submit(synthesis_task, mid_score)
                            audio_futures.append(future)
                        else:
                            audio_futures.append(None)  # 오디오 생성 안 함

                    except Exception as e:
                        st.error(f"결과 {i+1} 처리 중 오류: {e}")
                        audio_futures.append(None)

            # 3. 오디오 출력 (렌더링 완료 후)
            if render_audio_checkbox:
                with st.spinner("오디오를 렌더링하는 중입니다..."):
                    for i in range(batch_size):
                        with cols[i]:
                            future = audio_futures[i]
                            if future:
                                try:
                                    audio_result = (
                                        future.result()
                                    )  # 스레드 작업 완료 기다림
                                    if audio_result:
                                        # audio_result 형식 확인 필요 (샘플링 레이트, 데이터)
                                        # 예: (44100, opus_bytes) 또는 numpy 배열 등
                                        # synthesizer.synthesis 결과 형식에 맞춰 처리
                                        # 여기서는 opus_bytes를 받았다고 가정하고 st.audio 사용
                                        sampling_rate = 44100  # 가정
                                        st.audio(
                                            audio_result,
                                            format="audio/opus",
                                            sample_rate=sampling_rate,
                                        )
                                        st.session_state.output_audio_data.append(
                                            audio_result
                                        )
                                    else:
                                        st.warning(f"결과 {i+1} 오디오 생성 실패")
                                        st.session_state.output_audio_data.append(None)
                                except Exception as e:
                                    st.error(f"결과 {i+1} 오디오 렌더링 중 오류: {e}")
                                    st.session_state.output_audio_data.append(None)
                            else:
                                st.info(f"결과 {i+1} 오디오 렌더링 건너뜀")
                                st.session_state.output_audio_data.append(None)

        except Exception as e:
            st.error(f"MIDI 생성 중 오류 발생: {e}")
            import traceback

            st.code(traceback.format_exc())  # 디버깅을 위해 전체 Traceback 출력

# 앱 종료 시 스레드 풀 종료 (선택적)
# Streamlit은 일반적으로 상태를 유지하므로 명시적 종료가 필수는 아님
# def cleanup():
#     if thread_pool:
#         thread_pool.shutdown()
# import atexit
# atexit.register(cleanup)
