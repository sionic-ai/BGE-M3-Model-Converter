# torch_tf_validator.py
import argparse
import os
import numpy as np
import tensorflow as tf
import traceback

# =========================================================================
# 1. TF1 환경 설정 및 상수
# =========================================================================

# TF1 환경 보장 (TF2 환경에서 실행 시 Eager Execution 비활성화)
try:
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    print("INFO: Running in TF1 compatibility mode.")
    tf1 = tf.compat.v1
except Exception as e:
    print("INFO: Running in native TF1 mode (or V2 behavior already disabled).")
    tf1 = tf

# Constants for TF1 SavedModel loading
try:
    SAVED_MODEL_TAG = tf1.saved_model.tag_constants.SERVING  # "serve"
except AttributeError:
    # TF1 버전이 매우 낮을 경우 대비
    SAVED_MODEL_TAG = "serve"

SIGNATURE_KEY = "serving_default"


# =========================================================================
# 2. 검증 실행 함수 (격리된 그래프 사용)
# =========================================================================

# ★★★ 함수명이 run_validation으로 변경되었습니다. (이전: run_session_once) ★★★
def run_validation(model_dir: str, dtype=np.int32, batch=2, seqlen=12):
    print(f"[1] SavedModel 로드 및 시그니처 점검: {model_dir}")

    # ★★★★★★★★★★★ 핵심 수정 사항: 격리된 그래프 생성 ★★★★★★★★★★★
    # FailedPreconditionError의 원인인 이름 충돌(예: bge_m3_tensorflow_1)을 방지하기 위해,
    # 모델을 기본(Default) 그래프가 아닌, 완전히 격리된 새 그래프로 로드합니다.
    graph = tf.Graph()

    with graph.as_default():
        # 이 깨끗한 그래프와 연결된 세션을 생성합니다.
        config = tf1.ConfigProto()

        # ★★★ 세션에 명시적으로 그래프 연결 ★★★
        with tf1.Session(graph=graph, config=config) as sess:

            # --- 1단계: 모델 로드 ---
            print(f"Loading model into isolated graph...")
            try:
                # 모델을 깨끗한 그래프(graph)와 세션(sess)으로 로드 (TF1 loader 사용)
                # WARNING 메시지는 무시해도 됩니다 (TF2 환경에서 TF1 loader 사용 시 발생)
                meta_graph_def = tf1.saved_model.loader.load(
                    sess,
                    [SAVED_MODEL_TAG],
                    model_dir
                )
            except Exception as e:
                print(f"ERROR: Failed to load SavedModel from {model_dir}. Error: {e}")
                return

            # --- 2단계: 시그니처 점검 ---
            if SIGNATURE_KEY not in meta_graph_def.signature_def:
                print(f"ERROR: Signature '{SIGNATURE_KEY}' not found.")
                return

            signature_def = meta_graph_def.signature_def[SIGNATURE_KEY]
            print(f" - 사용 시그니처: {SIGNATURE_KEY}")

            # Shape 출력을 위한 헬퍼 함수 (TF1 방식)
            def format_shape(tensor_info):
                try:
                    # TensorShapeProto에서 shape 추출
                    return [d.size for d in tensor_info.tensor_shape.dim]
                except:
                    return "Unknown"

            print(" - 입력들:")
            for key, tensor_info in signature_def.inputs.items():
                print(
                    f"   • key='{key}', dtype={tf.dtypes.as_dtype(tensor_info.dtype).name}, shape={format_shape(tensor_info)}, name='{tensor_info.name}'")

            print(" - 출력들:")
            for key, tensor_info in signature_def.outputs.items():
                print(
                    f"   • key='{key}', dtype={tf.dtypes.as_dtype(tensor_info.dtype).name}, shape={format_shape(tensor_info)}, name='{tensor_info.name}'")

            # --- 3단계: 추론 테스트 실행 ---
            print(f"\n[2] TF1 세션으로 1회 추론 실행 (입력 dtype={dtype.__name__}, B={batch}, T={seqlen})")

            # 더미 입력 데이터 준비 (int32 요구)
            input_ids_data = np.random.randint(100, 10000, size=(batch, seqlen)).astype(dtype)
            attention_mask_data = np.ones((batch, seqlen)).astype(dtype)

            # 시그니처에서 입출력 텐서 이름 식별
            try:
                input_ids_tname = signature_def.inputs['input_ids'].name
                attention_mask_tname = signature_def.inputs['attention_mask'].name
                last_h_tname = signature_def.outputs['last_hidden_state'].name
                colbert_tname = signature_def.outputs['colbert_vecs'].name
            except KeyError as e:
                print(f"ERROR: Expected tensor key not found in signature: {e}")
                return

            feed_dict = {
                input_ids_tname: input_ids_data,
                attention_mask_tname: attention_mask_data,
            }

            fetches = [last_h_tname, colbert_tname]

            # 추론 실행
            try:
                print("Running session...")
                # ★★★ 그래프가 격리되었으므로 성공해야 합니다. ★★★
                last_h, colbert = sess.run(fetches, feed_dict=feed_dict)

                print("\n[SUCCESS] Inference successful!")
                print(f" - last_hidden_state shape: {last_h.shape}, dtype: {last_h.dtype}")
                print(f" - colbert_vecs shape: {colbert.shape}, dtype: {colbert.dtype}")

            except Exception as e:
                print(f"\n[FAILURE] ERROR during inference: {type(e).__name__}")
                if "FailedPreconditionError" in str(type(e)):
                    print("FailedPreconditionError가 여전히 발생했습니다.")
                    print("이는 모델 변환 과정(BGEM3WeightConverter.py)에서 이미 이름 불일치가 발생하여 저장되었음을 의미합니다.")
                    print("해결 방법: 모델 폴더(converted_bge_m3_tf1safe)를 삭제하고, 완전히 새로운 터미널에서 변환 스크립트를 다시 실행 후 검증하세요.")
                traceback.print_exc()


# =========================================================================
# 3. 실행 로직
# =========================================================================

def main():
    default_model_dir = "./converted_bge_m3_tf1safe"

    parser = argparse.ArgumentParser(description="Validate converted BGE-M3 TensorFlow SavedModel in TF1 environment.")
    parser.add_argument("--model_dir", type=str, default=default_model_dir,
                        help="Path to the SavedModel directory (e.g., converted_bge_m3_tf1safe)")
    args = parser.parse_args()

    # 경로 확인 로직 (model 하위 폴더 자동 탐색)
    model_path = args.model_dir

    # 1. 지정된 경로 확인
    if os.path.exists(os.path.join(model_path, "saved_model.pb")):
        pass  # 경로 정상
    # 2. 하위 'model' 폴더 확인
    elif os.path.exists(os.path.join(model_path, "model", "saved_model.pb")):
        model_path = os.path.join(model_path, "model")
    else:
        print(f"Error: saved_model.pb not found in {args.model_dir} or {os.path.join(args.model_dir, 'model')}.")
        return

    # ★★★ 수정된 함수 호출 ★★★
    run_validation(model_path, dtype=np.int32, batch=2, seqlen=12)


if __name__ == "__main__":
    main()