MLLM_PATH=$1

# OCG dataset

OCG_DATAPATH=$2
IMG_ASPECTS='[[1, 4], [9, 21], [9, 19], [1, 2], [9, 16], [4, 3], [16, 9], [2, 1], [21, 9], [4, 1]]'

python eval_mm/ocg/eval_MiniCPMV2_5.py \
    --mllm_path ${MLLM_PATH} \
    --data_path ${OCG_DATAPATH} \
    --image_aspects "${IMG_ASPECTS}" \
    --save-dir output/ocg/mv2_5 \
    --sampling \
    # --vis_flag

python eval_mm/ocg/eval_TAG.py \
    --mllm_path ${MLLM_PATH} \
    --data_path ${OCG_DATAPATH} \
    --image_aspects "${IMG_ASPECTS}" \
    --save-dir output/ocg/tag \
    --batchsize 4 \
    #--vis_flag


# ScreenSpot dataset
SCREENSPOT_IMGS=$3
SCREENSPOT_TEST=$4

python eval_mm/screenspot/eval_MiniCPMV2_5.py \
    --mllm_path ${MLLM_PATH} \
    --screenspot_imgs ${SCREENSPOT_IMGS} \
    --screenspot_test ${SCREENSPOT_TEST} \
    --save-dir output/screenspot/mv2_5 \
    #--vis_flag

python eval_mm/screenspot/eval_TAG.py \
    --mllm_path ${MLLM_PATH} \
    --screenspot_imgs ${SCREENSPOT_IMGS} \
    --screenspot_test ${SCREENSPOT_TEST} \
    --save-dir output/screenspot/tag \
    #--vis_flag

# Mind2Web dataset
MIND2WEB_DATAPATH=$5

TASKTYPES=(task website domain)
for TASK in "${TASKTYPES[@]}"
do
    python eval_mm/mind2web/eval_TAG.py \
        --mllm_path ${MLLM_PATH} \
        --data_dir ${MIND2WEB_DATAPATH} \
        --task ${TASK} \
        --num_history 4 \
        --filter_thresold 0.5 \
        --save-dir output/mind2web/tag \
        #--vis_flag

    python eval_mm/mind2web/eval_MiniCPMV2_5.py \
        --mllm_path ${MLLM_PATH} \
        --data_dir ${MIND2WEB_DATAPATH} \
        --task ${TASK} \
        --num_history 4 \
        --save-dir output/mind2web/mv2_5
done
