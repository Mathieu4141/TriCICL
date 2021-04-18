TASK_NAME=split_mnist
METHOD=naive

cd ../../../..

pwd

DATETIME=$(date +%Y%m%d_%H%M%S)

JOB_DIR=gs://tricicl-public/packages/${DATETIME}

poetry build -f wheel

for METHOD in naive; do
  for SEED in 1; do
    JOB_NAME=${TASK_NAME}_${METHOD}_${SEED}_${DATETIME}

  #  gcloud ai-platform local train \
    echo "Starting job ${JOB_NAME}"
    gcloud ai-platform jobs submit training ${JOB_NAME} \
      --config src/research/mathieu/gcloud/split_mnist_demo.yaml \
      --labels task="${TASK_NAME}",method="${METHOD}" \
      --job-dir "${JOB_DIR}" \
      --packages dist/tricicl-0.1.0-py3-none-any.whl \
      --module-name research.mathieu.gcloud.demo_split_mnist \
      -- \
      --method-name ${METHOD} \
      --seed ${SEED}
  done
done

