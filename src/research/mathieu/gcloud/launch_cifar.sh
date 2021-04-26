
TASK_NAME=cifar100_10

cd ../../../..

DATETIME=$(date +%Y%m%d_%H%M%S)

JOB_DIR=gs://tricicl-public/packages/${DATETIME}

poetry build -f wheel


for METHOD in replay_avalanche; do
  for SEED in 1; do
    norm_method_name=${(L)METHOD//-/_}
    JOB_NAME=${TASK_NAME}_${norm_method_name}_${SEED}_${DATETIME}

    echo "Starting job ${JOB_NAME}"
    gcloud ai-platform jobs submit training "${JOB_NAME}" \
      --config src/research/mathieu/gcloud/job_gpu.yaml \
      --labels task="${TASK_NAME}",method="${norm_method_name}" \
      --job-dir "${JOB_DIR}" \
      --packages dist/tricicl-0.1.0-py3-none-any.whl \
      --module-name research.mathieu.gcloud.launch_cifar \
      -- \
      --method-name ${METHOD} \
      --seed ${SEED} \
      --n-classes-per-batch 10 \
      --memory-size 2000
  done
done

echo "See jobs at https://console.cloud.google.com/ai-platform/jobs?authuser=2&project=mathieu-tricicl"
poetry run tensorboard --logdir gs://tricicl-public/logs/tb/${TASK_NAME}
