import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep, ProcessingStep, CacheConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionEquals
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.functions import Join
import os
from sagemaker.tensorflow import TensorFlow
from sagemaker.processing import TensorFlowProcessor
# --- 1. THIẾT LẬP CHUNG ---
pipeline_session = PipelineSession()
role = 'arn:aws:iam::339713121931:role/service-role/AmazonSageMaker-ExecutionRole-20251217T201397'
bucket = 'cat-dog-classification-bucket'
region = 'ap-southeast-1' 

cache_config = CacheConfig(enable_caching=True, expire_after="30d")

# Pipeline Parameters
skip_training = ParameterBoolean(
    name="SkipTraining",
    default_value=False
)

pretrained_model_path = ParameterString(
    name="PretrainedModelPath",
    default_value=f"s3://{bucket}/data/raw/output/models/default/model.tar.gz"  # Default path
)

# --- 2. BƯỚC HUẤN LUYỆN ---
estimator = TensorFlow(
    entry_point='./training/train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    framework_version='2.13',
    py_version='py310',
    sagemaker_session=pipeline_session,
    hyperparameters={
        'epochs': 3,
        'batch-size': 32,
        'learning-rate': 0.001
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'accuracy: ([0-9\\.]+)'},
        {'Name': 'validation:loss', 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_accuracy: ([0-9\\.]+)'},
    ]
)

step_train = TrainingStep(
    name="CatDog-Training",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/data/raw/train"
        )
    },
    cache_config=cache_config
)

# --- 3. BƯỚC ĐÁNH GIÁ (DUY NHẤT - dùng cho cả 2 trường hợp) ---
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation_metrics",
    path="evaluation.json"
)

tf_processor = TensorFlowProcessor(
    framework_version='2.13',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    py_version='py310',
    sagemaker_session=pipeline_session,
    base_job_name='catdog-eval'
)

step_eval = ProcessingStep(
    name="CatDog-Evaluation",
    step_args=tf_processor.run(
        code='./training/validate.py',
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=f"s3://{bucket}/data/raw/test",
                destination='/opt/ml/processing/input/data'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source='/opt/ml/processing/output'
            )
        ]
    ),
    property_files=[evaluation_report],
    cache_config=cache_config
)

# --- 4. ĐÁNH GIÁ CHO PRETRAINED MODEL ---
step_eval_pretrained = ProcessingStep(
    name="CatDog-Evaluation-Pretrained",
    step_args=tf_processor.run(
        code='./training/validate.py',
        inputs=[
            ProcessingInput(
                source=pretrained_model_path,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=f"s3://{bucket}/data/raw/test",
                destination='/opt/ml/processing/input/data'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source='/opt/ml/processing/output'
            )
        ]
    ),
    property_files=[evaluation_report],  # DÙNG CHUNG evaluation_report
    cache_config=cache_config
)

# --- 5. BƯỚC ĐĂNG KÝ MODEL ---
model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework='tensorflow',
        region=region,
        version='2.13',
        py_version='py310',
        image_scope='inference',
        instance_type='ml.m5.xlarge'
    ),
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point='./serving/serving.py',
    sagemaker_session=pipeline_session
)

step_model_reg = ModelStep(
    name="CatDog-RegisterModel",
    step_args=model.register(
        content_types=["application/x-image", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="CatDogModelGroup",
        approval_status="PendingManualApproval",
        description="Cat vs Dog classification model"
    )
)

# --- 6. ĐĂNG KÝ PRETRAINED MODEL ---
model_pretrained = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework='tensorflow',
        region=region,
        version='2.13',
        py_version='py310',
        image_scope='inference',
        instance_type='ml.m5.xlarge'
    ),
    model_data=pretrained_model_path,
    role=role,
    entry_point='serving.py',
    sagemaker_session=pipeline_session
)

step_model_reg_pretrained = ModelStep(
    name="CatDog-RegisterPretrainedModel",
    step_args=model_pretrained.register(
        content_types=["application/x-image", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="CatDogModelGroup",
        approval_status="PendingManualApproval",
        description="Cat vs Dog classification model - pretrained"
    )
)

# --- 7. ĐIỀU KIỆN ACCURACY - DUY NHẤT ---
# Tùy theo branch nào chạy, sẽ đọc metrics từ step tương ứng
# Nhưng vì cả 2 eval steps dùng chung evaluation_report name, nên chỉ cần 1 condition
# --- SỬA TRONG BƯỚC 7: ĐIỀU KIỆN ACCURACY ---
# --- 7. ĐIỀU KIỆN ACCURACY (Sửa đúng JsonPath) ---

# Cấu hình Condition cho New Model
cond_accuracy_new = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="metrics.accuracy" # Đã sửa khớp với file JSON của bạn
    ),
    right=0.80
)

step_check_new = ConditionStep(
    name="CheckAccuracy-NewModel",
    conditions=[cond_accuracy_new],
    if_steps=[step_model_reg],     # Nếu Pass -> Đăng ký model mới
    else_steps=[]
)

# Cấu hình Condition cho Pretrained Model
cond_accuracy_pretrained = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval_pretrained.name,
        property_file=evaluation_report,
        json_path="metrics.accuracy" # Đã sửa khớp với file JSON của bạn
    ),
    right=0.80
)

step_check_pretrained = ConditionStep(
    name="CheckAccuracy-Pretrained",
    conditions=[cond_accuracy_pretrained],
    if_steps=[step_model_reg_pretrained], # Nếu Pass -> Đăng ký model có sẵn
    else_steps=[]
)

# --- 8. ĐIỀU KIỆN CHÍNH (Điều hướng luồng chạy) ---
cond_skip = ConditionEquals(
    left=skip_training,
    right=True
)

step_main_condition = ConditionStep(
    name="TrainOrSkipDecision",
    conditions=[cond_skip],
    if_steps=[step_eval_pretrained, step_check_pretrained],
    else_steps=[step_train, step_eval, step_check_new]
)

# --- 10. TẠO VÀ CHẠY PIPELINE ---
pipeline = Pipeline(
    name="CatDog-EndToEnd-Pipeline",
    parameters=[skip_training, pretrained_model_path],
    steps=[step_main_condition],
    sagemaker_session=pipeline_session
)
role_arn = os.environ.get("SAGEMAKER_PIPELINE_ROLE_ARN")
print("Cập nhật bản thiết kế Pipeline lên SageMaker...")
pipeline.upsert(role_arn=role_arn)
definition = pipeline.definition()
with open("pipeline_definition.json", "w") as f:
    f.write(definition)
print("Hoàn tất! Pipeline đã được cập nhật thành công.")