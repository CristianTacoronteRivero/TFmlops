from tfmlops.breastcancer import BcSklearn
import mlflow
import session_info

mlflow.set_tracking_uri("http://127.0.0.1:5001")  # OJO CON EL PUERTO!
mlflow.set_experiment("GridSearchCV-BreastCancer")
mlflow.start_run()

RANDOM_STATE = 55
TEST_SIZE = 0.45

mlflow.log_param("seed.RANDOM_STATE_SPLIT", RANDOM_STATE)
mlflow.log_param("TEST_SIZE", TEST_SIZE)

ins = BcSklearn()

X, y = ins.get_X_y()
X_train, X_test, y_train, y_test = ins.split_train_test(
    random_state=RANDOM_STATE, test_size=TEST_SIZE
)
model = ins.build_pipeline(X_train)

model_opt = ins.search_and_fit(
    model,
    X_train,
    y_train,
    {"n_neighbors": [1, 2, 3], "p": [3, 4, 5]},
    {"n_estimators": [100, 200]},
)

r2, mse = ins.metricas_regresion(X_test, y_test)

mlflow.log_metrics({"test.r2": r2, "test.mse": mse})

ins.save_model("modelo.pkl")

mlflow.log_artifact("modelo.pkl")
mlflow.log_dict(model_opt.best_params_, "best_params.json")

session_info.show(write_req_file=True, req_file_name="requirements.txt")
mlflow.log_artifact("requirements.txt")

mlflow.end_run()
