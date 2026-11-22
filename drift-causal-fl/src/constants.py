"""Torch-free constants shared by the simulator and the config layer (keeps
lightweight processes — sweep parent, status queries — from importing torch)."""
MECHANISMS = ("mcar", "mar", "mnar", "mnar_threshold")
METHODS = ("fedavg", "fedipw_oracle", "fedipw_estimated", "flfdms", "feddrift")
DRIFT_ESTIMATORS = ("model_relative", "model_independent")
