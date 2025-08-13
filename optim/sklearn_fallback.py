"""
Scikit-learn fallback utilities for non-PyTorch workflows.

Provides a simple, trusted path to train linear models with SGD when users
do not use PyTorch. This is not a drop-in replacement for torch optimizers;
it is intended for numpy/scikit-learn pipelines.
"""
from typing import Any, Dict, Optional, Tuple


def train_with_sklearn_sgd(
	X,
	y,
	*,
	problem: Optional[str] = None,
	max_iter: int = 1000,
	learning_rate: str = "optimal",
	eta0: float = 0.01,
	penalty: str = "l2",
	alpha: float = 0.0001,
	random_state: Optional[int] = None,
	**kwargs: Dict[str, Any],
):
	"""
	Fit a scikit-learn SGD model.

	Args:
		X: Feature matrix (numpy array or compatible)
		y: Targets (classification labels or regression values)
		problem: 'classification'|'regression'|None to auto-detect
		max_iter: Max iterations
		learning_rate: Schedule (e.g., 'optimal', 'constant', 'invscaling', 'adaptive')
		eta0: Initial learning rate for schedules that require it
		penalty: Regularization ('l2', 'l1', 'elasticnet')
		alpha: Regularization strength
		random_state: RNG seed
		kwargs: Additional keyword args passed to sklearn estimator

	Returns:
		(model, metrics)
	"""
	try:
		import numpy as np
		from sklearn.linear_model import SGDClassifier, SGDRegressor
		from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
	except Exception as e:
		raise RuntimeError(
			"scikit-learn is required for train_with_sklearn_sgd. Install via `pip install scikit-learn`."
		) from e

	# Auto-detect problem type if not provided
	ptype = problem
	if ptype is None:
		# Heuristic: discrete integer labels -> classification; else regression
		unique = np.unique(y)
		ptype = "classification" if np.issubdtype(unique.dtype, np.integer) and unique.size <= 1000 else "regression"

	if ptype == "classification":
		model = SGDClassifier(
			loss="log_loss",
			max_iter=max_iter,
			learning_rate=learning_rate,
			eta0=eta0,
			penalty=penalty,
			alpha=alpha,
			random_state=random_state,
			**kwargs,
		)
		model.fit(X, y)
		pred = model.predict(X)
		metrics = {"accuracy": float(accuracy_score(y, pred))}
		return model, metrics
	elif ptype == "regression":
		model = SGDRegressor(
			max_iter=max_iter,
			learning_rate=learning_rate,
			eta0=eta0,
			penalty=penalty,
			alpha=alpha,
			random_state=random_state,
			**kwargs,
		)
		model.fit(X, y)
		pred = model.predict(X)
		metrics = {
			"r2": float(r2_score(y, pred)),
			"mse": float(mean_squared_error(y, pred)),
		}
		return model, metrics
	else:
		raise ValueError("Unknown problem type. Use 'classification' or 'regression'.")


