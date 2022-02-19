# neural-cloning


## Installation
1. Make sure you use Python 3.8.
2. Install the requirements:
```
$ pip install -r requirements/basic.txt
```
3. Install the package in editable mode:
```
$ pip install -e .
```

## Contributing
1. Install tools
```
$ pip install -r requirements/dev.txt
```
2. Add git commit hooks
```
$ pre-commit install
```

## Getting started
1. Train your first neural cloning model using
```
$ python src/cartpole/train.py
```
2. Check the generated directory (inside `outputs`) to see the training.
3. Now evaluate how well the cloner worked on a set of random actions:
```
$ python src/cartpole/evaluate.py PATH_TO_MODEL_DIRECTORY
```
4. Check whether the model looks alright. (A new directory in `outputs` should have appeared).
5. Now you can play a game!.
You can use `A` and `D` keyboard keys to apply force to the cart. Use `K` to end the game and save the trajectory (to yet another generated directory).
- To play using the real environment, use
```
$ python src/cartpole/generate_trajectory.py
```
- To play using the cloned environment, use
```
$ python src/cartpole/generate_trajectory.py PATH_TO_MODEL_DIRECTORY
```

