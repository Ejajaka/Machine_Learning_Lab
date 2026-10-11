# Machine Learning Lab

Python programs written for the Machine Learning lab sessions of my course, organised one folder per lab. Each lab is a set of standalone scripts (A1, A2, ...) that build a concept from scratch and print or plot the result.

## Labs

- **Lab 1** - Python warm-up, including a vowel/consonant counter and similar basics.
- **Lab 2** - Linear algebra for machine learning: building a feature matrix, checking its rank, and solving for weights with the pseudo-inverse, using `Lab_Session_Data.xlsx`.
- **Lab 3** - The same linear-algebra work, plus a second set of scripts run against a project dataset and a `simulation_500.csv` sample.
- **Lab 4** - Re-implementing common library functions by hand (label encoding, one-hot encoding, distance, dot product, norm, mean, variance). `A2/` adds `unittest` tests, and `A3/` benchmarks two k-means implementations, one written by hand against one produced with an AI assistant, plotting the runtime comparison. The AI-assisted workflow is written up in `prompt_log.txt`.
- **Lab 5** - k-nearest neighbours from scratch: Euclidean distance, neighbour search, and majority-vote prediction on `simulation_500.csv`.
- **Lab 6** - A tidier k-NN module with majority and distance-weighted voting, covered by a `pytest` suite.
- **Lab 7** - Information gain and decision-tree basics, including entropy over the `treat` column of `ACTG175.csv` and tree plotting.
- **Lab 8** - Neural network building blocks: weighted sums, activation functions (step, bipolar step, sigmoid, tanh, ReLU, leaky ReLU), and error terms.
- **Lab 9** - Perceptron and multi-layer perceptron classification with `RandomizedSearchCV` hyperparameter tuning on `ACTG175_X_train.csv`.

## Running the scripts

Each script reads its data from the same lab folder, so run it from there:

```bash
cd "Lab 5"
python A1.py
```

Install the packages the scripts rely on:

```bash
pip install numpy pandas scikit-learn matplotlib pytest openpyxl
```

Scripts that read `.xlsx` files need `openpyxl`, and the plots in Labs 4 and 6 need `matplotlib`.

## Notes

These are coursework exercises, so the code favours the building blocks taught in class over library shortcuts. A few scripts are written to demonstrate one idea at a time rather than as reusable modules, and later labs build on earlier ones.
