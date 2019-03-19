# structure of variable set definitions
- each variable set consists of a dictionary `variables`
- for each jet-tag category add a list to the `variables` dictionary containing the input features used for the training of this category, e.g.
```python
variables["4j_ge3t"] = [
    "some_variable",
    "some_other_variable",
    "vector_variable[1]",
    "vector_variable[2]"
    ]
```
- at the end of the file create a set of all variables which is used for preprocessing and jet-tag categories not specified
```python
all_variables = list(set( [v for key in variables for v in variables[key] ] )=
```
