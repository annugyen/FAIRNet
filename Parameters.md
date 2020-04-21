## Parameters
### Paramters can be displayed
Example 1:
```
model.add(Dense(200, activation='relu'))
```
Example 2:
```
def genre_classification_baby(input_shape=(128, 130), nb_genres=10):
    model = Sequential()
    model.add(Conv1D(filters=128, 
                     kernel_size=3, 
                     input_shape=input_shape, 
                     activation='relu', 
                     kernel_initializer='normal', 
                     padding='valid'))
    model.add(Dense(nb_genres, activation='softmax'))
```
Example 3:
```
model = genre_classification_baby(input_shape=(128, 388), nb_genres=5)
```
1. Explicit parameters, e.g. `200` in example 1, which is the parameter of this dense layer;
2. Explicit keywords, e.g. `activation='relu'` in example 1, which is the keyword of this dense layer;
3. Default value in function definition, e.g. `input_shape=(128, 130), nb_genres=10` in example 2, which are also keywords of layers;
4. Value of parameters and keywords when function is called, e.g. `input_shape=(128, 388), nb_genres=5` in example 3, which will replace the default value.

### Parameters cannot be displayed
1. External parameters, which are defined in other parts of code or even in other files;
2. Paramaters defined with expression, e.g. `w * 4` and `w = 16` in some cases;
3. Parameters defined in `class`.