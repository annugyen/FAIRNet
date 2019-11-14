# NN Architecture extraction
<font size=3>The task ”extract neural network architecture from code” consists of following:</font>
  - the function for the extraction right know can be found under DataCollection/DataCollection/ModelLoader.py. The function you can modify is extract_architecture_from_python(). You can either re-use some parts of the code or write your own.
  - Under HiWiExercise/Architecture/FAIRnets_Layers_wrong.xlsx you can find some examples to check your code. More examples in the respective repositories are in the files files.json and filtered_data.json. With load_file.py you can read the json files. You just need to save the GitHub repository link and the architectures (like in files.json the repo_url and py_data). Start with the simple architectures. If the architecture is too complex, e.g. with loops, you can skip them at the beginning. We can talk about those cases later.
  - So far the function extract_architecture_from_python()
     - just saves the first model architecture of the code. You should extend this, so all architectures in the .py file is saved.
     - searches for ”Sequential" and ".compile()" to extract the NN architecture (see line 245). You should extent it with the keras.applications [1] because some NNs are based on for example ResNet and you should extent the end of an architecture with "return model” as some are not using ”.compile” (see HiWiExercise/Architecture/FAIRnets_Layers_wrong.xlsx).
     - does not display the parameter ”units of neurons” correctly if the unit is indicated as variable. The default if the unit is not indicated is ”0” which is in case of ”dropout” incorrect. There the number of neurons is (1-rate)*number_of_neurons_of_previous_layer.
     - does not save the information about the metrics. Please include this (e.g. ”binary_crossentropy”, ”accuracy”), see [2].

[1] https://keras.io/applications/  
[2] https://keras.io/metrics/