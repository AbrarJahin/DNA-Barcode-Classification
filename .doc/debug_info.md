## Get a model layer input shape-

model.layers[0].input_shape
model.layers[0].output_shape

### check if input shape are OK
i == 0....n-1
model.layers[i].output_shape == model.layers[i+1].input_shape
