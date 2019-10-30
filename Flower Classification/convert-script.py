# import coremltools
#
# coreml_model = coremltools.converters.caffe.convert(
# ('bvlc_alexnet.caffemodel', 'deploy.prototxt'),
# predicted_feature_name='class_labels.txt'
# )
#
# coreml_model.save('BVLCObjectClassifier.mlmodel')


import coremltools

caffe_model = ('oxford102.caffemodel', 'deploy.prototxt')

labels = 'flower-labels.txt'

coreml_model = coremltools.converters.caffe.convert(
    caffe_model,
    class_labels=labels,
    image_input_names='data'
)

coreml_model.save('FlowerClassifier.mlmodel')
