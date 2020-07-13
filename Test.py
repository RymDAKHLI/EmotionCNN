import Model
net = Model.EmotionNet(layers=[3, 4, 23, 3])
net.test_model_show('test')
