import Model
net = Model.EmotionNet(layers=[3, 4, 23, 3])
net.train_model('train', 'model-01', epochs=200, csvout='model-01-csv.csv')
net.load_checkpoint('best-model-01.pth.tar')
net.valid_model('valid')
net.test_model_show('test')
