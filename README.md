
# Exemplo de Transfer Learning

## Resumo

Este projeto exemplifica a "transferência de aprendizado" de uma rede neural artificial para um novo modelo. O código original a partir do qual realizamos esta atividade pode ser encontrado no link abaixo:

* [LINK](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb) : fonte com o código original completo.

1ª Parte: construção de um novo modelo de classificação de imagens. Para tanto, são feitas algumas alterações em outro modelo já treinado com uma quantidade signficativa de imagens: o modelo VGG, cujos pesos foram salvos e disponivilizados ao público. 

```py
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

```

```py
# make a reference to VGG's input layer
inp = vgg.input
num_classes = 2
# make a new softmax layer with num_classes neurons
new_classification_layer = Dense(num_classes, activation='softmax')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(vgg.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)

```

Congelam-se os pesos de todas as camadas, exceto o da última, que continuará livre para o treinamento com nosso limitado conjunto de images (2 categorias, que totalizam apenas 100 imagens).

```py
# make all layers untrainable by freezing weights (except for last layer)
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

# ensure the last layer is trainable/not frozen
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

```


2ª Parte: treinamento do novo modelo com 2 conjuntos/categorias de imagens: Brontossauros e Stegossaurus. Mesmo com poucas imagens e apenas 10 epochs (devido a limitações de tempo), a acurácia do novo modelo foi de:


IMAGENS dinossauros



IMAGEM acurácia





Exemplo imagens

![image](https://)




![image](https://)



