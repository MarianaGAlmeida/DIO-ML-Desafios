
# Exemplo de Transfer Learning

## Resumo

Na 1ª Parte, 
Foi utilizada:

* [LINK](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb) : fonte com o código original completo.

Alterações:

-
-
-
-



Na 2ª Parte, duas categorias: Brontossauros e Stegossaurus (a escolha inspirada pelos meus sobrinhos). Formatação das imagens: 


## 1ª Parte: Construção do novo modelo a partir de VGG16

Após as importações (vide arquivo)

```py
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

```
Texto 
imagem resumo rede vgg


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

Texto
imagem resumo modelo novo



## 2ª Parte: 
```py
root = 'DIO_desafioTL_categories'

train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(root) if x[0]][1:]
print(categories)

```

['DIO_desafioTL_categories/brontosaurus', 'DIO_desafioTL_categories/stegosaurus']

```py
# helper function to load image and return it and input vector
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

```

```py
data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames 
              in os.walk(category) for f in filenames 
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        data.append({'x':np.array(x[0]), 'y':c})

# count the number of classes
num_classes = len(categories)
```


```py
random.shuffle(data)
```


```py
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]
```


```py
x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)
```
[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]

```py
# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)
```

```py
# summary
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

```

```py
history2 = model_new.fit(x_train, y_train, 
                         batch_size=10, 
                         epochs=10, 
                         validation_data=(x_val, y_val))
```



```py
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)

ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)

ax2.plot(history2.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()
```

IMAGEM acurácia



COMPARAR COM IMAGEM ACURÁCIA MODELO do zero




Exemplo imagens

![image](https://user-images.githubusercontent.com/93783315/143912595-e8fe17c3-f563-4794-a77c-cbd052e4c0ca.png)




![image](https://user-images.githubusercontent.com/93783315/143915435-01ca03c0-0aae-42c6-8553-18bd203e14a8.png)





