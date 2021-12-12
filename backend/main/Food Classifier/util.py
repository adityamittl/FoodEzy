import numpy as np
import tensorflow as tf
from PIL import Image

with open('./Saved Model/foodclassifier_model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the model architecture
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('./Saved Model/foodclassifier_weights.h5')
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="sparse_categorical_crossentropy", metrics=["accuracy"])

labels = ['adhirasam',
 'aloo_gobi',
 'aloo_matar',
 'aloo_methi',
 'aloo_shimla_mirch',
 'aloo_tikki',
 'anarsa',
 'ariselu',
 'bandar_laddu',
 'basundi',
 'bhatura',
 'bhindi_masala',
 'biryani',
 'boondi',
 'butter_chicken',
 'chak_hao_kheer',
 'cham_cham',
 'chana_masala',
 'chapati',
 'chhena_kheeri',
 'chicken_razala',
 'chicken_tikka',
 'chicken_tikka_masala',
 'chikki',
 'daal_baati_churma',
 'daal_puri',
 'dal_makhani',
 'dal_tadka',
 'dharwad_pedha',
 'doodhpak',
 'double_ka_meetha',
 'dum_aloo',
 'gajar_ka_halwa',
 'gavvalu',
 'ghevar',
 'gulab_jamun',
 'imarti',
 'jalebi',
 'kachori',
 'kadai_paneer',
 'kadhi_pakoda',
 'kajjikaya',
 'kakinada_khaja',
 'kalakand',
 'karela_bharta',
 'kofta',
 'kuzhi_paniyaram',
 'lassi',
 'ledikeni',
 'litti_chokha',
 'lyangcha',
 'maach_jhol',
 'makki_di_roti_sarson_da_saag',
 'malapua',
 'misi_roti',
 'misti_doi',
 'modak',
 'mysore_pak',
 'naan',
 'navrattan_korma',
 'palak_paneer',
 'paneer_butter_masala',
 'phirni',
 'pithe',
 'poha',
 'poornalu',
 'pootharekulu',
 'qubani_ka_meetha',
 'rabri',
 'ras_malai',
 'rasgulla',
 'sandesh',
 'shankarpali',
 'sheer_korma',
 'sheera',
 'shrikhand',
 'sohan_halwa',
 'sohan_papdi',
 'sutar_feni',
 'unni_appam']

def classify_image(file_path):
    image = Image.open(file_path) # reading the image
    img = np.asarray(image) # converting it to numpy array
    img = np.expand_dims(img, 0)
    predictions = model.predict(img) # predicting the class
    c = np.argmax(predictions[0]) # extracting the class with maximum probablity
    probab = float(round(predictions[0][c]*100, 2))

    result = {
        'plant': labels[c],
        'probablity': probab
    }

    return result

#print(classify_image('3.jpg'))