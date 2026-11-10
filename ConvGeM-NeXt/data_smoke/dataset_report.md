# Dataset report

## PlantVillage

- Classes: **38** (paper: 38)
- Images: **380** (paper: 54,284)
- Images per class: 10 – 10
- Missing official classes: none

| Class | Images |
|---|---|
| Apple___Apple_scab | 10 |
| Apple___Black_rot | 10 |
| Apple___Cedar_apple_rust | 10 |
| Apple___healthy | 10 |
| Blueberry___healthy | 10 |
| Cherry_(including_sour)___Powdery_mildew | 10 |
| Cherry_(including_sour)___healthy | 10 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 10 |
| Corn_(maize)___Common_rust_ | 10 |
| Corn_(maize)___Northern_Leaf_Blight | 10 |
| Corn_(maize)___healthy | 10 |
| Grape___Black_rot | 10 |
| Grape___Esca_(Black_Measles) | 10 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 10 |
| Grape___healthy | 10 |
| Orange___Haunglongbing_(Citrus_greening) | 10 |
| Peach___Bacterial_spot | 10 |
| Peach___healthy | 10 |
| Pepper,_bell___Bacterial_spot | 10 |
| Pepper,_bell___healthy | 10 |
| Potato___Early_blight | 10 |
| Potato___Late_blight | 10 |
| Potato___healthy | 10 |
| Raspberry___healthy | 10 |
| Soybean___healthy | 10 |
| Squash___Powdery_mildew | 10 |
| Strawberry___Leaf_scorch | 10 |
| Strawberry___healthy | 10 |
| Tomato___Bacterial_spot | 10 |
| Tomato___Early_blight | 10 |
| Tomato___Late_blight | 10 |
| Tomato___Leaf_Mold | 10 |
| Tomato___Septoria_leaf_spot | 10 |
| Tomato___Spider_mites Two-spotted_spider_mite | 10 |
| Tomato___Target_Spot | 10 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 10 |
| Tomato___Tomato_mosaic_virus | 10 |
| Tomato___healthy | 10 |

## PlantDoc

- Classes in both train/ and test/: **27** (paper: 27)
- Dropped (only in one split): ['Tomato two spotted spider mites leaf']
- Images: train **270**, test **81**
- Classes without a PlantVillage mapping (cross-corpus): none

| Class | Train | Test | PlantVillage counterpart |
|---|---|---|---|
| Apple Scab Leaf | 10 | 3 | Apple___Apple_scab |
| Apple leaf | 10 | 3 | Apple___healthy |
| Apple rust leaf | 10 | 3 | Apple___Cedar_apple_rust |
| Bell_pepper leaf | 10 | 3 | Pepper,_bell___healthy |
| Bell_pepper leaf spot | 10 | 3 | Pepper,_bell___Bacterial_spot |
| Blueberry leaf | 10 | 3 | Blueberry___healthy |
| Cherry leaf | 10 | 3 | Cherry_(including_sour)___healthy |
| Corn Gray leaf spot | 10 | 3 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| Corn leaf blight | 10 | 3 | Corn_(maize)___Northern_Leaf_Blight |
| Corn rust leaf | 10 | 3 | Corn_(maize)___Common_rust_ |
| Peach leaf | 10 | 3 | Peach___healthy |
| Potato leaf early blight | 10 | 3 | Potato___Early_blight |
| Potato leaf late blight | 10 | 3 | Potato___Late_blight |
| Raspberry leaf | 10 | 3 | Raspberry___healthy |
| Soyabean leaf | 10 | 3 | Soybean___healthy |
| Squash Powdery mildew leaf | 10 | 3 | Squash___Powdery_mildew |
| Strawberry leaf | 10 | 3 | Strawberry___healthy |
| Tomato Early blight leaf | 10 | 3 | Tomato___Early_blight |
| Tomato Septoria leaf spot | 10 | 3 | Tomato___Septoria_leaf_spot |
| Tomato leaf | 10 | 3 | Tomato___healthy |
| Tomato leaf bacterial spot | 10 | 3 | Tomato___Bacterial_spot |
| Tomato leaf late blight | 10 | 3 | Tomato___Late_blight |
| Tomato leaf mosaic virus | 10 | 3 | Tomato___Tomato_mosaic_virus |
| Tomato leaf yellow virus | 10 | 3 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| Tomato mold leaf | 10 | 3 | Tomato___Leaf_Mold |
| grape leaf | 10 | 3 | Grape___healthy |
| grape leaf black rot | 10 | 3 | Grape___Black_rot |

## Splits (seed 42)

- PlantVillage: train 266 / val 57 / test 57 → `PlantVillage_split_seed42.json`
- PlantDoc: train 222 / val 48 / test 81 → `PlantDoc_split_seed42.json`

**Status: READY**