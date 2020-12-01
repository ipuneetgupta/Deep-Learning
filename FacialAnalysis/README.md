# face-analyser-generator

## dataset-extractor

### Setup
(make sure you have node and npm installed on your machine)
- npm i
- node index.js (to extract dataset)

### how to read dataset

```javascript
└── dataset-generator
    ├── < facial-attribute >_dataset
    | └── < facial-attribute-type >
    |       ├── images.zip (Must extract this file before use)
    |       └── dataset.csv
    |        
    ├── index.js (can be used to extract more dataset from API)
    ├── env.js (create this file and keep you API_KEY here only)
```

#### Convention followed for images under images folder

We have 100 different images in /images and each image.jpg is available in 5 size (32, 64, 128, 256, 512)
