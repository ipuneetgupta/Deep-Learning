const https = require('https');
const fs = require('fs');
const archiver = require('archiver');

const {API_KEY} = require("./env");

console.log(API_KEY)

const URL = `https://api.generated.photos/api/v1/faces?api_key=${API_KEY}&hair_length=short&per_page=100`;

let chunks = []

let objList = []

/**
 * Change directory paths appropriately according to your requirement then use   
 */
const req =  https.get(URL, (res) => {
    res.on('data',  (data) => {
        chunks.push(data);
    }).on('end', function() {
        const data = Buffer.concat(chunks);       
        // console.log('data came');
        
        const body = JSON.parse(data.toString());
        // console.log(body);
        console.log(body.faces.length);
        
        if(body.faces.length === 0) return;

        let i = 1;

        /**
          * take care of directory
          */
        fs.mkdir("./hair_length_dataset/short/images", (err) =>  {
            if(err){
                console.log("Error in Images folder creation" + err);
                return;
            }

            for(let face of body.faces){
                
                let obj = {};

                for(let metaI in face.meta){
                    if(Array.isArray(face.meta[metaI])){
                        obj[metaI] = face.meta[metaI][0];
                    }else{
                        obj[metaI] = face.meta[metaI];
                    }
                }
                

                for(let imgUrlObj of face.urls){
                    
                    for(let size in imgUrlObj){
                        const imgName = `${i}_${size}`;
                        
                        const curObj = {...obj};
                        curObj["image"] = `./images/${imgName}.jpg`;
                        
                        /**
                          * take care of directory
                          */
                        const file = fs.createWriteStream(`./hair_length_dataset/short/images/${imgName}.jpg`);
                        const request = https.get(imgUrlObj[size], (response) => response.pipe(file));

                        objList.push(curObj);
                    }
                }

                i++;
            }
        
            console.log(objList[0]);
            for(let t in objList[0]){
                console.log(`${t} : ${objList[0][t]}`)
            }

            /**
              ***** Code for generating CSVfile ****
              */
            const createCsvWriter = require('csv-writer').createObjectCsvWriter;
            const csvWriter = createCsvWriter({
                /**
                  * take care of directory
                  */
                path: './hair_length_dataset/short/dataset.csv',
                header: [
                    {id: 'image', title: 'Image'},
                    {id: 'emotion', title: 'Emotion'},
                    {id: 'hair_length', title: 'Hair_length'},
                    {id: 'hair_color', title: 'Hair_color'},
                    {id: 'eye_color', title: 'Eye'},
                    {id: 'gender', title: 'Gender'},
                    {id: 'ethnicity', title: 'Ethnicity'},
                    {id: 'age', title: 'Age'},
                    {id: 'confidence', title: 'Confidence'}
                ]
            });

            csvWriter
            .writeRecords(objList)
            .then(()=> console.log('The CSV file was written successfully'))
            .catch((e) => console.log('error occured'));

            /**
            **** Code for generating creating zip file (of images folder) ****
            */
             
            /**
              * take care of directory
              */
            const output = fs.createWriteStream('./hair_length_dataset/short/images.zip');
            var archive = archiver('zip');


            output.on('close', function () {
                console.log(archive.pointer() + ' total bytes');
                console.log('archiver has been finalized and the output file descriptor has closed.');
            });

            archive.on('error', function(err){
                throw err;
            });

            archive.pipe(output);
            /**
              * take care of directory
              */
            archive.directory('./hair_length_dataset/short/images/', 'images');

            archive.finalize();
        }) 
    });

})
