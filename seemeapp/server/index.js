const express = require('express');
const cors = require('cors');
const multer = require('multer');
const bodyParser = require('body-parser');
const spawn = require('child_process').spawn;
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const Storage = multer.diskStorage({
  destination(req, file, callback) {
    callback(null, './images/uploads');
  },
  filename(req, file, callback) {
    callback(null, `${ file.fieldname }_${ Date.now() }_${ file.originalname }`);
  },
});

const limits = {
  files: 1, // allow only 1 file per request
  fileSize: 5 * 1024 * 1024, // 5 MB (max file size)
};

const upload = multer({
    storage: Storage,
    limits: limits
});

app.get('/', (req, res) => {
  res.status(200).send('You can post to /api/upload.');
});

app.post('/api/upload', upload.single('photo'), (req, res) => {

  const modelPath = path.join(__dirname, '..', 'SeeMePredictOnceNEU.py');
  const pythonProcess = spawn('python3', [modelPath, req.file.filename]);
  var res_image;
  var responseData = '';
  pythonProcess.stdout.on('data', (image) => {
    //res_image += image;
  });
  pythonProcess.on('close', async code => {
    console.log(responseData);
    let imagePath = path.join(__dirname, 'images/model_outputs/' + req.file.filename);
    let image = await fs.readFile(imagePath, 'base64');
    res.status(200).json({
      image: image
    });
  });
  pythonProcess.stdout.on('end', () => {
    //res.status(200).json({ image: res_image });
  });
  pythonProcess.stderr.on('data', data => {
    responseData += data.toString();
  });
});

app.listen(3000, () => {
  console.log('App running on http://localhost:3000');
});
