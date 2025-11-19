import express from 'express';
import multer from 'multer';
import fs from 'fs/promises'; // ✅ dùng bản promise
import path from 'path';
import { fileURLToPath } from 'url';
import { uploadvideo, checkstatus, getresult, deletevideo } from '../controllers/maincontroller.js';


// ✅ Tạo __dirname cho môi trường ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uploadroutes = express.Router();

// ✅ Cấu hình Multer storage
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
    try {
      await fs.mkdir(uploadDir, { recursive: true }); 
      cb(null, uploadDir);
    } catch (error) {
      cb(error);
    }
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only video files are allowed.'));
    }
  }
});

// ✅ Đăng ký routes
uploadroutes.post('/upload', upload.single('video'), uploadvideo);
uploadroutes.get('/status/:VideoId', checkstatus);
uploadroutes.get('/result/:VideoId', getresult);
uploadroutes.delete('/video/:VideoId', deletevideo);
export default uploadroutes;
