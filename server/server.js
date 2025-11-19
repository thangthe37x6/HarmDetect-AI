import express from 'express'
import cors from 'cors'
import auth_routes from './routes/authen.js'
import path from 'path'
import { fileURLToPath } from 'url'
import sequelize from './models/index.js'
import uploadroutes from './routes/mainevent.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const app = express()
const PORT = 3000

app.use(cors())
app.use(express.json())
app.use(express.static(path.join(__dirname, 'publics')))
app.use('uploads', express.static(path.join(__dirname, 'uploads')))


app.use('/', auth_routes)
app.use('/api', uploadroutes)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

sequelize.sync({
    alter: true
}).then(() => {
app.listen(PORT, () => {
    console.log(`🚀 VideoGuard Server chạy tại: http://localhost:${PORT}`);
} )
}).catch(err => console.log(' Database sync error:', err))