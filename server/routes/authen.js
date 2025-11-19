import express from 'express'
import { RegisterUser, LoginUser } from '../controllers/authencontroller.js'
const auth_routes = express.Router()


auth_routes.post('/api/login', LoginUser)
auth_routes.post('/api/register', RegisterUser)

export default auth_routes