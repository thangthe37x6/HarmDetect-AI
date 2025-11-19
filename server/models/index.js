import { Sequelize } from "sequelize";
import dotenv from 'dotenv';
dotenv.config();
const password_portsql = process.env.password_portSQL || null
const sequelize = new Sequelize("postgres", "postgres", password_portsql, {
  host: "localhost",
  dialect: "postgres",
  logging: false, 
});

try {
  await sequelize.authenticate();
  console.log("✅ PostgreSQL connected");
} catch (err) {
  console.error("❌ DB connect error:", err);
}

export default sequelize;
