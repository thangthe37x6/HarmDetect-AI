import { DataTypes } from "sequelize";
import sequelize from "./index.js";

const User = sequelize.define("User", {
    name: {type: DataTypes.STRING, allowNull: false, unique: true},
    email: {type: DataTypes.STRING, allowNull: false, unique: true},
    password: {type: DataTypes.STRING, allowNull: false}

}, {tableName: "users", timestamps: true})

export default User