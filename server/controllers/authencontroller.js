// controllers/authController.js
import User from "../models/authModel.js";

// Register user
export const RegisterUser = async (req, res) => {
  try {
    const { email, password } = req.body;

    // check required fields
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "Email and Password are required",
      });
    }

    // check duplicate email
    const existingUser = await User.findOne({ where: { email } });
    if (existingUser) {
      return res.status(409).json({
        success: false,
        message: "Email already exists",
      });
    }

    // create new user
    const username = email.split("@")[0];
    const CUser = await User.create({ name: username, email, password });
    const newUser = {
      id: CUser.id,
      email: CUser.email,
      password: CUser.password,
      name: CUser.name,
    };

    return res
      .status(200)
      .json({ success: true, user: newUser, message: "Register success" });
  } catch (error) {
    console.error("Register error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error",
    });
  }
};

// Login user
export const LoginUser = async (req, res) => {
  try {
    const { email, password } = req.body;

    // check required fields
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "Email and Password are required",
      });
    }

    // find user
    const FUser = await User.findOne({ where: { email } });
    if (!FUser) {
      return res.status(401).json({
        success: false,
        message: "Email or Password is wrong",
      });
    }

    // check password
    if (FUser.password !== password) {
      return res.status(401).json({
        success: false,
        message: "Email or Password is wrong",
      });
    }

    // success
    const userData = {
      id: FUser.id,
      email: FUser.email,
      name: FUser.name ?? FUser.email.split("@")[0],
    };

    return res.json({
      success: true,
      user: userData,
      message: "Login success",
    });
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({
      success: false,
      message: "Server error",
    });
  }
};

