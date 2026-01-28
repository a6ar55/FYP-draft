const express = require('express');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const User = require('../models/User');
const { protect } = require('../middleware/auth');

const router = express.Router();

// Generate JWT token
const signToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET || 'fallback-secret-key', {
    expiresIn: '7d'
  });
};

// Create and send token
const createSendToken = (user, statusCode, res) => {
  const token = signToken(user._id);

  res.status(statusCode).json({
    status: 'success',
    token,
    data: {
      user: {
        id: user._id,
        name: user.name,
        email: user.email
      }
    }
  });
};

// Check MongoDB connection
const checkMongoConnection = (res) => {
  if (mongoose.connection.readyState !== 1) {
    return res.status(503).json({
      status: 'error',
      message: 'Authentication service is unavailable. Please check MongoDB connection.',
      hint: 'Update MONGODB_URI in .env file or continue without authentication'
    });
  }
  return null;
};

// Signup
router.post('/signup', async (req, res) => {
  try {
    // Check MongoDB connection
    const connectionError = checkMongoConnection(res);
    if (connectionError) return connectionError;

    const { name, email, password } = req.body;

    // Check if user already exists (with timeout)
    const existingUser = await User.findOne({ email }).maxTimeMS(5000);
    if (existingUser) {
      return res.status(400).json({
        status: 'error',
        message: 'Email already registered'
      });
    }

    // Create new user
    const newUser = await User.create({
      name,
      email,
      password
    });

    createSendToken(newUser, 201, res);
  } catch (error) {
    if (error.name === 'MongooseError' || error.message.includes('buffering timed out')) {
      return res.status(503).json({
        status: 'error',
        message: 'Database connection timeout. Please check MongoDB connection.',
        hint: 'Update MONGODB_URI in .env file'
      });
    }
    
    res.status(400).json({
      status: 'error',
      message: error.message
    });
  }
});

// Login
router.post('/login', async (req, res) => {
  try {
    // Check MongoDB connection
    const connectionError = checkMongoConnection(res);
    if (connectionError) return connectionError;

    const { email, password } = req.body;

    // Check if email and password exist
    if (!email || !password) {
      return res.status(400).json({
        status: 'error',
        message: 'Please provide email and password'
      });
    }

    // Check if user exists and password is correct (with timeout)
    const user = await User.findOne({ email }).select('+password').maxTimeMS(5000);
    
    if (!user || !(await user.correctPassword(password, user.password))) {
      return res.status(401).json({
        status: 'error',
        message: 'Incorrect email or password'
      });
    }

    // Update last login
    user.lastLogin = Date.now();
    await user.save({ validateBeforeSave: false });

    createSendToken(user, 200, res);
  } catch (error) {
    if (error.name === 'MongooseError' || error.message.includes('buffering timed out')) {
      return res.status(503).json({
        status: 'error',
        message: 'Database connection timeout. Please check MongoDB connection.',
        hint: 'Update MONGODB_URI in .env file'
      });
    }
    
    res.status(400).json({
      status: 'error',
      message: error.message
    });
  }
});

// Get current user
router.get('/me', protect, async (req, res) => {
  res.status(200).json({
    status: 'success',
    data: {
      user: {
        id: req.user._id,
        name: req.user.name,
        email: req.user.email,
        createdAt: req.user.createdAt,
        lastLogin: req.user.lastLogin
      }
    }
  });
});

module.exports = router;
