const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');

exports.protect = async (req, res, next) => {
  try {
    // If MongoDB is not connected, allow access without authentication (dev mode)
    if (mongoose.connection.readyState !== 1) {
      console.log('⚠ MongoDB not connected - allowing open access (dev mode)');
      req.user = { 
        _id: 'guest', 
        name: 'Guest User', 
        email: 'guest@stockai.com' 
      };
      return next();
    }

    const User = require('../models/User');

    // Get token from header
    let token;
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }

    if (!token) {
      return res.status(401).json({
        status: 'error',
        message: 'You are not logged in. Please log in to access this resource.'
      });
    }

    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key');

    // Check if user still exists (with timeout)
    const user = await User.findById(decoded.id).maxTimeMS(5000);
    if (!user) {
      return res.status(401).json({
        status: 'error',
        message: 'The user belonging to this token no longer exists.'
      });
    }

    // Grant access
    req.user = user;
    next();
  } catch (error) {
    if (error.name === 'MongooseError' || error.message.includes('buffering timed out')) {
      // MongoDB connection issue - fallback to open access
      console.log('⚠ MongoDB timeout - allowing open access');
      req.user = { 
        _id: 'guest', 
        name: 'Guest User', 
        email: 'guest@stockai.com' 
      };
      return next();
    }
    
    return res.status(401).json({
      status: 'error',
      message: 'Invalid token. Please log in again.'
    });
  }
};
