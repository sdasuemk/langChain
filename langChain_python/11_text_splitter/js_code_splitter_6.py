"""
LangChain JavaScript/Node.js & React Code Text Splitter Tutorial
================================================================
This file demonstrates how to use the RecursiveCharacterTextSplitter 
to split JavaScript (Node.js/ES6) and React (JSX/TSX) component code 
based on JS/TS syntax rules.

Key Concepts:
- Language.JS & Language.TS: Sets the splitter to use syntax-specific 
  separators (e.g., function definitions, classes, object blocks, imports).
- Syntax-Aware Splitting: Keeps import blocks, functions, and React component 
  declarations intact, preventing arbitrary splits inside critical statements.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# ---------------------------------------------------------
# STEP 1: Define JavaScript / Node.js & React Source Code
# ---------------------------------------------------------
# 1. Express/NodeJS Controller Code
nodejs_code = """
const express = require('express');
const router = express.Router();

// Get user profile controller
router.get('/profile', async (req, res) => {
    try {
        const user = await UserService.findById(req.user.id);
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json(user);
    } catch (err) {
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

module.exports = router;
"""

# 2. React JSX Component Code
react_code = """
import React, { useState } from 'react';

export default function UserCard({ username, email }) {
    const [isFollowed, setIsFollowed] = useState(false);

    const handleFollow = () => {
        setIsFollowed(!isFollowed);
    };

    return (
        <div className="user-card">
            <h3>{username}</h3>
            <p>{email}</p>
            <button onClick={handleFollow}>
                {isFollowed ? 'Following' : 'Follow'}
            </button>
        </div>
    );
}
"""

# ---------------------------------------------------------
# STEP 2: Initialize JavaScript and React Splitter
# ---------------------------------------------------------
# We instantiate a splitter using the JS setting.
# The default JS separators split on classes, functions, block syntax, etc.
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=150,            # Small chunk size to demonstrate block-level splitting
    chunk_overlap=0,
)

# ---------------------------------------------------------
# STEP 3: Split JavaScript/Node.js Code
# ---------------------------------------------------------
print("=== Node.js/JS Chunking ===")
js_chunks = js_splitter.split_text(nodejs_code)
for i, chunk in enumerate(js_chunks):
    print(f"--- Node.js Chunk {i+1} ---")
    print(chunk.strip())
    print("-" * 40)

# ---------------------------------------------------------
# STEP 4: Split React Code
# ---------------------------------------------------------
# React (JSX) is written in JavaScript/TypeScript syntax.
# The JS splitter gracefully chunks React components keeping functions/JSX elements logical.
print("\n=== React JSX Component Chunking ===")
react_chunks = js_splitter.split_text(react_code)
for i, chunk in enumerate(react_chunks):
    print(f"--- React Chunk {i+1} ---")
    print(chunk.strip())
    print("-" * 40)
