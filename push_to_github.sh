#!/bin/bash
echo "Initializing Git repository..."
git init

echo "Adding remote repository..."
git remote add origin https://github.com/Pranesh-Ramachandran/Spam-Detection-using-naive-Bayes.git

echo "Adding all files..."
git add .

echo "Creating initial commit..."
git commit -m "Initial commit: Spam Detection using Naive Bayes project"

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Done! Project pushed to GitHub."
read -p "Press any key to continue..."