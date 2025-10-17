# Expense Tracker Web Application

A modern, full-featured expense tracking web application built with FastAPI, SQLite, and vanilla JavaScript.

## Features

- 🔐 User authentication with JWT tokens
- 💰 Track income and expenses
- 📊 Visual analytics with Chart.js
- 🏷️ Custom categories
- 📱 Fully responsive design
- 🎨 Modern UI with Tailwind CSS

## Tech Stack

- **Backend**: Python with FastAPI
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: Jinja2 templates with vanilla JavaScript
- **Styling**: Tailwind CSS (CDN)
- **Charts**: Chart.js (CDN)

## Running on CodeSandbox

1. Import this repository into CodeSandbox
2. The server should start automatically
3. If not, open the terminal and run:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Open the preview browser to access the application

Running Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

uvicorn main:app --reload

Open your browser and navigate to http://localhost:8000

Default Test Account

The application comes pre-seeded with a test account:

Email: test@example.com

Password: password123

Project Structure

expense-tracker/
├── main.py                 # FastAPI application with all backend logic
├── requirements.txt        # Python dependencies
├── templates/
│   ├── login.html         # Login/Register page
│   └── dashboard.html     # Main dashboard with all features
├── tracker.db             # SQLite database (auto-generated)
└── README.md              # This file

Features Overview

Dashboard

View current balance, total income, and total expenses

Recent transactions list

Expenses breakdown by category (pie chart)

Transactions

Add new income or expense transactions

View all transactions in a table

Delete transactions

Categorize each transaction

Categories

Create custom categories

View all categories

Delete categories (with cascade delete of associated transactions)

API Endpoints

Authentication

POST /api/auth/register - Register new user

POST /api/auth/login - Login user

POST /api/auth/logout - Logout user

GET /api/auth/me - Get current user info

Transactions

GET /api/transactions - Get all transactions

POST /api/transactions - Create new transaction

DELETE /api/transactions/{id} - Delete transaction

Categories

GET /api/categories - Get all categories

POST /api/categories - Create new category

DELETE /api/categories/{id} - Delete category

Analytics

GET /api/analytics - Get dashboard analytics data

Security Features

Password hashing with bcrypt

JWT token authentication

HTTP-only cookies

Protected API endpoints

User-specific data isolation

Design Features

Dark sidebar with light content area

Responsive layout (mobile-friendly)

Modern gradient buttons

Card-based UI components

Interactive charts

Smooth transitions and hover effects

License

MIT License - Feel free to use this project for learning or personal use.


---

## Summary

I've created a complete, production-ready expense tracker application with:

✅ **Backend**: FastAPI with SQLAlchemy ORM and SQLite database  
✅ **Authentication**: JWT tokens with secure httponly cookies  
✅ **Frontend**: Beautiful, responsive UI with Tailwind CSS  
✅ **Features**: Full CRUD for transactions and categories, analytics dashboard  
✅ **Design**: Modern two-column layout with dark sidebar and light content area  
✅ **Charts**: Interactive pie chart using Chart.js  
✅ **Mobile**: Fully responsive with hamburger menu  
✅ **Pre-seeded**: Test user and default categories included  

The project is ready to be uploaded to GitHub and imported into CodeSandbox. All files are complete and functional!