from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta, date
from typing import Optional, List
import csv
import io
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import httpx
import asyncio
from lxml import etree as ET

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 43200  # 30 days

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tracker.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    categories = relationship("Category", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="categories")
    transactions = relationship("Transaction", back_populates="category")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    description = Column(String)
    type = Column(String)  # "income" or "expense"
    category_id = Column(Integer, ForeignKey("categories.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="transactions")
    category = relationship("Category", back_populates="transactions")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class CategoryCreate(BaseModel):
    name: str

class CategoryUpdate(BaseModel):
    name: str

class CategoryResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TransactionCreate(BaseModel):
    amount: float
    description: str
    type: str
    category_id: int

class TransactionUpdate(BaseModel):
    amount: float
    description: str
    type: str
    category_id: int
    created_at: Optional[datetime] = None

class TransactionResponse(BaseModel):
    id: int
    amount: float
    description: str
    type: str
    category_id: int
    created_at: datetime
    category_name: Optional[str] = None
    
    class Config:
        from_attributes = True

# FastAPI app
app = FastAPI(title="Expense Tracker")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def on_startup():
    """
    Принудительно загружает курсы валют при старте приложения.
    Если загрузка не удалась, приложение продолжит работу с курсами по умолчанию.
    """
    print("Application startup: Fetching initial exchange rates...")
    await fetch_and_cache_rates()
    print("Application startup complete.")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Seed database
def seed_database():
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(User).filter(User.email == "test@example.com").first()
        if not existing_user:
            # Create default user
            hashed_password = pwd_context.hash("password123")
            user = User(email="test@example.com", hashed_password=hashed_password)
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Create default categories
            categories = [
                Category(name="Food", user_id=user.id),
                Category(name="Transport", user_id=user.id),
                Category(name="Salary", user_id=user.id),
                Category(name="Entertainment", user_id=user.id),
                Category(name="Utilities", user_id=user.id),
            ]
            db.add_all(categories)
            db.commit()
            
            print("Database seeded successfully!")
    finally:
        db.close()

# Seed on startup
seed_database()

# --- Exchange Rate Logic ---
exchange_rate_cache = {
    "rates": {
        "USD": 84.5,
        "EUR": 92.0,
        "RUB": 1.1,
        "KZT": 0.19,
        "CNY": 12.0,
    },
    "last_updated": None
}

async def fetch_and_cache_rates():
    """
    Асинхронно загружает курсы валют и обновляет кэш.
    Эта функция предназначена для фонового выполнения.
    """
    today = date.today()
    target_codes = ["USD", "EUR", "RUB", "KZT", "CNY"]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://www.nbkr.kg/XML/daily.xml", timeout=10)
            response.raise_for_status()
            
            xml_content = response.content
            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(xml_content, parser=parser)
            
            new_rates = {}
            for currency in root.findall('Currency'):
                code = currency.get('ISOCode')
                if code in target_codes:
                    value_str = currency.find('Value').text
                    nominal_str = currency.find('Nominal').text
                    rate = float(value_str.replace(',', '.')) / float(nominal_str.replace(',', '.'))
                    new_rates[code] = rate
            
            if len(new_rates) == len(target_codes):
                exchange_rate_cache["rates"] = new_rates
                exchange_rate_cache["last_updated"] = today
                print(f"Successfully updated exchange rates in background: {new_rates}")
            else:
                print("Could not find all target currencies in NBKR response.")

    except (httpx.RequestError, ET.ParseError, AttributeError) as e:
        print(f"Background task failed to fetch new exchange rate. Using cached/default rate. Error: {e}")

def get_exchange_rates(background_tasks: BackgroundTasks):
    """
    Возвращает курсы валют из кэша и, если они устарели,
    запускает фоновую задачу для их обновления.
    """
    if exchange_rate_cache["last_updated"] != date.today():
        print("Exchange rates are outdated. Fetching in background.")
        background_tasks.add_task(fetch_and_cache_rates)
    return exchange_rate_cache["rates"]

# Auth utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
    except JWTError:
        return None
    
    user = db.query(User).filter(User.email == email).first()
    return user

def require_auth(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# Routes - HTML Pages
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Простой эндпоинт для проверки работоспособности сервиса."""
    return {"status": "ok"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# API Routes - Auth
@app.post("/api/auth/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = User(email=user_data.email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create default categories for new user
    default_categories = [
        Category(name="Food", user_id=user.id),
        Category(name="Transport", user_id=user.id),
        Category(name="Salary", user_id=user.id),
        Category(name="Entertainment", user_id=user.id),
        Category(name="Utilities", user_id=user.id),
    ]
    db.add_all(default_categories)
    db.commit()
    
    return {"message": "User created successfully"}

@app.post("/api/auth/login")
async def login(user_data: UserLogin, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    
    return {"message": "Login successful"}

@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
async def get_me(user: User = Depends(require_auth)):
    return {"email": user.email, "id": user.id}

# API Route - Exchange Rate
@app.get("/api/rate")
async def get_rate(background_tasks: BackgroundTasks):
    return {"rates": get_exchange_rates(background_tasks)}

# API Routes - Categories
@app.get("/api/categories", response_model=List[CategoryResponse])
async def get_categories(user: User = Depends(require_auth), db: Session = Depends(get_db)):
    categories = db.query(Category).filter(Category.user_id == user.id).all()
    return categories

@app.post("/api/categories", response_model=CategoryResponse)
async def create_category(
    category_data: CategoryCreate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    category = Category(name=category_data.name, user_id=user.id)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category

@app.delete("/api/categories/{category_id}")
async def delete_category(
    category_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    category = db.query(Category).filter(
        Category.id == category_id,
        Category.user_id == user.id
    ).first()
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    db.delete(category)
    db.commit()
    return {"message": "Category deleted successfully"}

@app.put("/api/categories/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: int,
    category_data: CategoryUpdate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    db_category = db.query(Category).filter(
        Category.id == category_id,
        Category.user_id == user.id
    ).first()

    if not db_category:
        raise HTTPException(status_code=404, detail="Категория не найдена")

    db_category.name = category_data.name
    db.commit()
    db.refresh(db_category)

    return db_category


# API Routes - Transactions
@app.get("/api/transactions")
async def get_transactions(
    category_id: Optional[int] = None,
    page: int = 1,
    limit: int = 10,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    query = db.query(Transaction).filter(
        Transaction.user_id == user.id
    )

    if category_id:
        query = query.filter(Transaction.category_id == category_id)
    
    if start_date:
        query = query.filter(Transaction.created_at >= start_date)

    if end_date:
        # Включаем весь день до 23:59:59
        end_of_day = datetime.combine(end_date, datetime.max.time())
        query = query.filter(Transaction.created_at <= end_of_day)

    # Получаем общее количество для пагинации перед применением limit/offset
    total_count = query.count()

    transactions = query.order_by(Transaction.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    
    result = []
    for t in transactions:
        result.append({
            "id": t.id,
            "amount": t.amount,
            "description": t.description,
            "type": t.type,
            "category_id": t.category_id,
            "category_name": t.category.name if t.category else None,
            "created_at": t.created_at.isoformat()
        })
    
    return {
        "items": result,
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": (total_count + limit - 1) // limit
    }

@app.get("/api/transactions/export")
async def export_transactions(
    category_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    query = db.query(Transaction).filter(
        Transaction.user_id == user.id
    )

    if category_id:
        query = query.filter(Transaction.category_id == category_id)
    
    if start_date:
        query = query.filter(Transaction.created_at >= start_date)

    if end_date:
        end_of_day = datetime.combine(end_date, datetime.max.time())
        query = query.filter(Transaction.created_at <= end_of_day)

    transactions = query.order_by(Transaction.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["Date", "Description", "Category", "Type", "Amount (USD)"])

    # Write data rows
    for t in transactions:
        writer.writerow([
            t.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            t.description,
            t.category.name if t.category else "N/A",
            t.type,
            t.amount
        ])

    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=transactions.csv"})

@app.post("/api/transactions")
async def create_transaction(
    transaction_data: TransactionCreate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    # Verify category belongs to user
    category = db.query(Category).filter(
        Category.id == transaction_data.category_id,
        Category.user_id == user.id
    ).first()
    
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    transaction = Transaction(
        amount=transaction_data.amount,
        description=transaction_data.description,
        type=transaction_data.type,
        category_id=transaction_data.category_id,
        user_id=user.id
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    
    return {
        "id": transaction.id,
        "amount": transaction.amount,
        "description": transaction.description,
        "type": transaction.type,
        "category_id": transaction.category_id,
        "category_name": category.name,
        "created_at": transaction.created_at.isoformat()
    }

@app.put("/api/transactions/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    transaction_id: int,
    transaction_data: TransactionUpdate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    db_transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.user_id == user.id
    ).first()

    if not db_transaction:
        raise HTTPException(status_code=404, detail="Транзакция не найдена")

    # Verify category belongs to user
    category = db.query(Category).filter(
        Category.id == transaction_data.category_id,
        Category.user_id == user.id
    ).first()
    
    if not category:
        raise HTTPException(status_code=404, detail="Категория не найдена")

    db_transaction.amount = transaction_data.amount
    db_transaction.description = transaction_data.description
    db_transaction.type = transaction_data.type
    db_transaction.category_id = transaction_data.category_id
    if transaction_data.created_at:
        db_transaction.created_at = transaction_data.created_at

    db.commit()
    db.refresh(db_transaction)

    return db_transaction

@app.delete("/api/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.user_id == user.id
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    db.delete(transaction)
    db.commit()
    return {"message": "Transaction deleted successfully"}

# API Routes - Analytics
@app.get("/api/analytics")
async def get_analytics(user: User = Depends(require_auth), db: Session = Depends(get_db)):
    total_income = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id,
        Transaction.type == "income"
    ).scalar() or 0
    
    total_expenses = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id,
        Transaction.type == "expense"
    ).scalar() or 0
    
    balance = total_income - total_expenses
    
    expenses_by_category = db.query(
        Category.name,
        func.sum(Transaction.amount).label("total")
    ).join(Transaction).filter(
        Transaction.user_id == user.id,
        Transaction.type == "expense"
    ).group_by(Category.name).all()
    
    category_data = [{"category": cat, "amount": float(total)} for cat, total in expenses_by_category]
    
    return {
        "balance": float(balance),
        "total_income": float(total_income),
        "total_expenses": float(total_expenses),
        "expenses_by_category": category_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
