# Трекер расходов

Простое веб-приложение для отслеживания доходов и расходов.

## Как запустить

### Локально
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```bash
    uvicorn main:app --reload
    ```
3.  Откройте в браузере `http://localhost:8000`

### Развертывание (Deploy)
-   **Команда для запуска**: `uvicorn main:app --host 0.0.0.0 --port 8000`

## Test Account
-   **Email**: `test@example.com`
-   **Пароль**: `password123`