## RUN Alembic Migrations

### Configuration

```bash
cp allembic.ini.example allembic.ini
```

- Update the `allembic.ini` with your database credentials (`sqlalchemy.url`)

### (Optional) Create a new migration

```bash
alembic revision --autogenerate -m "Add ..."
```

### Upgrade the database

```bash
alembic upgrade head
```