from alembic import context

connection = context.config.attributes.get('connection')
if connection is None:
    raise RuntimeError('Run migrations through python manage_database.py migrate')
context.configure(connection=connection, compare_type=True)
with context.begin_transaction():
    context.run_migrations()
