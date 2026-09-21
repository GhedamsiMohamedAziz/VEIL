"""sku unique per organization, one active run per experiment

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


ACTIVE = sa.text("status IN ('queued', 'running')")
# The old global UNIQUE(sku) was declared inline and has no name. SQLite's batch
# mode reflects it under this convention; PostgreSQL named it itself.
NAMING = {"uq": "uq_%(table_name)s_%(column_0_name)s"}


def _global_sku_name() -> str:
    return "garments_sku_key" if op.get_bind().dialect.name == "postgresql" else "uq_garments_sku"


def upgrade() -> None:
    with op.batch_alter_table("experiment_runs") as batch_op:
        batch_op.create_index("uq_active_run", ["experiment_id"], unique=True,
                              sqlite_where=ACTIVE, postgresql_where=ACTIVE)

    # Autogenerate cannot see an unnamed constraint: the drop is written by hand.
    # Without it the global constraint would survive next to the per-organization one.
    with op.batch_alter_table("garments", naming_convention=NAMING) as batch_op:
        batch_op.drop_constraint(_global_sku_name(), type_="unique")
        batch_op.create_unique_constraint("uq_garment_org_sku", ["organization_id", "sku"])


def downgrade() -> None:
    with op.batch_alter_table("garments", naming_convention=NAMING) as batch_op:
        batch_op.drop_constraint("uq_garment_org_sku", type_="unique")
        batch_op.create_unique_constraint(_global_sku_name(), ["sku"])

    with op.batch_alter_table("experiment_runs") as batch_op:
        batch_op.drop_index("uq_active_run")
